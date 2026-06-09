"""python -m analysis.final_stats"""
from __future__ import annotations

import glob
import os
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL_DIR = os.path.join(ROOT, "results", "final")
TABLES_DIR = os.path.join(HERE, "output", "tables")

ALGO_LABEL = {
    "base": "DE",
    "MSR-population": "MSR-pop",
    "PSR-population": "PSR-pop",
    "MSR-trial": "MSR-trial",
    "PSR-trial": "PSR-trial",
}
DIMS = [10, 30]
MAIN_ALGOS = ["base", "MSR-population", "PSR-population"]
TIE_TOL = 1e-8


def load_summary() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(FINAL_DIR, "*_summary.csv")))
    if not files:
        raise FileNotFoundError(f"Brak *_summary.csv w {FINAL_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["error"] = df["error"].clip(lower=0.0)
    return df


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["config", "dim", "func_id"])
    tab = g["error"].agg(
        median="median", mean="mean", std="std", best="min", worst="max"
    )
    tab["solved"] = g["solved"].sum()
    tab["runs"] = g["error"].size()
    tab = tab.reset_index()
    tab["algo"] = tab["config"].map(ALGO_LABEL)
    return tab.sort_values(["dim", "config", "func_id"]).reset_index(drop=True)


def wilcoxon_pair(df: pd.DataFrame, a: str, b: str, dim: int, agg: str) -> dict:
    """h1 A is better than B accordingly to agg. If agg is less tnan A is better
    """
    rep = (
        df[df.dim == dim]
        .groupby(["config", "func_id"])["error"]
        .agg(agg)
        .unstack("config")
    )
    xa, xb = rep[a], rep[b]
    n_ties = int((xa == xb).sum())
    stat, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="less")
    return {
        "dim": dim,
        "agg": agg,
        "hypothesis": f"{ALGO_LABEL[a]} < {ALGO_LABEL[b]}",
        "A": ALGO_LABEL[a],
        "B": ALGO_LABEL[b],
        "A_wins": int((xa < xb).sum()),
        "B_wins": int((xb < xa).sum()),
        "ties": n_ties,
        "A_err": float(xa.median()),
        "B_err": float(xb.median()),
        "W_stat": float(stat),
        "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
    }


def best_of_table(df: pd.DataFrame, dim: int, algos: list[str]):
    """Best-of-runs: zwycięzca wg najniższego pojedynczego błędu z 30 przebiegów."""
    best = (
        df[(df.dim == dim) & (df.config.isin(algos))]
        .groupby(["func_id", "config"])["error"]
        .min()
        .unstack("config")[algos]
    )
    rows, sole_wins, shared_wins = [], {a: 0 for a in algos}, {a: 0 for a in algos}
    for func_id, r in best.iterrows():
        lo = r.min()
        winners = [a for a in algos if r[a] - lo <= TIE_TOL]
        is_tie = len(winners) > 1
        for a in winners:
            shared_wins[a] += 1
            if not is_tie:
                sole_wins[a] += 1
        rows.append({
            "dim": dim,
            "func_id": int(func_id),
            **{ALGO_LABEL[a]: float(r[a]) for a in algos},
            "winner": "tie" if is_tie else ALGO_LABEL[winners[0]],
        })
    per_func = pd.DataFrame(rows)
    counts = pd.DataFrame({
        "dim": dim,
        "algo": [ALGO_LABEL[a] for a in algos],
        "best_of_wins": [shared_wins[a] for a in algos],
        "sole_wins": [sole_wins[a] for a in algos],
    })
    return per_func, counts


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = load_summary()

    desc = descriptive_table(df)
    desc_path = os.path.join(TABLES_DIR, "descriptive_stats.csv")
    desc.to_csv(desc_path, index=False)
    print(f"[zapis] {desc_path}  ({len(desc)} wierszy)")

    pairs = [
        ("MSR-population", "base"),            # H1
        ("PSR-population", "base"),            # H2
        ("MSR-population", "PSR-population"),  # H3
    ]
    rows = [
        wilcoxon_pair(df, a, b, d, agg)
        for d in DIMS
        for agg in ("median", "mean")
        for (a, b) in pairs
    ]
    wil = pd.DataFrame(rows)
    wil_path = os.path.join(TABLES_DIR, "wilcoxon.csv")
    wil.to_csv(wil_path, index=False)
    print(f"[zapis] {wil_path}")
    print()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(wil.to_string(index=False))

    per_func_all, counts_all = [], []
    for d in DIMS:
        pf, cnt = best_of_table(df, d, MAIN_ALGOS)
        per_func_all.append(pf)
        counts_all.append(cnt)
    pf = pd.concat(per_func_all, ignore_index=True)
    counts = pd.concat(counts_all, ignore_index=True)
    pf.to_csv(os.path.join(TABLES_DIR, "best_of_per_function.csv"), index=False)
    counts.to_csv(os.path.join(TABLES_DIR, "best_of_wins.csv"), index=False)
    print(f"\n[zapis] {os.path.join(TABLES_DIR, 'best_of_wins.csv')}")
    print("\nBest-of-30 (zwycięstwa per algorytm; best_of_wins liczy też remisy):")
    with pd.option_context("display.width", 160):
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
