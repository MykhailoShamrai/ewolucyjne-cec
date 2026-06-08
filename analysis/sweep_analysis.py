"""Analyse one or more sweep CSVs and pick the best hyperparameter per algorithm.

Run from the project root:
    python -m analysis.sweep_analysis results/sweep_30.csv
    python -m analysis.sweep_analysis results/sweep_30.csv results/sweep_10.csv --top 3
    python -m analysis.sweep_analysis results/sweep_*.csv --param-col np --out-dir results/analysis
"""
import argparse
import os

import pandas as pd


def load(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    identity = [c for c in df.columns if c != "seconds"]
    df = df.drop_duplicates(subset=identity, ignore_index=True)
    return df


def per_seed_aggregate(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    group = ["config", "method", "mode", "dim", "func_id", param_col]
    agg = df.groupby(group, as_index=False).agg(
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        std_error=("error", lambda s: s.std(ddof=0)),
        best_error=("error", "min"),
        worst_error=("error", "max"),
        solved_rate=("solved", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return agg


def per_function_top(agg: pd.DataFrame, param_col: str, top: int) -> pd.DataFrame:
    agg = agg.copy()
    keys = ["config", "dim", "func_id"]
    agg["rank"] = agg.groupby(keys)["mean_error"].rank(method="min")
    top_df = agg[agg["rank"] <= top].sort_values(keys + ["rank"])
    cols = keys + ["rank", param_col, "mean_error", "median_error",
                   "solved_rate", "n_seeds"]
    return top_df[cols].reset_index(drop=True)


def recommended_param(agg: pd.DataFrame, param_col: str) -> pd.DataFrame:
    """One recommended parameter value per (config, dim), via mean rank over funcs"""
    agg = agg.copy()
    func_keys = ["config", "dim", "func_id"]
    agg["rank"] = agg.groupby(func_keys)["mean_error"].rank(method="min")
    agg["is_best"] = (agg["rank"] == 1.0).astype(int)

    param_keys = ["config", "method", "mode", "dim", param_col]
    summary = agg.groupby(param_keys, as_index=False).agg(
        mean_rank=("rank", "mean"),
        wins=("is_best", "sum"),
        n_funcs=("func_id", "nunique"),
        mean_solved_rate=("solved_rate", "mean"),
    )
    summary["best_rank"] = summary.groupby(["config", "dim"])["mean_rank"].rank(
        method="min")
    summary = summary.sort_values(["config", "dim", "mean_rank"])
    return summary.reset_index(drop=True)


def _fmt_block(title: str, body: str) -> str:
    line = "=" * len(title)
    return f"\n{line}\n{title}\n{line}\n{body}"


def main():
    ap = argparse.ArgumentParser(description="Analyse sweep CSV(s); pick best hp")
    ap.add_argument("csv", nargs="+", help="one or more sweep CSV files")
    ap.add_argument("--param-col", default="hp_value",
                    help="the swept parameter column (hp_value, or np for sweep #2)")
    ap.add_argument("--top", type=int, default=3, help="top-N per function")
    ap.add_argument("--out-dir", default="results/analysis")
    args = ap.parse_args()

    df = load(args.csv)
    if args.param_col not in df.columns:
        raise SystemExit(f"--param-col {args.param_col!r} not in CSV columns: "
                         f"{list(df.columns)}")

    agg = per_seed_aggregate(df, args.param_col)
    top_df = per_function_top(agg, args.param_col, args.top)
    rec = recommended_param(agg, args.param_col)

    os.makedirs(args.out_dir, exist_ok=True)
    agg.to_csv(os.path.join(args.out_dir, "per_param_stats.csv"), index=False)
    top_df.to_csv(os.path.join(args.out_dir, "per_function_top.csv"), index=False)
    rec.to_csv(os.path.join(args.out_dir, "recommended_param.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3g}")

    print(f"loaded {len(df)} runs | configs={sorted(df.config.unique())} | "
          f"dims={sorted(df.dim.unique())} | "
          f"funcs={df.func_id.nunique()} | seeds={df.seed.nunique()} | "
          f"param={args.param_col}")

    # Recommended single value per (config, dim): the row with best_rank == 1.
    best = rec[rec["best_rank"] == 1.0].copy()
    show = ["config", "dim", args.param_col, "mean_rank", "wins", "n_funcs",
            "mean_solved_rate"]
    print(_fmt_block("RECOMMENDED hyperparameter per (config, dim)  [mean-rank]",
                     best[show].to_string(index=False)))

    # Runner-up context: top-3 values by mean rank per (config, dim).
    runners = rec[rec["best_rank"] <= 3]
    print(_fmt_block("Top-3 by mean rank (for context / tie-breaks)",
                     runners[show].to_string(index=False)))

    print(_fmt_block(f"Per-function top-{args.top} written to",
                     os.path.join(args.out_dir, "per_function_top.csv")))
    print(f"\nwrote: {args.out_dir}/{{per_param_stats,per_function_top,"
          f"recommended_param}}.csv")


if __name__ == "__main__":
    main()