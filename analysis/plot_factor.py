"""
Run: python -m analysis.plot_factor
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL_DIR = os.path.join(ROOT, "results", "final")
FIG_DIR = os.path.join(HERE, "output", "figures", "factor")

ALGOS = ["base", "MSR-population", "PSR-population"]
LABELS = {"base": "DE", "MSR-population": "MSR-pop", "PSR-population": "PSR-pop"}
COLORS = {"base": "#888888", "MSR-population": "#1f77b4", "PSR-population": "#d62728"}

DIMS = [10, 30]


def load_history(dim: int) -> dict[str, pd.DataFrame]:
    out = {}
    for a in ALGOS:
        path = os.path.join(FINAL_DIR, f"{a}_D{dim}_history.csv")
        out[a] = pd.read_csv(path, usecols=["func_id", "seed", "n_evals", "factor"])
    return out


def aggregate_factor(sub: pd.DataFrame):
    """Mediana + IQR współczynnika F po seedach na wspólnej siatce n_evals."""
    piv = sub.pivot_table(index="n_evals", columns="seed", values="factor")
    piv = piv.sort_index().ffill()
    x = piv.index.to_numpy(dtype=float)
    med = piv.median(axis=1).to_numpy()
    q25 = piv.quantile(0.25, axis=1).to_numpy()
    q75 = piv.quantile(0.75, axis=1).to_numpy()
    return x, med, q25, q75


def plot_func(hist: dict, dim: int, func_id: int, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for a in ALGOS:
        sub = hist[a][hist[a].func_id == func_id]
        x, med, q25, q75 = aggregate_factor(sub)
        if a != "base":
            ax.fill_between(x, q25, q75, color=COLORS[a], alpha=0.15)
        ax.plot(x, med, color=COLORS[a], lw=1.8, label=LABELS[a])

    ax.set_xlabel("liczba ewaluacji")
    ax.set_ylabel("mediana F")
    ax.set_title(f"f{func_id},  D = {dim}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"f{func_id}.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    for dim in DIMS:
        hist = load_history(dim)
        out_dir = os.path.join(FIG_DIR, f"D{dim}")
        funcs = sorted(hist["base"].func_id.unique())
        for func_id in funcs:
            plot_func(hist, dim, int(func_id), out_dir)
        print(f"[zapis] {out_dir}  ({len(funcs)} plików)")


if __name__ == "__main__":
    main()