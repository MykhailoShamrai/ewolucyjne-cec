"""
Run: python -m analysis.plot_boxplots
"""
from __future__ import annotations

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL_DIR = os.path.join(ROOT, "results", "final")
FIG_DIR = os.path.join(HERE, "output", "figures", "boxplots")

ALGOS = ["base", "MSR-population", "PSR-population"]
LABELS = {"base": "DE", "MSR-population": "MSR-pop", "PSR-population": "PSR-pop"}
COLORS = {"base": "#888888", "MSR-population": "#1f77b4", "PSR-population": "#d62728"}

DIMS = [10, 30]
FLOOR = 1e-9
TARGET = 1e-8


def load_summary() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(FINAL_DIR, "*_summary.csv")))
    if not files:
        raise FileNotFoundError(f"Brak *_summary.csv w {FINAL_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["error"] = df["error"].clip(lower=0.0)
    return df


def plot_func(df: pd.DataFrame, dim: int, func_id: int, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data, colors = [], []
    for a in ALGOS:
        err = df[(df.dim == dim) & (df.func_id == func_id) & (df.config == a)]["error"]
        data.append(np.maximum(err.to_numpy(dtype=float), FLOOR))
        colors.append(COLORS[a])
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.5),
                    medianprops=dict(color="black", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    allvals = np.concatenate(data)
    vmin, vmax = allvals.min(), allvals.max()
    if vmin <= TARGET * 10:
        ax.axhline(TARGET, color="green", lw=1.0, ls="--", alpha=0.7,
                   label="próg CEC")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_ylim(bottom=FLOOR * 0.5, top=vmax * 5)
    else:
        ax.set_ylim(bottom=vmin / 5, top=vmax * 5)
    ax.set_xticks(range(1, len(ALGOS) + 1))
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_ylabel("błąd")
    ax.set_title(f"f{func_id},  D = {dim}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"f{func_id}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    df = load_summary()
    for dim in DIMS:
        out_dir = os.path.join(FIG_DIR, f"D{dim}")
        funcs = sorted(df[df.dim == dim].func_id.unique())
        for func_id in funcs:
            plot_func(df, dim, int(func_id), out_dir)
        print(f"[zapis] {out_dir}  ({len(funcs)} plików)")


if __name__ == "__main__":
    main()