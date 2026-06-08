"""Analyse the population-size sweep and pick the best pop_size per (config, dim).

Run from the project root:
    python -m analysis.pop_size_analysis
    python -m analysis.pop_size_analysis results/sweep_np.csv --top 3
"""
import argparse

from analysis.sweep_analysis import load, report

PARAM = "pop_size"


def main():
    ap = argparse.ArgumentParser(description="Population-size sweep analysis")
    ap.add_argument("csv", nargs="*", default=["results/sweep_pop.csv"],
                    help="NP sweep CSV file(s) (default: results/sweep_pop.csv)")
    ap.add_argument("--top", type=int, default=3, help="top-N per function")
    ap.add_argument("--out-dir", default="results/analysis/pop_size")
    args = ap.parse_args()
    report(load(args.csv), PARAM, args.top, args.out_dir)


if __name__ == "__main__":
    main()