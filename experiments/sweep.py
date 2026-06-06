"""Hyperparameter sweep for the MSR/PSR mutation-factor adaptations.

Runs each of the 4 configurations (MSR/PSR x trial/population) over a grid of its
hyperparameter (comparison_quantile for MSR, z_star for PSR), on every CEC 2017
function, repeated over several seeds, under the CEC protocol (MaxFES = 10^4*D,
NP = 10*D, terminate on error < 1e-8). One CSV row per run.

Run from the project root:
    python -m experiments.sweep --quick
    python -m experiments.sweep
    python -m experiments.sweep --dims 10 30 --out results/sweep_full.csv
"""
import os
import argparse
import csv
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

from cecpy.benchmark import CECEdition, CECEvaluator

from src.cec_problem import CEC2017Problem
from src.de_msr import DEMSR
from src.de_psr import DEPSR
from experiments.config import (
    CR, FACTOR_INIT, FACTOR_MIN, FACTOR_MAX, FUNCTIONS, TOL,
    QUANTILE_GRID, ZSTAR_GRID_TRIAL, ZSTAR_GRID_POPULATION, max_fes, pop_size,
)

Config = namedtuple("Config", "method mode hp_name grid")

CONFIGS = {
    "MSR-trial": Config("MSR", "trial", "comparison_quantile", QUANTILE_GRID),
    "MSR-population": Config("MSR", "population", "comparison_quantile", QUANTILE_GRID),
    "PSR-trial": Config("PSR", "trial", "z_star", ZSTAR_GRID_TRIAL),
    "PSR-population": Config("PSR", "population", "z_star", ZSTAR_GRID_POPULATION),
}

FIELDS = ["config", "method", "mode", "hp_name", "hp_value", "func_id", "dim",
          "seed", "best_fitness", "error", "n_evals", "solved", "seconds"]

_EVALUATORS: dict[int, CECEvaluator] = {}


def _problem(func_id: int, dim: int) -> CEC2017Problem:
    evaluator = _EVALUATORS.get(dim)
    if evaluator is None:
        evaluator = CECEvaluator(CECEdition.CEC2017, [dim])
        _EVALUATORS[dim] = evaluator
    return CEC2017Problem(func_id, dim, evaluator=evaluator)


# hp = the swept hyperparameter value (comparison_quantile for MSR, z_star for PSR)
def _build(cfg, hp, problem, dim, seed):
    common = dict(
        func=problem, dim=dim, bounds=problem.bounds,
        population_size=pop_size(dim), crossover_probability=CR,
        max_evals=max_fes(dim), seed=seed, target=problem.optimum + TOL,
        factor_init=FACTOR_INIT, factor_min=FACTOR_MIN, factor_max=FACTOR_MAX,
        comparison_mode=cfg.mode,
    )
    if cfg.method == "MSR":
        return DEMSR(**common, comparison_quantile=hp)
    else:
        return DEPSR(**common, z_star=hp)


def run_job(job):
    config_name, hp, func_id, dim, seed = job
    cfg = CONFIGS[config_name]
    problem = _problem(func_id, dim)
    start = time.perf_counter()
    result = _build(cfg, hp, problem, dim, seed).run()
    error = result.best_fitness - problem.optimum
    return {
        "config": config_name, "method": cfg.method, "mode": cfg.mode,
        "hp_name": cfg.hp_name, "hp_value": hp, "func_id": func_id, "dim": dim,
        "seed": seed, "best_fitness": result.best_fitness, "error": error,
        "n_evals": result.n_evals, "solved": int(error <= TOL),
        "seconds": round(time.perf_counter() - start, 3),
    }


def build_jobs(configs, dims, seeds, functions, grid_override=None):
    jobs = []
    for config_name in configs:
        cfg = CONFIGS[config_name]
        grid = grid_override.get(cfg.hp_name, cfg.grid) if grid_override else cfg.grid
        for hp in grid:
            for func_id in functions:
                for dim in dims:
                    for seed in seeds:
                        jobs.append((config_name, hp, func_id, dim, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="MSR/PSR hyperparameter sweep")
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--dims", type=int, nargs="+", default=[10])
    ap.add_argument("--seeds", type=int, default=15, help="run seeds 1..N per (func, dim)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    functions = FUNCTIONS
    grid_override = None
    if args.quick:
        functions = [1, 7, 21]
        seeds = [1, 2]
        grid_override = {"comparison_quantile": [0.3, 0.5, 0.7],
                         "z_star": [-0.2, 0.0, 0.3]}

    jobs = build_jobs(args.configs, args.dims, seeds, functions, grid_override)
    total = len(jobs)
    print(f"{total} runs | dims={args.dims} seeds={seeds[0]}..{seeds[-1]} "
          f"workers={args.workers}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start = time.perf_counter()
    with open(args.out, "w", newline="") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        done = 0
        for row in pool.map(run_job, jobs, chunksize=4):
            writer.writerow(row)
            fh.flush()
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.perf_counter() - start
                eta = elapsed / done * (total - done)
                print(f"  {done}/{total} | elapsed {elapsed / 60:.1f}m | "
                      f"ETA {eta / 3600:.1f}h", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
