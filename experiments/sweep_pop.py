import os
import argparse
import csv
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

from cecpy.benchmark import CECEdition, CECEvaluator

from src.cec_problem import CEC2017Problem
from src.de_base import DEBase
from src.de_msr import DEMSR
from src.de_psr import DEPSR
from experiments.config import (
    CR, FACTOR_INIT, FACTOR_MIN, FACTOR_MAX, FUNCTIONS, TOL, max_fes,
)

Config = namedtuple("Config", "method mode")

CONFIGS = {
    "base": Config("base", "-"),
    "MSR-population": Config("MSR", "population"),
    "PSR-population": Config("PSR", "population"),
}

BEST_QUANTILE = 0.45   # MSR-population
BEST_ZSTAR = 0.10      # PSR-population

NP_GRID = {
    10: [25, 50, 100],
    30: [50, 100, 150],
}

FIELDS = ["config", "method", "mode", "pop_size", "func_id", "dim",
          "seed", "best_fitness", "error", "n_evals", "solved", "seconds"]

_EVALUATORS: dict[int, CECEvaluator] = {}

def _problem(func_id: int, dim: int) -> CEC2017Problem:
    evaluator = _EVALUATORS.get(dim)
    if evaluator is None:
        evaluator = CECEvaluator(CECEdition.CEC2017, [dim])
        _EVALUATORS[dim] = evaluator
    return CEC2017Problem(func_id, dim, evaluator=evaluator)


def _build(cfg, np_size, problem, dim, seed):
    common = dict(
        func=problem, dim=dim, bounds=problem.bounds,
        population_size=np_size, crossover_probability=CR,
        max_evals=max_fes(dim), seed=seed, target=problem.optimum + TOL,
    )
    if cfg.method == "base":
        return DEBase(**common, factor=FACTOR_INIT)
    common.update(
        factor_init=FACTOR_INIT, factor_min=FACTOR_MIN, factor_max=FACTOR_MAX,
        comparison_mode=cfg.mode,
    )
    if cfg.method == "MSR":
        return DEMSR(**common, comparison_quantile=BEST_QUANTILE)
    else:
        return DEPSR(**common, z_star=BEST_ZSTAR)


def run_job(job):
    config_name, np_size, func_id, dim, seed = job
    cfg = CONFIGS[config_name]
    problem = _problem(func_id, dim)
    start = time.perf_counter()
    result = _build(cfg, np_size, problem, dim, seed).run()
    error = result.best_fitness - problem.optimum
    return {
        "config": config_name, "method": cfg.method, "mode": cfg.mode,
        "pop_size": np_size, "func_id": func_id, "dim": dim,
        "seed": seed, "best_fitness": result.best_fitness, "error": error,
        "n_evals": result.n_evals, "solved": int(error <= TOL),
        "seconds": round(time.perf_counter() - start, 3),
    }


def build_jobs(configs, dims, seeds, functions, np_grid):
    jobs = []
    for config_name in configs:
        for dim in dims:
            for np_size in np_grid[dim]:
                for func_id in functions:
                    for seed in seeds:
                        jobs.append((config_name, np_size, func_id, dim, seed))
    return jobs


def _fmt(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _stage_plan(jobs):
    order, totals = [], {}
    for config_name, *_ in jobs:
        if config_name not in totals:
            order.append(config_name)
            totals[config_name] = 0
        totals[config_name] += 1
    return order, totals


def main():
    ap = argparse.ArgumentParser(description="NP confirmation sweep")
    ap.add_argument("--out", default="results/sweep_np.csv")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--seeds", type=int, default=10, help="run seeds 1..N per (func, dim)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--log-every", type=int, default=1,
                    help="print one progress line every N finished runs (1 = every run)")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    functions = FUNCTIONS
    np_grid = NP_GRID
    if args.quick:
        functions = [1, 7, 21]
        seeds = [1, 2]
        np_grid = {10: [25, 100], 30: [100, 300]}

    missing = [d for d in args.dims if d not in np_grid]
    if missing:
        ap.error(f"no NP grid for dims {missing}; known dims: {sorted(np_grid)}")

    jobs = build_jobs(args.configs, args.dims, seeds, functions, np_grid)
    total = len(jobs)
    stage_order, stage_total = _stage_plan(jobs)

    print(f"{total} runs | {len(stage_order)} stages | dims={args.dims} "
          f"seeds={seeds[0]}..{seeds[-1]} workers={args.workers}", flush=True)
    print("NP grid: " + "  ".join(f"D{d}={np_grid[d]}" for d in args.dims), flush=True)
    print("stages: " + "  ".join(f"{s}({stage_total[s]})" for s in stage_order),
          flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start = time.perf_counter()
    stage_done = {s: 0 for s in stage_order}      # finished runs per stage
    stage_cpu = {s: 0.0 for s in stage_order}     # summed per-run duration (compute)
    prev_stage_end = start                        # wall-clock when the previous stage finished
    done = 0
    with open(args.out, "w", newline="") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        log_every = max(1, args.log_every)
        # pool.map preserves job order, so results arrive stage-by-stage.
        for job, row in zip(jobs, pool.map(run_job, jobs, chunksize=4)):
            writer.writerow(row)
            fh.flush()
            done += 1

            stage = job[0]
            stage_done[stage] += 1
            stage_cpu[stage] += row["seconds"]

            elapsed = time.perf_counter() - start
            eta = elapsed / done * (total - done)

            stage_complete = stage_done[stage] == stage_total[stage]
            if done % log_every == 0 or done == total or stage_complete:
                flag = "OK " if row["solved"] else "    "
                print(f"[{done}/{total} {100 * done / total:4.0f}%] {flag}"
                      f"{stage} f{row['func_id']:02d} D{row['dim']} s{row['seed']} "
                      f"NP={row['pop_size']:<3} | "
                      f"err={row['error']:.2e} {row['seconds']:.1f}s | "
                      f"{_fmt(elapsed)} elapsed, ETA {_fmt(eta)}", flush=True)

            if stage_complete:
                now = time.perf_counter()
                wall = now - prev_stage_end
                prev_stage_end = now
                finished = sum(1 for s in stage_order
                               if stage_done[s] == stage_total[s])
                remaining = [s for s in stage_order
                             if stage_done[s] < stage_total[s]]
                print(f"✓ stage done: {stage} | {stage_total[stage]} runs | "
                      f"{_fmt(wall)} wall, {_fmt(stage_cpu[stage])} CPU | "
                      f"{finished}/{len(stage_order)} stages | ETA {_fmt(eta)}",
                      flush=True)
                if remaining:
                    print(f"    remaining stages: {', '.join(remaining)}", flush=True)

    total_elapsed = time.perf_counter() - start
    print(f"done in {_fmt(total_elapsed)} | wrote {args.out}", flush=True)
    for s in stage_order:
        print(f"  {s}: {stage_total[s]} runs, {_fmt(stage_cpu[s])} CPU", flush=True)


if __name__ == "__main__":
    main()