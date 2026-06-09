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
    ap = argparse.ArgumentParser(description="MSR/PSR hyperparameter sweep")
    ap.add_argument("--out", default="results/sweep_10.csv")
    ap.add_argument("--dims", type=int, nargs="+", default=[10])
    ap.add_argument("--seeds", type=int, default=15, help="run seeds 1..N per (func, dim)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--log-every", type=int, default=1,
                    help="print one progress line every N finished runs (1 = every run)")
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
    stage_order, stage_total = _stage_plan(jobs)

    print(f"{total} runs | {len(stage_order)} stages | dims={args.dims} "
          f"seeds={seeds[0]}..{seeds[-1]} workers={args.workers}", flush=True)
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
                      f"{row['hp_name']}={row['hp_value']:+.2f} | "
                      f"err={row['error']:.2e} {row['seconds']:.1f}s | "
                      f"{_fmt(elapsed)} elapsed, ETA {_fmt(eta)}", flush=True)

            if stage_complete:
                now = time.perf_counter()
                wall = now - prev_stage_end          # time since the previous stage finished
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
