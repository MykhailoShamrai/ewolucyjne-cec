"""Flexible experiment launcher.

Run from the project root:
    python -m experiments.run_experiment
    python -m experiments.run_experiment --algos base MSR-population PSR-population --seeds 15
    python -m experiments.run_experiment --algos MSR-trial --funcs 18 --dims 30 --seed-list 3 7 9
"""
import os
import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor

from cecpy.benchmark import CECEdition, CECEvaluator

from src.cec_problem import CEC2017Problem
from src.de_base import DEBase
from src.de_msr import DEMSR
from src.de_psr import DEPSR
from experiments.config import (
    CR, FACTOR_INIT, FACTOR_MIN, FACTOR_MAX, FUNCTIONS, TOL, max_fes,
)

ALGO_SPECS = {
    "base":           ("base", "-",          {}),
    "MSR-population": ("MSR",  "population",  {"comparison_quantile": 0.45}),
    "MSR-trial":      ("MSR",  "trial",       {"comparison_quantile": 0.90}),
    "PSR-population": ("PSR",  "population",  {"z_star": 0.10}),
    "PSR-trial":      ("PSR",  "trial",       {"z_star": -0.50}),
}
DEFAULT_ALGOS = ["base", "MSR-population", "PSR-population"]

POP_SIZE = {
    "base":           {10: 100, 30: 100},
    "MSR-population": {10: 100, 30: 150},
    "MSR-trial":      {10: 100, 30: 150},
    "PSR-population": {10: 100, 30: 150},
    "PSR-trial":      {10: 100, 30: 150},
}

HISTORY_POINTS = {10: 200, 30: 300}
HISTORY_POINTS_FALLBACK = 250   # for dimensions not listed above

SUMMARY_FIELDS = ["config", "method", "mode", "func_id", "dim", "seed",
                  "pop_size", "best_fitness", "error", "n_evals", "solved", "seconds"]
HISTORY_FIELDS = ["config", "func_id", "dim", "seed", "n_evals", "best_error", "factor"]

_EVALUATORS: dict[int, CECEvaluator] = {}


def _problem(func_id: int, dim: int) -> CEC2017Problem:
    evaluator = _EVALUATORS.get(dim)
    if evaluator is None:
        evaluator = CECEvaluator(CECEdition.CEC2017, [dim])
        _EVALUATORS[dim] = evaluator
    return CEC2017Problem(func_id, dim, evaluator=evaluator)


def _build(algo, pop_size, problem, dim, seed):
    method, mode, hp = ALGO_SPECS[algo]
    common = dict(
        func=problem, dim=dim, bounds=problem.bounds,
        population_size=pop_size, crossover_probability=CR,
        max_evals=max_fes(dim), seed=seed, target=problem.optimum + TOL,
    )
    if method == "base":
        return DEBase(**common, factor=FACTOR_INIT)
    common.update(
        factor_init=FACTOR_INIT, factor_min=FACTOR_MIN, factor_max=FACTOR_MAX,
        comparison_mode=mode,
    )
    if method == "MSR":
        return DEMSR(**common, **hp)
    else:
        return DEPSR(**common, **hp)


def stride_for(dim, pop_size, explicit, cap):
    """Fixed stride for a dimension: same for every seed, so traces stay aligned.

    Computed from the FULL expected generation count (budget / NP), not from an
    individual (possibly early-stopped) run, so the sampled generations -- and thus
    the n_evals checkpoints -- coincide across seeds. `cap` is the target number of
    points to keep per run; an explicit stride (>0) overrides it.
    """
    if explicit > 0:
        return explicit
    expected_gens = max_fes(dim) // pop_size
    return max(1, math.ceil(expected_gens / cap))


def run_job(job):
    algo, func_id, dim, seed, pop_size, stride, save_history = job
    problem = _problem(func_id, dim)
    start = time.perf_counter()
    result = _build(algo, pop_size, problem, dim, seed).run()
    seconds = round(time.perf_counter() - start, 3)

    method, mode, _ = ALGO_SPECS[algo]
    error = result.best_fitness - problem.optimum
    summary = {
        "config": algo, "method": method, "mode": mode,
        "func_id": func_id, "dim": dim, "seed": seed, "pop_size": pop_size,
        "best_fitness": result.best_fitness, "error": error,
        "n_evals": result.n_evals, "solved": int(error <= TOL), "seconds": seconds,
    }
    if not save_history:
        return summary, None

    opt = problem.optimum
    best_f = result.history_best_fitness
    factor = result.history_factor
    n_evals = result.history_n_evals
    idx = list(range(0, len(best_f), stride))
    if idx[-1] != len(best_f) - 1:
        idx.append(len(best_f) - 1)
    history = [{
        "config": algo, "func_id": func_id, "dim": dim, "seed": seed,
        "n_evals": n_evals[i], "best_error": best_f[i] - opt, "factor": factor[i],
    } for i in idx]
    return summary, history


def _fmt(secs):
    """Human-readable duration: '1h05m', '7m12s' or '9s'."""
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ExperimentRunner:
    """Runs a grid of (algorithm, function, dimension, seed) and writes per-algo files.

    Each (algorithm, dimension) pair gets its own pair of files in `out_dir`:
        <algo>_D<dim>_summary.csv   -- one row per run (final error, n_evals, time)
        <algo>_D<dim>_history.csv   -- down-sampled convergence trace (best_error, factor)
    """

    def __init__(self, algos=None, seeds=None, funcs=None, dims=None,
                 pop_size=None, out_dir="results/final", history_stride=0,
                 history_points=None, save_history=True, workers=None,
                 log_every=20):
        self.algos = list(algos) if algos else list(DEFAULT_ALGOS)
        self.seeds = list(seeds) if seeds else list(range(1, 16))
        self.funcs = list(funcs) if funcs else list(FUNCTIONS)
        self.dims = list(dims) if dims else [10, 30]
        self.pop_size_override = pop_size   # None -> use the tuned POP_SIZE table
        self.out_dir = out_dir
        self.history_stride = history_stride
        self.history_points = history_points
        self.save_history = save_history
        self.workers = workers or os.cpu_count()
        self.log_every = max(1, log_every)

    def _pop_size(self, algo, dim):
        if self.pop_size_override is not None:
            return self.pop_size_override
        return POP_SIZE[algo][dim]

    def _jobs(self):
        jobs = []
        for algo in self.algos:
            for dim in self.dims:
                pop_size = self._pop_size(algo, dim)
                cap = (self.history_points if self.history_points is not None
                       else HISTORY_POINTS.get(dim, HISTORY_POINTS_FALLBACK))
                stride = stride_for(dim, pop_size, self.history_stride, cap)
                for func_id in self.funcs:
                    for seed in self.seeds:
                        jobs.append((algo, func_id, dim, seed, pop_size,
                                     stride, self.save_history))
        return jobs

    def run(self):
        jobs = self._jobs()
        total = len(jobs)
        os.makedirs(self.out_dir, exist_ok=True)

        handles = []          # all open files, closed in finally
        writers = {}          # (algo, dim) -> (summary_writer, history_writer, summary_fh, history_fh)

        def writers_for(algo, dim):
            key = (algo, dim)
            if key not in writers:
                sp = os.path.join(self.out_dir, f"{algo}_D{dim}_summary.csv")
                sf = open(sp, "w", newline="")
                sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
                sw.writeheader()
                handles.append(sf)
                hw = hf = None
                if self.save_history:
                    hp = os.path.join(self.out_dir, f"{algo}_D{dim}_history.csv")
                    hf = open(hp, "w", newline="")
                    hw = csv.DictWriter(hf, fieldnames=HISTORY_FIELDS)
                    hw.writeheader()
                    handles.append(hf)
                writers[key] = (sw, hw, sf, hf)
            return writers[key]

        np_desc = self.pop_size_override if self.pop_size_override is not None else "tuned"
        print(f"{total} runs | algos={self.algos} | dims={self.dims} | "
              f"funcs={len(self.funcs)} | seeds={len(self.seeds)} | "
              f"NP={np_desc} | history={'on' if self.save_history else 'off'} | "
              f"workers={self.workers}", flush=True)

        start = time.perf_counter()
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                for job, (summary, history) in zip(
                        jobs, pool.map(run_job, jobs, chunksize=4)):
                    sw, hw, sf, hf = writers_for(job[0], job[2])
                    sw.writerow(summary)
                    sf.flush()
                    if hw is not None and history is not None:
                        hw.writerows(history)
                        hf.flush()
                    done += 1

                    if done % self.log_every == 0 or done == total:
                        elapsed = time.perf_counter() - start
                        eta = elapsed / done * (total - done)
                        flag = "OK " if summary["solved"] else "   "
                        print(f"[{done}/{total} {100 * done / total:4.0f}%] {flag}"
                              f"{summary['config']:>14} f{summary['func_id']:02d} "
                              f"D{summary['dim']} s{summary['seed']} | "
                              f"err={summary['error']:.2e} | "
                              f"{_fmt(elapsed)} elapsed, ETA {_fmt(eta)}", flush=True)
        finally:
            for fh in handles:
                fh.close()

        kinds = "{summary,history}" if self.save_history else "summary"
        print(f"done in {_fmt(time.perf_counter() - start)} | "
              f"wrote {self.out_dir}/<algo>_D<dim>_{kinds}.csv", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Flexible experiment launcher")
    ap.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS, choices=list(ALGO_SPECS),
                    help="algorithm names to run (default: the three final algorithms)")
    ap.add_argument("--seeds", type=int, default=15, help="run seeds 1..N")
    ap.add_argument("--seed-list", type=int, nargs="+",
                    help="explicit seeds (overrides --seeds), e.g. --seed-list 3 7 9")
    ap.add_argument("--funcs", type=int, nargs="+", default=FUNCTIONS,
                    help="function ids (default: all CEC 2017 except f2)")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--pop-size", type=int, default=None,
                    help="override NP for all runs (default: tuned per algorithm/dimension)")
    ap.add_argument("--out-dir", default="results/final")
    ap.add_argument("--history-points", type=int, default=None,
                    help="target points/run for all dims (default: per-dim %s)" % HISTORY_POINTS)
    ap.add_argument("--history-stride", type=int, default=0,
                    help="force every k-th generation (overrides --history-points)")
    ap.add_argument("--no-history", action="store_true",
                    help="write only summary files (skip the convergence traces)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    seeds = args.seed_list if args.seed_list else list(range(1, args.seeds + 1))
    ExperimentRunner(
        algos=args.algos, seeds=seeds, funcs=args.funcs, dims=args.dims,
        pop_size=args.pop_size, out_dir=args.out_dir,
        history_stride=args.history_stride, history_points=args.history_points,
        save_history=not args.no_history,
        workers=args.workers, log_every=args.log_every,
    ).run()


if __name__ == "__main__":
    main()