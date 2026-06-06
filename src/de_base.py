import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DEResult:
    best_x: np.ndarray
    best_fitness: float
    history_best_fitness: list[float] = field(default_factory=list)
    history_mean_fitness: list[float] = field(default_factory=list)
    history_median_fitness: list[float] = field(default_factory=list)
    history_worst_fitness: list[float] = field(default_factory=list)
    history_std_fitness: list[float] = field(default_factory=list)
    history_factor: list[float] = field(default_factory=list)
    history_n_evals: list[int] = field(default_factory=list)
    n_evals: int = 0


class DEBase:

    def __init__(self, func: Callable, dim: int, bounds: tuple[float, float],
                 population_size: int = 100, factor: float = 0.5, crossover_probability: float = 0.9,
                 max_evals: int = 100_000, seed: int | None = None, target: float | None = None):
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.population_size = population_size
        self.factor = factor
        self.CR = crossover_probability
        self.max_evals = max_evals
        self.target = target  # stop early once best fitness <= target (CEC: optimum + 1e-8)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _init_population(self) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(self.population_size, self.dim))

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _mutate(self, population: np.ndarray, idx: int, factor: float) -> np.ndarray:
        ids = list(range(self.population_size))
        ids.remove(idx)
        r1, r2, r3 = self.rng.choice(ids, 3, replace=False)
        return population[r1] + factor * (population[r2] - population[r3])

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        trial = target.copy()
        j_rand = self.rng.integers(0, self.dim)
        for j in range(self.dim):
            if self.rng.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        # Use the objective's batch interface (CEC2017Problem.evaluate) when available
        batch = getattr(self.func, "evaluate", None)
        if batch is not None:
            return np.asarray(batch(X), dtype=float)
        return np.array([self.func(x) for x in X], dtype=float)

    def get_factor(self) -> float:
        return self.factor

    def update_factor(self, generation: int, parent_fitness: np.ndarray,
                      trial_fitness: np.ndarray, new_fitness: np.ndarray) -> None:
        """No-op in the base algorithm; the factor stays constant."""

    def run(self) -> DEResult:
        population = self._init_population()
        fitness = self._evaluate(population)
        n_evals = self.population_size

        result = DEResult(best_x=np.empty(self.dim), best_fitness=float("inf"))

        def _log(factor: float) -> None:
            result.history_best_fitness.append(float(np.min(fitness)))
            result.history_mean_fitness.append(float(np.mean(fitness)))
            result.history_median_fitness.append(float(np.median(fitness)))
            result.history_worst_fitness.append(float(np.max(fitness)))
            result.history_std_fitness.append(float(np.std(fitness)))
            result.history_factor.append(factor)
            result.history_n_evals.append(n_evals)

        _log(self.get_factor())

        target = self.target if self.target is not None else -np.inf
        generation = 0
        while n_evals < self.max_evals and float(np.min(fitness)) > target:
            generation += 1
            current_factor = self.get_factor()

            # Build every trial first (RNG order identical to the scalar loop),
            # then evaluate the whole batch in one call. A partial last generation
            # only builds/evaluates the k trials the budget still allows.
            k = min(self.population_size, self.max_evals - n_evals)
            trials = population.copy()
            for i in range(k):
                mutant = self._clip(self._mutate(population, i, current_factor))
                trials[i] = self._clip(self._crossover(population[i], mutant))

            trial_fitness = fitness.copy()  # neutral default for the unevaluated tail
            trial_fitness[:k] = self._evaluate(trials[:k])
            n_evals += k

            take = trial_fitness <= fitness
            take[k:] = False  # untouched tail never replaces its parent
            new_population = np.where(take[:, None], trials, population)
            new_fitness = np.where(take, trial_fitness, fitness)

            self.update_factor(generation, fitness, trial_fitness, new_fitness)
            population = new_population
            fitness = new_fitness

            _log(current_factor)

        best_idx = np.argmin(fitness)
        result.best_x = population[best_idx]
        result.best_fitness = float(fitness[best_idx])
        result.n_evals = n_evals
        return result