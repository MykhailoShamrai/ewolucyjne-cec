import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DEResult:
    best_x: np.ndarray
    best_f: float
    history_best_factor: list[float] = field(default_factory=list)
    history_mean_factor: list[float] = field(default_factory=list)
    history_factor: list[float] = field(default_factory=list)
    n_evals: int = 0


class DEBase:
    # TODO: zmienić że liczba max_evals zależy od wymiarowości oraz liczność populacji jest zależna od wymiary
    # TODO: https://algorithmafternoon.com/differential/de_rand_1_bin/
    def __init__(self, func: Callable, dim: int, bounds: tuple[float, float],
                 population_size: int = 100, factor: float = 0.5, crossover_probability: float = 0.9,
                 max_evals: int = 100_000, seed: int | None = None):
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.population_size = population_size
        self.F = factor
        self.CR = crossover_probability
        self.max_evals = max_evals
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

    # TODO: Tu trzeba pomyśleć jak to ładnie zaprojektować, żeby mieć jakąś ładną architekturę i nie pisać 3 razy to samo
    def get_factor(self) -> float:
        return self.F

    def update_factor(self, generation: int, population: np.ndarray, fitness: np.ndarray,
                      new_population: np.ndarray, new_fitness: np.ndarray) -> None:
        # TODO: To będzie tylko w tych naszych modyfikacjach
        pass

    def run(self) -> DEResult:
        population = self._init_population()
        fitness = np.array([self.func(v) for v in population])
        n_evals = self.population_size

        history_best_factor = [float(np.min(fitness))]
        history_mean_factor = [float(np.mean(fitness))]
        history_factor = [self.get_factor()]

        generation = 0
        while n_evals < self.max_evals:
            generation += 1
            current_factor = self.get_factor()

            new_population = np.empty_like(population)
            new_fitness = np.empty(self.population_size)

            for i in range(self.population_size):
                if n_evals >= self.max_evals:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    continue
                mutant = self._clip(self._mutate(population, i, current_factor))
                trial = self._crossover(population[i], mutant)
                trial = self._clip(trial)
                f_trial = self.func(trial)
                n_evals += 1

                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            self.update_factor(generation, population, fitness, new_population, new_fitness)
            population = new_population
            fitness = new_fitness

            history_best_factor.append(float(np.min(fitness)))
            history_mean_factor.append(float(np.mean(fitness)))
            history_factor.append(current_factor)

        best_idx = np.argmin(fitness)
        return DEResult(
            best_x=population[best_idx],
            best_f=float(fitness[best_idx]),
            history_best_factor=history_best_factor,
            history_mean_factor=history_mean_factor,
            history_factor=history_factor,
            n_evals=n_evals,
        )
