import numpy as np

from src.de_base import DEBase


class DEMSR(DEBase):
    """DE/rand/1/bin with Median Success Rule adaptation of mutation factor."""

    def __init__(
        self,
        *args,
        F_init: float = 0.5,
        F_min: float = 0.1,
        F_max: float = 1.0,
        target_success: float = 0.2,
        damping: float = 1.0,
        **kwargs,
    ):
        if F_min <= 0.0:
            raise ValueError("F_min must be positive")
        if F_max < F_min:
            raise ValueError("F_max must be >= F_min")
        if not (0.0 <= target_success <= 1.0):
            raise ValueError("target_success must be in [0, 1]")
        if damping <= 0.0:
            raise ValueError("damping must be positive")

        kwargs["factor"] = F_init
        super().__init__(*args, **kwargs)

        self.F_min = F_min
        self.F_max = F_max
        self.target_success = target_success
        self.damping = damping
        self.F = float(np.clip(F_init, self.F_min, self.F_max))

    def get_F(self) -> float:
        return self.F

    def get_factor(self) -> float:
        return self.get_F()

    def update_F(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        new_population: np.ndarray,
        new_fitness: np.ndarray,
    ) -> None:
        del generation, population, new_population
        median_parent_fitness = float(np.median(fitness))
        success_rate = float(np.mean(new_fitness < median_parent_fitness))
        self.F *= float(np.exp(self.damping * (success_rate - self.target_success)))
        self.F = float(np.clip(self.F, self.F_min, self.F_max))

    def update_factor(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        new_population: np.ndarray,
        new_fitness: np.ndarray,
    ) -> None:
        self.update_F(generation, population, fitness, new_population, new_fitness)
