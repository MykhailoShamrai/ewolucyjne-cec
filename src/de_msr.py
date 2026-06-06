import numpy as np
from typing import Callable

from src.de_base import DEBase
from src.success_rule import SuccessRuleAdapter


class DEMSR(DEBase, SuccessRuleAdapter):

    def __init__(
        self,
        func: Callable,
        dim: int,
        bounds: tuple[float, float],
        population_size: int = 100,
        crossover_probability: float = 0.9,
        max_evals: int = 100_000,
        seed: int | None = None,
        target: float | None = None,
        factor_init: float = 0.5,
        factor_min: float = 0.1,
        factor_max: float = 1.0,
        comparison_quantile: float = 0.7,
        comparison_mode: str = "trial",
        c_sigma: float = 0.3,
        d_sigma: float | None = None,
    ):
        if not (0.0 < comparison_quantile <= 1.0):
            raise ValueError("comparison_quantile must be in (0, 1]")
        if comparison_mode not in ("trial", "population"):
            raise ValueError("comparison_mode must be 'trial' or 'population'")

        super().__init__(
            func=func,
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            factor=factor_init,
            crossover_probability=crossover_probability,
            max_evals=max_evals,
            seed=seed,
            target=target,
        )

        if d_sigma is None:
            d_sigma = 2.0 * (self.dim - 1) / self.dim
        self._init_adapter(factor_init, factor_min, factor_max, c_sigma, d_sigma)
        self.comparison_quantile = comparison_quantile
        self.comparison_mode = comparison_mode

    def update_factor(
        self,
        generation: int,
        parent_fitness: np.ndarray,
        trial_fitness: np.ndarray,
        new_fitness: np.ndarray,
    ) -> None:
        del generation
        offspring = trial_fitness if self.comparison_mode == "trial" else new_fitness

        lam = len(parent_fitness)
        sorted_parents = np.sort(parent_fitness)  # ascending: best (smallest) first

        j = float(np.clip(self.comparison_quantile * lam, 1.0, lam))
        j_lo = int(np.floor(j))
        j_hi = int(np.ceil(j))
        frac = j - j_lo

        count_lo = float(np.sum(offspring <= sorted_parents[j_lo - 1]))
        count_hi = float(np.sum(offspring <= sorted_parents[j_hi - 1]))
        k_succ = (1.0 - frac) * count_lo + frac * count_hi

        z = (2.0 / lam) * (k_succ - (lam + 1) / 2.0)
        self._apply_success_signal(z)