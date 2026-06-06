import numpy as np
from typing import Callable
from scipy.stats import rankdata

from src.de_base import DEBase
from src.success_rule import SuccessRuleAdapter


class DEPSR(DEBase, SuccessRuleAdapter):
    """DE/rand/1/bin with Population Success Rule adaptation of the mutation factor."""

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
        z_star: float = 0.0,
        comparison_mode: str = "trial",
        c_sigma: float = 0.3,
        d_sigma: float = 1.0,
    ):
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
        self._init_adapter(factor_init, factor_min, factor_max, c_sigma, d_sigma)
        self.z_star = z_star
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
        mixed = np.concatenate([parent_fitness, offspring])
        ranks = rankdata(mixed)  # 1 = best (smallest fitness)
        rank_sum_parents = float(np.sum(ranks[:lam]))
        rank_sum_offspring = float(np.sum(ranks[lam:]))

        z = (rank_sum_parents - rank_sum_offspring) / (lam ** 2) - self.z_star
        self._apply_success_signal(z)