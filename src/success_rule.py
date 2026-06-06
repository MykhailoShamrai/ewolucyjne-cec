import numpy as np


class SuccessRuleAdapter:
    def _init_adapter(self, factor_init: float, factor_min: float, factor_max: float,
                      c_sigma: float, d_sigma: float) -> None:
        if factor_min <= 0.0:
            raise ValueError("factor_min must be positive")
        if factor_max < factor_min:
            raise ValueError("factor_max must be >= factor_min")
        if not (0.0 < c_sigma <= 1.0):
            raise ValueError("c_sigma must be in (0, 1]")
        if d_sigma <= 0.0:
            raise ValueError("d_sigma must be positive")

        self.factor_min = factor_min
        self.factor_max = factor_max
        self.c_sigma = c_sigma
        self.d_sigma = d_sigma
        self.s = 0.0
        self.factor = float(np.clip(factor_init, factor_min, factor_max))

    def _apply_success_signal(self, z: float) -> None:
        self.s = (1.0 - self.c_sigma) * self.s + self.c_sigma * float(z)
        self.factor = float(np.clip(self.factor * np.exp(self.s / self.d_sigma),
                                    self.factor_min, self.factor_max))