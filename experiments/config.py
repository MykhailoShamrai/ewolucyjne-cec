TOL = 1e-8
BOUNDS = (-100.0, 100.0)
DEPRECATED = {2}            # f2 is deprecated in CEC 2017
FUNCTIONS = [i for i in range(1, 31) if i not in DEPRECATED]
DIMENSIONS = [10, 30]


def max_fes(dim: int) -> int:
    return 10_000 * dim


def pop_size(dim: int) -> int:
    return 10 * dim


CR = 0.9
FACTOR_INIT = 0.5
FACTOR_MIN = 0.1
FACTOR_MAX = 1.0

QUANTILE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

ZSTAR_GRID_TRIAL = [-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10,
                    -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

ZSTAR_GRID_POPULATION = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]