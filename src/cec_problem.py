import numpy as np
from cecpy.benchmark import CECEdition, CECEvaluator
from cecpy.benchmark.evaluator import problem_optimum_value

BOUNDS = (-100.0, 100.0)


class CEC2017Problem:
    """Adapts a CEC 2017 function to a callable usable as ``DEBase(func=...)``.

    ``__call__(x)`` takes a single point (1-D, length ``dim``) and returns the
    raw objective value f(x); ``evaluate(X)`` evaluates a batch of points given
    row-major (n_points, dim). The known optimum f* and the search bounds are
    exposed so a runner can report error = f(x) - f*.
    """

    def __init__(self, func_id: int, dim: int, evaluator: CECEvaluator | None = None):
        self.func_id = func_id
        self.dim = dim
        self.bounds = BOUNDS
        self.optimum = problem_optimum_value(CECEdition.CEC2017, func_id)
        self._evaluator = evaluator or CECEvaluator(CECEdition.CEC2017, [dim])

    def __call__(self, x: np.ndarray) -> float:
        col = np.ascontiguousarray(np.asarray(x, dtype=float).reshape(self.dim, 1))
        return float(self._evaluator(self.func_id, col)[0])

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        cols = np.ascontiguousarray(np.asarray(X, dtype=float).T)
        return np.asarray(self._evaluator(self.func_id, cols), dtype=float)