from pathlib import Path

import cecpy
import numpy as np
import pytest

from src.cec_problem import CEC2017Problem, BOUNDS
from src.de_base import DEBase

CEC2017_DATA_DIR = Path(cecpy.__file__).parent / "data" / "cec2017"


def load_shift(func_id: int, dim: int) -> np.ndarray:
    shift = np.loadtxt(CEC2017_DATA_DIR / f"shift_data_{func_id}.txt")
    return shift[:dim]


@pytest.mark.parametrize("func_id", [1, 5, 30])
def test_optimum_is_func_id_times_100(func_id: int):
    assert CEC2017Problem(func_id, 10).optimum == func_id * 100


def test_bounds_are_standard_range():
    assert CEC2017Problem(1, 10).bounds == BOUNDS == (-100.0, 100.0)


@pytest.mark.parametrize("dim", [10, 30])
def test_call_returns_finite_scalar(dim: int):
    value = CEC2017Problem(3, dim)(np.zeros(dim))
    assert isinstance(value, float)
    assert np.isfinite(value)


@pytest.mark.parametrize("dim", [10, 30])
def test_f1_at_known_optimum_equals_f_star(dim: int):
    prob = CEC2017Problem(1, dim)
    value = prob(load_shift(1, dim))
    assert value == pytest.approx(prob.optimum, abs=1e-6)


def test_evaluate_batch_matches_pointwise_and_shape():
    prob = CEC2017Problem(7, 10)
    rng = np.random.default_rng(0)
    X = rng.uniform(-100, 100, size=(5, 10))
    batch = prob.evaluate(X)
    assert batch.shape == (5,)
    pointwise = np.array([prob(x) for x in X])
    np.testing.assert_allclose(batch, pointwise)


class TestDEOnCEC:
    def test_base_de_runs_and_improves_on_f1(self):
        prob = CEC2017Problem(1, 10)
        result = DEBase(func=prob, dim=10, bounds=prob.bounds,
                        population_size=20, max_evals=2_000, seed=1).run()
        assert np.isfinite(result.best_fitness)
        assert result.best_fitness <= result.history_best_fitness[0]
        assert result.best_fitness - prob.optimum >= -1e-6  # error is non-negative