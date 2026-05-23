from pathlib import Path

import cecpy
import numpy as np
import pytest
from cecpy.benchmark import CECEdition, CECEvaluator
from cecpy.benchmark.evaluator import problem_optimum_value

DIMENSIONS = [10, 30]
CEC2017_DATA_DIR = Path(cecpy.__file__).parent / "data" / "cec2017"


@pytest.fixture(scope="module")
def evaluator() -> CECEvaluator:
    return CECEvaluator(CECEdition.CEC2017, DIMENSIONS)


def load_shift(func_id: int, dim: int) -> np.ndarray:
    shift = np.loadtxt(CEC2017_DATA_DIR / f"shift_data_{func_id}.txt")
    return shift[:dim]


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_evaluator_returns_number(evaluator: CECEvaluator, dim: int) -> None:
    x = np.zeros((dim, 1))
    out = evaluator(1, x)
    assert isinstance(out, list)
    assert len(out) == 1
    assert np.isfinite(out[0])


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_output_shape_matches_population_size(
    evaluator: CECEvaluator, dim: int
) -> None:
    population_size = 5
    x = np.zeros((dim, population_size))
    out = evaluator(1, x)
    assert len(out) == population_size


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_f1_at_known_optimum_equals_f_star(
    evaluator: CECEvaluator, dim: int
) -> None:
    x_star = load_shift(1, dim).reshape(dim, 1)
    f_star = problem_optimum_value(CECEdition.CEC2017, 1)
    assert f_star == 100

    value = evaluator(1, x_star)[0]
    assert value == pytest.approx(f_star, abs=1e-6)