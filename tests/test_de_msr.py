import numpy as np
import pytest

from src.de_msr import DEMSR


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _make_de_msr(**overrides) -> DEMSR:
    params = dict(
        func=sphere,
        dim=5,
        bounds=(-5.0, 5.0),
        population_size=30,
        crossover_probability=0.9,
        max_evals=5_000,
        seed=42,
        F_init=0.5,
        F_min=0.1,
        F_max=1.0,
        target_success=0.2,
        damping=1.0,
    )
    params.update(overrides)
    return DEMSR(**params)


class TestMSRFactorUpdate:
    def test_factor_increases_when_success_rate_above_target(self):
        de = _make_de_msr(F_init=0.5, target_success=0.25, damping=1.0)
        fitness = np.array([5.0, 6.0, 7.0, 8.0])
        new_fitness = np.array([1.0, 2.0, 3.0, 4.0])

        de.update_F(
            generation=1,
            population=np.zeros((4, 5)),
            fitness=fitness,
            new_population=np.zeros((4, 5)),
            new_fitness=new_fitness,
        )

        assert de.F > 0.5

    def test_factor_decreases_when_success_rate_below_target(self):
        de = _make_de_msr(F_init=0.5, target_success=0.75, damping=1.0)
        fitness = np.array([5.0, 6.0, 7.0, 8.0])
        new_fitness = np.array([7.5, 8.0, 9.0, 10.0])

        de.update_F(
            generation=1,
            population=np.zeros((4, 5)),
            fitness=fitness,
            new_population=np.zeros((4, 5)),
            new_fitness=new_fitness,
        )

        assert de.F < 0.5

    def test_factor_unchanged_when_success_rate_equals_target(self):
        de = _make_de_msr(F_init=0.5, target_success=0.5, damping=1.0)
        fitness = np.array([4.0, 6.0, 8.0, 10.0])
        # median parent = 7.0, exactly 2/4 offspring below median => success_rate = 0.5
        new_fitness = np.array([3.0, 6.0, 7.0, 9.0])

        de.update_F(
            generation=1,
            population=np.zeros((4, 5)),
            fitness=fitness,
            new_population=np.zeros((4, 5)),
            new_fitness=new_fitness,
        )

        assert de.F == pytest.approx(0.5)

    def test_factor_is_clipped_to_bounds(self):
        de = _make_de_msr(F_init=0.99, F_min=0.2, F_max=1.0, target_success=0.0, damping=5.0)
        fitness = np.array([5.0, 6.0, 7.0, 8.0])
        new_fitness = np.array([1.0, 2.0, 3.0, 4.0])

        de.update_F(
            generation=1,
            population=np.zeros((4, 5)),
            fitness=fitness,
            new_population=np.zeros((4, 5)),
            new_fitness=new_fitness,
        )

        assert de.F == 1.0


class TestMSRIntegration:
    def test_sphere_converges_close_to_zero(self):
        result = _make_de_msr().run()
        assert result.best_f < 1e-3

    def test_factor_history_respects_bounds(self):
        result = _make_de_msr(F_min=0.2, F_max=0.8).run()
        assert all(0.2 <= f <= 0.8 for f in result.history_factor)

    def test_determinism_same_seed(self):
        r1 = _make_de_msr(seed=123).run()
        r2 = _make_de_msr(seed=123).run()
        assert r1.best_f == r2.best_f
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        assert r1.history_factor == r2.history_factor


class TestMSRValidation:
    def test_invalid_target_success_raises(self):
        with pytest.raises(ValueError):
            _make_de_msr(target_success=1.5)

    def test_invalid_damping_raises(self):
        with pytest.raises(ValueError):
            _make_de_msr(damping=0.0)

    def test_invalid_factor_bounds_raise(self):
        with pytest.raises(ValueError):
            _make_de_msr(F_min=0.0)

        with pytest.raises(ValueError):
            _make_de_msr(F_min=0.9, F_max=0.8)
