import numpy as np
import pytest

from src.de_psr import DEPSR


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _make_de_psr(**overrides) -> DEPSR:
    params = dict(
        func=sphere,
        dim=5,
        bounds=(-5.0, 5.0),
        population_size=30,
        crossover_probability=0.9,
        max_evals=5_000,
        seed=42,
        factor_init=0.5,
        factor_min=0.1,
        factor_max=1.0,
        z_star=0.0,
        c_sigma=0.3,
        d_sigma=1.0,
    )
    params.update(overrides)
    return DEPSR(**params)


def _update(de: DEPSR, parent_fitness, trial_fitness) -> None:
    de.update_factor(
        generation=1,
        parent_fitness=np.asarray(parent_fitness, dtype=float),
        trial_fitness=np.asarray(trial_fitness, dtype=float),
        new_fitness=np.asarray(parent_fitness, dtype=float),
    )


class TestPSRFactorUpdate:
    def test_factor_increases_when_trials_outrank_parents(self):
        de = _make_de_psr()
        _update(de, parent_fitness=[5.0, 6.0, 7.0, 8.0], trial_fitness=[1.0, 2.0, 3.0, 4.0])
        assert de.factor > 0.5

    def test_factor_decreases_when_trials_underrank_parents(self):
        de = _make_de_psr()
        _update(de, parent_fitness=[5.0, 6.0, 7.0, 8.0], trial_fitness=[9.0, 10.0, 11.0, 12.0])
        assert de.factor < 0.5

    def test_factor_unchanged_when_rank_sums_balance(self):
        de = _make_de_psr()
        _update(de, parent_fitness=[10.0, 40.0, 60.0, 70.0], trial_fitness=[20.0, 30.0, 50.0, 80.0])
        assert de.factor == pytest.approx(0.5)

    def test_z_star_shifts_operating_point_downward(self):
        de = _make_de_psr(z_star=0.25)
        _update(de, parent_fitness=[10.0, 40.0, 60.0, 70.0], trial_fitness=[20.0, 30.0, 50.0, 80.0])
        assert de.factor < 0.5

    def test_factor_is_clipped_to_bounds(self):
        de = _make_de_psr(factor_init=0.99, factor_min=0.2, factor_max=1.0)
        _update(de, parent_fitness=[5.0, 6.0, 7.0, 8.0], trial_fitness=[1.0, 2.0, 3.0, 4.0])
        assert de.factor == 1.0


class TestPSRRankInvariance:
    def test_factor_history_invariant_to_scaling(self):
        base = _make_de_psr(seed=7, max_evals=3_000).run()
        scaled = DEPSR(
            func=lambda x: 1000.0 * sphere(x),
            dim=5, bounds=(-5.0, 5.0), population_size=30,
            max_evals=3_000, seed=7,
        ).run()
        assert base.history_factor == scaled.history_factor

    def test_factor_history_invariant_to_shift(self):
        base = _make_de_psr(seed=7, max_evals=3_000).run()
        shifted = DEPSR(
            func=lambda x: sphere(x) + 500.0,
            dim=5, bounds=(-5.0, 5.0), population_size=30,
            max_evals=3_000, seed=7,
        ).run()
        assert base.history_factor == shifted.history_factor


class TestPSRIntegration:
    def test_sphere_converges_close_to_zero(self):
        result = _make_de_psr(z_star=-0.2).run()
        assert result.best_fitness < 1e-3

    def test_factor_history_respects_bounds(self):
        result = _make_de_psr(factor_min=0.2, factor_max=0.8).run()
        assert all(0.2 <= f <= 0.8 for f in result.history_factor)

    def test_determinism_same_seed(self):
        r1 = _make_de_psr(seed=123).run()
        r2 = _make_de_psr(seed=123).run()
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        assert r1.history_factor == r2.history_factor


class TestPSRValidation:
    def test_invalid_c_sigma_raises(self):
        with pytest.raises(ValueError):
            _make_de_psr(c_sigma=0.0)

    def test_invalid_d_sigma_raises(self):
        with pytest.raises(ValueError):
            _make_de_psr(d_sigma=0.0)

    def test_invalid_factor_bounds_raise(self):
        with pytest.raises(ValueError):
            _make_de_psr(factor_min=0.0)

        with pytest.raises(ValueError):
            _make_de_psr(factor_min=0.9, factor_max=0.8)

    def test_invalid_comparison_mode_raises(self):
        with pytest.raises(ValueError):
            _make_de_psr(comparison_mode="offspring")


class TestPSRComparisonMode:
    def _update_mode(self, mode, parent, trial, new):
        de = _make_de_psr(comparison_mode=mode)
        de.update_factor(
            generation=1,
            parent_fitness=np.asarray(parent, dtype=float),
            trial_fitness=np.asarray(trial, dtype=float),
            new_fitness=np.asarray(new, dtype=float),
        )
        return de.factor

    def test_population_mode_uses_new_fitness_not_trial(self):
        factor = self._update_mode(
            "population",
            parent=[5.0, 6.0, 7.0, 8.0],
            trial=[100.0, 100.0, 100.0, 100.0],
            new=[1.0, 2.0, 3.0, 4.0],
        )
        assert factor > 0.5

    def test_population_mode_signal_is_nonnegative_under_greedy(self):
        factor = self._update_mode(
            "population",
            parent=[1.0, 2.0, 3.0, 4.0],
            trial=[100.0, 100.0, 100.0, 100.0],
            new=[1.0, 2.0, 3.0, 4.0],
        )
        assert factor == pytest.approx(0.5)