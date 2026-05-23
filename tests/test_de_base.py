import numpy as np
import pytest

from src.de_base import DEBase, DEResult


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _make_de(func, **overrides) -> DEBase:
    params = dict(
        func=func,
        dim=5,
        bounds=(-5.0, 5.0),
        population_size=30,
        factor=0.5,
        crossover_probability=0.9,
        max_evals=5_000,
        seed=42,
    )
    params.update(overrides)
    return DEBase(**params)


class TestConvergence:
    def test_sphere_converges_close_to_zero(self):
        result = _make_de(sphere).run()
        assert result.best_f < 1e-3

    def test_sphere_best_x_near_origin(self):
        result = _make_de(sphere).run()
        assert np.allclose(result.best_x, 0.0, atol=0.05)


class TestDeterminism:
    def test_same_seed_same_result(self):
        r1 = _make_de(sphere, seed=123).run()
        r2 = _make_de(sphere, seed=123).run()
        assert r1.best_f == r2.best_f
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        assert r1.history_best_factor == r2.history_best_factor

    def test_different_seeds_differ(self):
        r1 = _make_de(sphere, seed=1, max_evals=1_000).run()
        r2 = _make_de(sphere, seed=2, max_evals=1_000).run()
        assert r1.best_f != r2.best_f


class TestInvariants:
    def test_budget_not_exceeded(self):
        result = _make_de(sphere, max_evals=1_000).run()
        assert result.n_evals <= 1_000

    def test_budget_almost_used(self):
        result = _make_de(sphere, max_evals=1_000).run()
        assert result.n_evals >= 1_000 - 30  # at most one partial generation left

    def test_best_x_within_bounds(self):
        result = _make_de(sphere).run()
        assert np.all(result.best_x >= -5.0)
        assert np.all(result.best_x <= 5.0)

    def test_best_f_matches_function_value(self):
        result = _make_de(sphere).run()
        assert result.best_f == pytest.approx(sphere(result.best_x))

    def test_best_f_monotonic_non_increasing(self):
        result = _make_de(sphere).run()
        history = result.history_best_factor
        diffs = np.diff(history)
        assert np.all(diffs <= 1e-12)  # greedy selection => best never worsens

    def test_history_lengths_consistent(self):
        result = _make_de(sphere).run()
        assert len(result.history_best_factor) == len(result.history_mean_factor)
        assert len(result.history_best_factor) == len(result.history_factor)
        assert len(result.history_best_factor) >= 2  # at least initial + 1 generation

    def test_factor_history_constant_in_base(self):
        result = _make_de(sphere).run()
        assert all(f == 0.5 for f in result.history_factor)


class TestComponents:
    def test_init_population_shape_and_bounds(self):
        de = _make_de(sphere, population_size=20, dim=7)
        pop = de._init_population()
        assert pop.shape == (20, 7)
        assert np.all(pop >= -5.0)
        assert np.all(pop <= 5.0)

    def test_clip_pulls_outliers_back(self):
        de = _make_de(sphere)
        x = np.array([-100.0, 0.0, 100.0, 3.0, -7.0])
        clipped = de._clip(x)
        assert np.all(clipped >= -5.0)
        assert np.all(clipped <= 5.0)

    def test_crossover_takes_at_least_one_gene_from_mutant(self):
        de = _make_de(sphere, crossover_probability=0.0, seed=0)
        target = np.zeros(de.dim)
        mutant = np.ones(de.dim)
        trial = de._crossover(target, mutant)
        # CR=0 means only j_rand position should come from mutant
        assert np.sum(trial == 1.0) == 1
        assert np.sum(trial == 0.0) == de.dim - 1

    def test_crossover_full_replacement_when_cr_is_one(self):
        de = _make_de(sphere, crossover_probability=1.0, seed=0)
        target = np.zeros(de.dim)
        mutant = np.ones(de.dim)
        trial = de._crossover(target, mutant)
        np.testing.assert_array_equal(trial, mutant)

    def test_mutate_picks_three_distinct_others(self):
        de = _make_de(sphere, population_size=10, dim=3)
        pop = np.arange(30, dtype=float).reshape(10, 3)
        # F=0 means mutant == pop[r1], so we can detect that r1 != idx
        de.F = 0.0
        for idx in range(10):
            mutant = de._mutate(pop, idx, factor=0.0)
            assert not np.array_equal(mutant, pop[idx])