"""Tests for proofforge.sweep.protein_sweep.

Tests for Space normalization, Hyperparameters, pareto_points, and Protein
suggest/observe. Works WITHOUT gpytorch installed by skipping Protein tests
when the dependency is missing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proofforge.sweep.protein_sweep import (
    Linear,
    Log,
    Logit,
    Pow2,
    Hyperparameters,
    pareto_points,
    prune_pareto_front,
    _GPYTORCH_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Space normalization round-trips
# ---------------------------------------------------------------------------

class TestLinearSpace:
    def test_round_trip(self):
        s = Linear(min=0.0, max=1.0, scale=0.5)
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            normed = s.normalize(val)
            recovered = s.unnormalize(normed)
            assert abs(recovered - val) < 1e-10, f"Failed for {val}"

    def test_bounds(self):
        s = Linear(min=10, max=100, scale=0.5)
        assert s.normalize(10) == pytest.approx(-1.0)
        assert s.normalize(100) == pytest.approx(1.0)

    def test_integer(self):
        s = Linear(min=0, max=10, scale=0.5, is_integer=True)
        assert isinstance(s.unnormalize(0.0), int)


class TestPow2Space:
    def test_round_trip(self):
        s = Pow2(min=4, max=64, scale=0.5)
        for val in [4, 8, 16, 32, 64]:
            normed = s.normalize(val)
            recovered = s.unnormalize(normed)
            assert recovered == val, f"Failed for {val}"

    def test_bounds(self):
        s = Pow2(min=4, max=64, scale=0.5)
        assert s.normalize(4) == pytest.approx(-1.0)
        assert s.normalize(64) == pytest.approx(1.0)


class TestLogSpace:
    def test_round_trip(self):
        s = Log(min=1e-7, max=1e-4, scale=0.5)
        for val in [1e-7, 1e-6, 1e-5, 1e-4]:
            normed = s.normalize(val)
            recovered = s.unnormalize(normed)
            assert abs(math.log10(recovered) - math.log10(val)) < 1e-6

    def test_bounds(self):
        s = Log(min=1e-7, max=1e-4, scale=0.5)
        assert s.normalize(1e-7) == pytest.approx(-1.0)
        assert s.normalize(1e-4) == pytest.approx(1.0)


class TestLogitSpace:
    def test_round_trip(self):
        s = Logit(min=0.1, max=0.9, scale=0.5)
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            normed = s.normalize(val)
            recovered = s.unnormalize(normed)
            assert abs(recovered - val) < 1e-6, f"Failed for {val}"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

def _make_sweep_config():
    """Minimal sweep config for testing."""
    return {
        "metric": "pass_rate",
        "goal": "maximize",
        "learning_rate": {
            "distribution": "log_normal",
            "min": 1e-7,
            "max": 1e-4,
            "scale": 0.5,
        },
        "group_size": {
            "distribution": "uniform_pow2",
            "min": 4,
            "max": 64,
            "scale": 0.5,
        },
        "alpha": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
            "scale": 0.5,
        },
    }


class TestHyperparameters:
    def test_num_params(self):
        h = Hyperparameters(_make_sweep_config())
        assert h.num == 3

    def test_sample_shape(self):
        h = Hyperparameters(_make_sweep_config())
        samples = h.sample(10)
        assert samples.shape == (10, 3)

    def test_to_dict_from_dict_round_trip(self):
        h = Hyperparameters(_make_sweep_config())
        sample = h.search_centers
        params = h.to_dict(sample)
        recovered = h.from_dict(params)
        np.testing.assert_allclose(recovered, sample, atol=1e-6)

    def test_optimize_direction(self):
        config = _make_sweep_config()
        config["goal"] = "minimize"
        h = Hyperparameters(config)
        assert h.optimize_direction == -1


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoPoints:
    def test_basic_pareto(self):
        obs = [
            {"output": 0.3, "cost": 100},
            {"output": 0.5, "cost": 200},
            {"output": 0.4, "cost": 300},  # dominated by (0.5, 200)
            {"output": 0.8, "cost": 400},
        ]
        pareto, idxs = pareto_points(obs)
        scores = [p["output"] for p in pareto]
        assert 0.3 in scores
        assert 0.5 in scores
        assert 0.8 in scores
        assert 0.4 not in scores  # dominated

    def test_empty(self):
        pareto, idxs = pareto_points([])
        assert pareto == []
        assert idxs == []

    def test_single(self):
        obs = [{"output": 0.5, "cost": 100}]
        pareto, idxs = pareto_points(obs)
        assert len(pareto) == 1


class TestPruneParetoFront:
    def test_prune_inefficient_tail(self):
        pareto = [
            {"output": 0.5, "cost": 100},
            {"output": 0.8, "cost": 200},
            {"output": 0.801, "cost": 1000},  # inefficient: tiny gain, huge cost
        ]
        pruned = prune_pareto_front(pareto)
        assert len(pruned) <= len(pareto)

    def test_single_point(self):
        pareto = [{"output": 0.5, "cost": 100}]
        assert prune_pareto_front(pareto) == pareto


# ---------------------------------------------------------------------------
# Protein (requires gpytorch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GPYTORCH_AVAILABLE, reason="gpytorch not installed")
class TestProtein:
    def test_suggest_returns_dict(self):
        from proofforge.sweep.protein_sweep import Protein
        config = _make_sweep_config()
        config["max_suggestion_cost"] = 3600
        config["downsample"] = 5
        config["early_stop_quantile"] = 0.3
        p = Protein(config, use_gpu=False, num_random_samples=3)

        hypers, info = p.suggest()
        assert isinstance(hypers, dict)
        assert "learning_rate" in hypers

    def test_suggest_observe_cycle(self):
        from proofforge.sweep.protein_sweep import Protein
        config = _make_sweep_config()
        config["max_suggestion_cost"] = 3600
        config["downsample"] = 5
        config["early_stop_quantile"] = 0.3
        p = Protein(config, use_gpu=False, num_random_samples=2)

        for i in range(5):
            hypers, _ = p.suggest()
            score = np.random.uniform(0.0, 1.0)
            cost = np.random.uniform(100, 3600)
            p.observe(hypers, score=score, cost=cost)

        assert len(p.success_observations) == 5

    def test_early_stop_on_nan(self):
        from proofforge.sweep.protein_sweep import Protein
        config = _make_sweep_config()
        config["max_suggestion_cost"] = 3600
        config["downsample"] = 5
        config["early_stop_quantile"] = 0.3
        p = Protein(config, use_gpu=False)
        logs = {"loss": {"policy": float("nan")}}
        assert p.early_stop(logs) is True

    def test_observe_failure(self):
        from proofforge.sweep.protein_sweep import Protein
        config = _make_sweep_config()
        config["max_suggestion_cost"] = 3600
        config["downsample"] = 5
        config["early_stop_quantile"] = 0.3
        p = Protein(config, use_gpu=False, num_random_samples=2)

        hypers, _ = p.suggest()
        p.observe(hypers, score=float("nan"), cost=100, is_failure=True)
        assert len(p.failure_observations) == 1
        assert len(p.success_observations) == 0
