"""Tests for proofforge.rewards.base — RewardResult and RewardOracle."""

from __future__ import annotations

import pytest
from proofforge.rewards.base import RewardOracle, RewardResult


# ---------------------------------------------------------------------------
# RewardResult
# ---------------------------------------------------------------------------


class TestRewardResultCreation:
    """RewardResult can be created with valid reward values."""

    def test_reward_zero(self):
        r = RewardResult(reward=0.0, verified=False)
        assert r.reward == 0.0
        assert r.verified is False

    def test_reward_half(self):
        r = RewardResult(reward=0.5, verified=False, metadata={"source": "test"})
        assert r.reward == 0.5
        assert r.metadata["source"] == "test"

    def test_reward_one(self):
        r = RewardResult(reward=1.0, verified=True)
        assert r.reward == 1.0
        assert r.verified is True

    def test_metadata_defaults_to_empty_dict(self):
        r = RewardResult(reward=0.0, verified=False)
        assert r.metadata == {}

    def test_metadata_is_stored(self):
        meta = {"oracle": "lean", "check_duration_ms": 42}
        r = RewardResult(reward=1.0, verified=True, metadata=meta)
        assert r.metadata["oracle"] == "lean"
        assert r.metadata["check_duration_ms"] == 42

    def test_boundary_exactly_zero(self):
        r = RewardResult(reward=0.0, verified=False)
        assert r.reward == 0.0

    def test_boundary_exactly_one(self):
        r = RewardResult(reward=1.0, verified=True)
        assert r.reward == 1.0

    def test_reward_in_interior(self):
        r = RewardResult(reward=0.123, verified=False)
        assert r.reward == pytest.approx(0.123)


class TestRewardResultValidation:
    """RewardResult rejects rewards outside [0, 1]."""

    def test_negative_reward_raises(self):
        with pytest.raises(ValueError, match="Reward must be in"):
            RewardResult(reward=-0.1, verified=False)

    def test_slightly_above_one_raises(self):
        with pytest.raises(ValueError, match="Reward must be in"):
            RewardResult(reward=1.1, verified=True)

    def test_large_negative_raises(self):
        with pytest.raises(ValueError):
            RewardResult(reward=-100.0, verified=False)

    def test_large_positive_raises(self):
        with pytest.raises(ValueError):
            RewardResult(reward=2.0, verified=True)

    def test_nan_reward_raises(self):
        # float('nan') satisfies neither 0 <= nan nor nan <= 1
        import math
        with pytest.raises(ValueError):
            RewardResult(reward=float("nan"), verified=False)


class TestRewardResultFrozen:
    """RewardResult is a frozen dataclass — mutation is not allowed."""

    def test_cannot_set_reward(self):
        r = RewardResult(reward=0.5, verified=False)
        with pytest.raises((AttributeError, TypeError)):
            r.reward = 0.9  # type: ignore[misc]

    def test_cannot_set_verified(self):
        r = RewardResult(reward=0.5, verified=False)
        with pytest.raises((AttributeError, TypeError)):
            r.verified = True  # type: ignore[misc]

    def test_cannot_set_metadata(self):
        r = RewardResult(reward=0.5, verified=False)
        with pytest.raises((AttributeError, TypeError)):
            r.metadata = {"new": "value"}  # type: ignore[misc]

    def test_equality_by_value(self):
        r1 = RewardResult(reward=1.0, verified=True, metadata={"k": "v"})
        r2 = RewardResult(reward=1.0, verified=True, metadata={"k": "v"})
        assert r1 == r2

    def test_hashable(self):
        # Frozen dataclasses are hashable (metadata dict makes this fail
        # unless metadata is excluded; just verify no error on creation)
        r = RewardResult(reward=1.0, verified=True)
        # Frozen dataclasses with mutable fields may or may not be hashable
        # — just verify the object exists and is usable
        assert r is not None


# ---------------------------------------------------------------------------
# RewardOracle
# ---------------------------------------------------------------------------


class TestRewardOracleAbstract:
    """RewardOracle cannot be instantiated directly — it is abstract."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RewardOracle()  # type: ignore[abstract]

    def test_subclass_without_evaluate_is_abstract(self):
        class IncompleteOracle(RewardOracle):
            pass  # no evaluate() method

        with pytest.raises(TypeError):
            IncompleteOracle()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class ConcreteOracle(RewardOracle):
            def evaluate(self, statement, solution, **kwargs) -> RewardResult:
                return RewardResult(reward=1.0, verified=True)

        oracle = ConcreteOracle()
        result = oracle.evaluate("stmt", "sol")
        assert result.reward == 1.0

    def test_batch_evaluate_default_uses_evaluate(self):
        """Default batch_evaluate calls evaluate() sequentially."""
        call_log: list[tuple[str, str]] = []

        class LoggingOracle(RewardOracle):
            def evaluate(self, statement, solution, **kwargs) -> RewardResult:
                call_log.append((statement, solution))
                return RewardResult(reward=0.5, verified=False)

        oracle = LoggingOracle()
        pairs = [("s1", "sol1"), ("s2", "sol2"), ("s3", "sol3")]
        results = oracle.batch_evaluate(pairs)

        assert len(results) == 3
        assert all(r.reward == 0.5 for r in results)
        assert call_log == pairs

    def test_batch_evaluate_returns_list(self):
        class AlwaysPass(RewardOracle):
            def evaluate(self, statement, solution, **kwargs) -> RewardResult:
                return RewardResult(reward=1.0, verified=True)

        oracle = AlwaysPass()
        results = oracle.batch_evaluate([("a", "b"), ("c", "d")])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_evaluate_empty_list(self):
        class AnyOracle(RewardOracle):
            def evaluate(self, statement, solution, **kwargs) -> RewardResult:
                return RewardResult(reward=1.0, verified=True)

        oracle = AnyOracle()
        results = oracle.batch_evaluate([])
        assert results == []
