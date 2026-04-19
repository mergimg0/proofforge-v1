"""Tests for proofforge.rewards.shaped."""

from __future__ import annotations

import pytest
from proofforge.rewards.base import RewardOracle, RewardResult
from proofforge.rewards.shaped import (
    FailureMode,
    ShapedConfig,
    ShapedReward,
    SHAPED_REWARDS,
    classify_failure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AlwaysVerifiedOracle(RewardOracle):
    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        return RewardResult(reward=1.0, verified=True, metadata={"oracle": "mock"})


class AlwaysFailOracle(RewardOracle):
    """Returns verified=False; optionally carries an error message in metadata."""

    def __init__(self, error: str = ""):
        self.error = error

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        return RewardResult(
            reward=0.0,
            verified=False,
            metadata={"oracle": "mock", "error": self.error},
        )


# ---------------------------------------------------------------------------
# classify_failure()
# ---------------------------------------------------------------------------


class TestClassifyFailureNoContent:
    def test_empty_string(self):
        assert classify_failure("", "") == FailureMode.NO_CONTENT

    def test_whitespace_only(self):
        assert classify_failure("   \n\t  ", "") == FailureMode.NO_CONTENT

    def test_plain_english_no_lean(self):
        # No Lean keywords, no structure → NO_CONTENT
        assert classify_failure("I don't know", "") == FailureMode.NO_CONTENT


class TestClassifyFailureGarbled:
    def test_lone_by_keyword(self):
        # "by" is a Lean keyword but no tactic and no structure
        result = classify_failure("by", "")
        assert result == FailureMode.GARBLED

    def test_lean_keywords_only_no_structure(self):
        # Has "sorry" but nothing else structural
        result = classify_failure("sorry", "")
        assert result == FailureMode.GARBLED

    def test_single_tactic_no_structure(self):
        # tactic_count=1, no PROOF_STRUCTURE_RE match
        result = classify_failure("simp", "")
        assert result == FailureMode.GARBLED


class TestClassifyFailureSyntaxError:
    def test_unknown_identifier_error(self):
        proof = "by exact foo_bar_baz"
        error = "unknown identifier 'foo_bar_baz'"
        result = classify_failure(proof, error)
        assert result == FailureMode.SYNTAX_ERROR

    def test_unknown_tactic_error(self):
        proof = "by my_custom_tactic"
        error = "unknown tactic 'my_custom_tactic'"
        result = classify_failure(proof, error)
        assert result == FailureMode.SYNTAX_ERROR


class TestClassifyFailureTacticFail:
    def test_tactic_failed_error(self):
        proof = "by simp"
        error = "tactic 'simp' failed to simplify"
        result = classify_failure(proof, error)
        assert result == FailureMode.TACTIC_FAIL

    def test_tactic_no_goals_error(self):
        proof = "by exact rfl"
        error = "tactic 'exact' failed, there are no goals"
        result = classify_failure(proof, error)
        assert result == FailureMode.TACTIC_FAIL

    def test_multiple_tactics_no_specific_error(self):
        # tactic_count >= 2 with no matching error pattern → TACTIC_FAIL
        proof = "by simp; exact rfl"
        result = classify_failure(proof, "some unrecognized error")
        assert result == FailureMode.TACTIC_FAIL


class TestClassifyFailureTypeMismatch:
    def test_type_mismatch_error(self):
        proof = "by exact h"
        error = "type mismatch: has type Nat, expected Int"
        result = classify_failure(proof, error)
        assert result == FailureMode.TYPE_MISMATCH

    def test_has_type_error(self):
        proof = "by exact h"
        error = "has type Foo, expected Bar"
        result = classify_failure(proof, error)
        assert result == FailureMode.TYPE_MISMATCH

    def test_expected_type_error(self):
        proof = "by exact h"
        error = "expected type Nat, got Int"
        result = classify_failure(proof, error)
        assert result == FailureMode.TYPE_MISMATCH

    def test_not_definitionally_equal_error(self):
        proof = "by rfl"
        error = "is not definitionally equal to the expected type"
        result = classify_failure(proof, error)
        assert result == FailureMode.TYPE_MISMATCH


class TestClassifyFailureWrongTheorem:
    def test_application_type_mismatch_error(self):
        # "application type mismatch" contains "type mismatch" which matches
        # TYPE_MISMATCH_RE before the WRONG_THEOREM check in the classifier.
        proof = "by apply Nat.add_comm"
        error = "application type mismatch"
        result = classify_failure(proof, error)
        assert result == FailureMode.TYPE_MISMATCH

    def test_incorrect_number_of_arguments_error(self):
        proof = "by exact Nat.add_comm 1 2 3"
        error = "incorrect number of arguments"
        result = classify_failure(proof, error)
        assert result == FailureMode.WRONG_THEOREM


# ---------------------------------------------------------------------------
# ShapedReward
# ---------------------------------------------------------------------------


class TestShapedRewardDuringWarmup:
    """During warmup, partial credit is given for failures."""

    def _make(self, warmup_steps: int = 50, error: str = "") -> ShapedReward:
        return ShapedReward(
            base_oracle=AlwaysFailOracle(error=error),
            config=ShapedConfig(warmup_steps=warmup_steps),
            current_step=0,
        )

    def test_tactic_fail_gets_partial_credit(self):
        sr = self._make(error="tactic 'simp' failed")
        # "by simp" → classify as TACTIC_FAIL → reward = 0.2
        result = sr.evaluate("stmt", "by simp")
        assert result.reward > 0.0
        assert result.reward == pytest.approx(SHAPED_REWARDS[FailureMode.TACTIC_FAIL])

    def test_garbled_gets_tiny_credit(self):
        sr = self._make()
        result = sr.evaluate("stmt", "sorry")
        assert result.reward == pytest.approx(SHAPED_REWARDS[FailureMode.GARBLED])

    def test_no_content_gets_zero(self):
        sr = self._make()
        result = sr.evaluate("stmt", "")
        assert result.reward == pytest.approx(0.0)

    def test_failure_mode_in_metadata(self):
        sr = self._make(error="tactic 'ring' failed")
        result = sr.evaluate("stmt", "by ring")
        assert "failure_mode" in result.metadata

    def test_reward_source_is_shaped_during_warmup(self):
        sr = self._make()
        result = sr.evaluate("stmt", "by simp")
        assert result.metadata.get("reward_source") == "shaped"


class TestShapedRewardAfterWarmup:
    """After warmup, all failures give 0.0 (pure binary)."""

    def _make(self, warmup_steps: int = 50, error: str = "tactic failed") -> ShapedReward:
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error=error),
            config=ShapedConfig(warmup_steps=warmup_steps),
            current_step=warmup_steps,  # exactly at warmup boundary
        )
        return sr

    def test_tactic_fail_after_warmup_gives_zero(self):
        sr = self._make(error="tactic 'simp' failed")
        result = sr.evaluate("stmt", "by simp")
        assert result.reward == pytest.approx(0.0)

    def test_type_mismatch_after_warmup_gives_zero(self):
        sr = self._make(error="type mismatch")
        result = sr.evaluate("stmt", "by exact h")
        assert result.reward == pytest.approx(0.0)

    def test_reward_source_is_binary_after_warmup(self):
        sr = self._make()
        result = sr.evaluate("stmt", "by simp")
        assert result.metadata.get("reward_source") == "binary"


class TestShapedRewardVerifiedAlwaysFullCredit:
    """Verified proofs always get 1.0 regardless of step."""

    def test_verified_during_warmup(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(warmup_steps=50),
            current_step=0,
        )
        result = sr.evaluate("stmt", "by rfl")
        assert result.reward == pytest.approx(1.0)
        assert result.verified is True

    def test_verified_after_warmup(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(warmup_steps=50),
            current_step=100,
        )
        result = sr.evaluate("stmt", "by rfl")
        assert result.reward == pytest.approx(1.0)
        assert result.verified is True


class TestShapedRewardUpdateStep:
    """update_step() controls warmup boundary."""

    def test_starts_in_warmup(self):
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic failed"),
            config=ShapedConfig(warmup_steps=50),
            current_step=0,
        )
        result = sr.evaluate("stmt", "by simp")
        assert result.reward > 0.0  # warmup → partial credit

    def test_update_step_exits_warmup(self):
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic failed"),
            config=ShapedConfig(warmup_steps=50),
            current_step=0,
        )
        sr.update_step(50)
        result = sr.evaluate("stmt", "by simp")
        assert result.reward == pytest.approx(0.0)  # binary

    def test_update_step_stores_value(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(warmup_steps=50),
            current_step=0,
        )
        sr.update_step(99)
        assert sr.current_step == 99


class TestShapedRewardFailureDistribution:
    """failure_distribution() counts correctly."""

    def test_empty_counts_initially(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(),
            current_step=0,
        )
        dist = sr.failure_distribution()
        assert all(v == 0 for v in dist.values())

    def test_verified_counted(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(),
            current_step=0,
        )
        sr.evaluate("stmt", "by rfl")
        sr.evaluate("stmt", "by rfl")
        dist = sr.failure_distribution()
        assert dist[FailureMode.VERIFIED.name] == 2

    def test_failure_mode_counted(self):
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic 'simp' failed"),
            config=ShapedConfig(warmup_steps=100),
            current_step=0,
        )
        sr.evaluate("stmt", "by simp")
        sr.evaluate("stmt", "by simp")
        dist = sr.failure_distribution()
        assert dist[FailureMode.TACTIC_FAIL.name] == 2

    def test_distribution_keys_are_strings(self):
        sr = ShapedReward(
            base_oracle=AlwaysVerifiedOracle(),
            config=ShapedConfig(),
        )
        dist = sr.failure_distribution()
        assert all(isinstance(k, str) for k in dist.keys())


class TestShapedRewardBlendMode:
    """With blend_factor > 0, after warmup: weighted combination."""

    def test_blend_gives_nonzero_reward_after_warmup(self):
        blend = 0.5
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic 'simp' failed"),
            config=ShapedConfig(warmup_steps=50, blend_factor=blend),
            current_step=50,  # past warmup
        )
        result = sr.evaluate("stmt", "by simp")
        # blend*shaped_r + (1-blend)*0 = 0.5 * 0.2 = 0.1
        shaped_r = SHAPED_REWARDS[FailureMode.TACTIC_FAIL]
        expected = blend * shaped_r
        assert result.reward == pytest.approx(expected)

    def test_blend_reward_source_is_blended(self):
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic failed"),
            config=ShapedConfig(warmup_steps=50, blend_factor=0.1),
            current_step=50,
        )
        result = sr.evaluate("stmt", "by simp")
        assert result.metadata.get("reward_source") == "blended"

    def test_zero_blend_factor_is_pure_binary(self):
        sr = ShapedReward(
            base_oracle=AlwaysFailOracle(error="tactic failed"),
            config=ShapedConfig(warmup_steps=50, blend_factor=0.0),
            current_step=50,
        )
        result = sr.evaluate("stmt", "by simp")
        assert result.reward == pytest.approx(0.0)
        assert result.metadata.get("reward_source") == "binary"
