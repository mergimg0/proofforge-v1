"""Tests for proofforge.rewards.efficiency."""

from __future__ import annotations

import pytest
from proofforge.rewards.base import RewardOracle, RewardResult
from proofforge.rewards.efficiency import (
    EfficiencyConfig,
    EfficiencyReward,
    TemperatureBreadthTracker,
    TrajectoryLengthMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AlwaysVerifiedOracle(RewardOracle):
    """Always returns verified=True with reward=1.0."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        return RewardResult(reward=1.0, verified=True, metadata={"oracle": "always_verified"})


class AlwaysFailOracle(RewardOracle):
    """Always returns verified=False with reward=0.0."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        return RewardResult(reward=0.0, verified=False, metadata={"oracle": "always_fail"})


# ---------------------------------------------------------------------------
# TemperatureBreadthTracker
# ---------------------------------------------------------------------------


class TestTemperatureBreadthTrackerBreadth:
    """breadth() returns the correct count of reliably-solved temperatures."""

    def setup_method(self):
        self.tracker = TemperatureBreadthTracker(min_attempts=4, breadth_threshold=0.5)

    def test_breadth_zero_for_unknown_theorem(self):
        assert self.tracker.breadth("unknown") == 0

    def test_breadth_zero_insufficient_attempts(self):
        # 3 attempts < min_attempts=4 → not counted
        for _ in range(3):
            self.tracker.record("thm1", 0.2, success=True)
        assert self.tracker.breadth("thm1") == 0

    def test_breadth_one_after_enough_attempts_one_temp(self):
        for _ in range(4):
            self.tracker.record("thm1", 0.2, success=True)
        assert self.tracker.breadth("thm1") == 1

    def test_breadth_two_after_two_temps(self):
        for _ in range(4):
            self.tracker.record("thm1", 0.2, success=True)
            self.tracker.record("thm1", 0.6, success=True)
        assert self.tracker.breadth("thm1") == 2

    def test_breadth_three_after_three_temps(self):
        for _ in range(4):
            self.tracker.record("thm1", 0.2, success=True)
            self.tracker.record("thm1", 0.6, success=True)
            self.tracker.record("thm1", 1.0, success=True)
        assert self.tracker.breadth("thm1") == 3

    def test_breadth_does_not_exceed_three(self):
        # Even with extra records, max is number of tracked temperatures
        for _ in range(10):
            self.tracker.record("thm1", 0.2, success=True)
            self.tracker.record("thm1", 0.6, success=True)
            self.tracker.record("thm1", 1.0, success=True)
        assert self.tracker.breadth("thm1") == 3

    def test_breadth_zero_when_pass_rate_below_threshold(self):
        # 4 attempts, only 1 success → rate = 0.25 < 0.5
        self.tracker.record("thm1", 0.2, success=True)
        for _ in range(3):
            self.tracker.record("thm1", 0.2, success=False)
        assert self.tracker.breadth("thm1") == 0


class TestTemperatureBreadthTrackerSnap:
    """_snap_temperature() snaps to the nearest tracked temperature."""

    def setup_method(self):
        self.tracker = TemperatureBreadthTracker(temperatures=(0.2, 0.6, 1.0))

    def test_snap_exact_low(self):
        assert self.tracker._snap_temperature(0.2) == 0.2

    def test_snap_exact_mid(self):
        assert self.tracker._snap_temperature(0.6) == 0.6

    def test_snap_exact_high(self):
        assert self.tracker._snap_temperature(1.0) == 1.0

    def test_snap_near_low(self):
        assert self.tracker._snap_temperature(0.1) == 0.2

    def test_snap_near_high(self):
        assert self.tracker._snap_temperature(0.95) == 1.0

    def test_snap_midpoint_between_low_and_mid(self):
        # 0.4 is equidistant from 0.2 and 0.6; min() picks the first match
        snapped = self.tracker._snap_temperature(0.4)
        assert snapped in (0.2, 0.6)

    def test_snap_just_above_mid(self):
        assert self.tracker._snap_temperature(0.65) == 0.6


class TestTemperatureBreadthTrackerMasterySummary:
    """mastery_summary() returns the expected structure."""

    def setup_method(self):
        self.tracker = TemperatureBreadthTracker(min_attempts=2)

    def test_empty_tracker(self):
        assert self.tracker.mastery_summary() == {}

    def test_summary_has_breadth_key(self):
        for _ in range(2):
            self.tracker.record("thm1", 0.2, success=True)
        summary = self.tracker.mastery_summary()
        assert "thm1" in summary
        assert "breadth" in summary["thm1"]

    def test_summary_has_temperatures_key(self):
        for _ in range(2):
            self.tracker.record("thm1", 0.6, success=True)
        summary = self.tracker.mastery_summary()
        assert "temperatures" in summary["thm1"]

    def test_summary_per_temperature_has_attempts_successes_rate(self):
        self.tracker.record("thm1", 0.6, success=True)
        self.tracker.record("thm1", 0.6, success=False)
        summary = self.tracker.mastery_summary()
        temp_data = summary["thm1"]["temperatures"]
        snap = self.tracker._snap_temperature(0.6)
        assert temp_data[snap]["attempts"] == 2
        assert temp_data[snap]["successes"] == 1
        assert temp_data[snap]["rate"] == pytest.approx(0.5)

    def test_summary_zero_attempts_gives_zero_rate(self):
        # Record at one temp only; others should show 0 attempts, 0.0 rate
        for _ in range(2):
            self.tracker.record("thm1", 0.2, success=True)
        summary = self.tracker.mastery_summary()
        high_snap = self.tracker._snap_temperature(1.0)
        assert summary["thm1"]["temperatures"][high_snap]["attempts"] == 0
        assert summary["thm1"]["temperatures"][high_snap]["rate"] == 0.0

    def test_summary_multiple_theorems(self):
        for _ in range(2):
            self.tracker.record("thm1", 0.2, success=True)
            self.tracker.record("thm2", 0.6, success=False)
        summary = self.tracker.mastery_summary()
        assert "thm1" in summary
        assert "thm2" in summary


class TestTemperatureBreadthTrackerProofLengths:
    """proof_lengths is recorded only for successful proofs with tokens > 0."""

    def setup_method(self):
        self.tracker = TemperatureBreadthTracker()

    def test_proof_length_recorded_on_success(self):
        self.tracker.record("thm1", 0.2, success=True, proof_tokens=100)
        snap = self.tracker._snap_temperature(0.2)
        assert self.tracker.proof_lengths["thm1"][snap] == [100]

    def test_proof_length_not_recorded_on_failure(self):
        self.tracker.record("thm1", 0.2, success=False, proof_tokens=100)
        assert self.tracker.proof_lengths.get("thm1", {}).get(0.2) is None

    def test_proof_length_not_recorded_when_zero_tokens(self):
        self.tracker.record("thm1", 0.2, success=True, proof_tokens=0)
        assert self.tracker.proof_lengths.get("thm1", {}).get(0.2) is None

    def test_multiple_lengths_accumulate(self):
        snap = self.tracker._snap_temperature(0.6)
        self.tracker.record("thm1", 0.6, success=True, proof_tokens=50)
        self.tracker.record("thm1", 0.6, success=True, proof_tokens=80)
        assert self.tracker.proof_lengths["thm1"][snap] == [50, 80]


# ---------------------------------------------------------------------------
# EfficiencyConfig
# ---------------------------------------------------------------------------


class TestEfficiencyConfigDefaults:
    def test_default_alpha_mastered(self):
        cfg = EfficiencyConfig()
        assert cfg.alpha_mastered == 0.5

    def test_default_alpha_moderate(self):
        cfg = EfficiencyConfig()
        assert cfg.alpha_moderate == 0.2

    def test_default_alpha_fragile_is_zero(self):
        cfg = EfficiencyConfig()
        assert cfg.alpha_fragile == 0.0

    def test_default_max_proof_tokens(self):
        cfg = EfficiencyConfig()
        assert cfg.max_proof_tokens == 512

    def test_default_phase_in_step(self):
        cfg = EfficiencyConfig()
        assert cfg.phase_in_step == 50

    def test_default_phase_in_pass_rate(self):
        cfg = EfficiencyConfig()
        assert cfg.phase_in_pass_rate == 0.20

    def test_default_length_floor_positive(self):
        cfg = EfficiencyConfig()
        assert cfg.length_floor > 0.0

    def test_default_length_floor_below_one(self):
        cfg = EfficiencyConfig()
        assert cfg.length_floor < 1.0


# ---------------------------------------------------------------------------
# EfficiencyReward
# ---------------------------------------------------------------------------


def _make_reward(base_oracle=None, **config_kwargs) -> EfficiencyReward:
    if base_oracle is None:
        base_oracle = AlwaysVerifiedOracle()
    cfg = EfficiencyConfig(**config_kwargs)
    return EfficiencyReward(base_oracle=base_oracle, config=cfg)


class TestEfficiencyRewardPhaseBehavior:
    """Before phase-in the reward is pure binary."""

    def test_before_phase_step_correct_proof_gives_one(self):
        er = _make_reward(phase_in_step=50, phase_in_pass_rate=0.0)
        er.update_training_state(step=10, pass_rate=0.9)
        result = er.evaluate("stmt", "proof", theorem_id="t1", proof_tokens=400)
        assert result.reward == pytest.approx(1.0)

    def test_before_phase_step_failed_proof_gives_zero(self):
        er = _make_reward(base_oracle=AlwaysFailOracle(), phase_in_step=50)
        er.update_training_state(step=10, pass_rate=0.9)
        result = er.evaluate("stmt", "proof", theorem_id="t1")
        assert result.reward == pytest.approx(0.0)

    def test_before_pass_rate_correct_proof_gives_one(self):
        er = _make_reward(phase_in_step=0, phase_in_pass_rate=0.20)
        er.update_training_state(step=100, pass_rate=0.05)
        result = er.evaluate("stmt", "proof", theorem_id="t1", proof_tokens=400)
        assert result.reward == pytest.approx(1.0)

    def test_exactly_at_phase_in_step_transitions(self):
        # step == phase_in_step satisfies current_step < phase_in_step? No — equal is NOT less than
        er = _make_reward(phase_in_step=50, phase_in_pass_rate=0.0)
        er.update_training_state(step=50, pass_rate=1.0)
        # At step=50 the phase-in condition `step < 50` is False, so alpha applies
        # breadth < 2 → alpha_fragile = 0.0 → still binary 1.0
        result = er.evaluate("stmt", "proof", theorem_id="t_no_history", proof_tokens=400)
        assert result.reward == pytest.approx(1.0)


class TestEfficiencyRewardAlphaApplication:
    """After phase-in, alpha is applied based on breadth."""

    def _make_high_breadth_tracker(self, tracker: TemperatureBreadthTracker, theorem_id: str, breadth: int):
        """Populate tracker to achieve a given breadth level."""
        temps = tracker.temperatures[:breadth]
        for temp in temps:
            for _ in range(tracker.min_attempts):
                tracker.record(theorem_id, temp, success=True)

    def test_breadth_3_applies_alpha_mastered(self):
        er = _make_reward(
            phase_in_step=0, phase_in_pass_rate=0.0,
            alpha_mastered=0.5, max_proof_tokens=512
        )
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm_master", breadth=3)

        # Proof at max length → factor = max(0.3, 1.0 - 0.5 * 1.0) = max(0.3, 0.5) = 0.5
        result = er.evaluate("stmt", "proof", theorem_id="thm_master", proof_tokens=512)
        assert result.reward == pytest.approx(0.5)
        assert result.metadata["efficiency_alpha"] == pytest.approx(0.5)

    def test_breadth_2_applies_alpha_moderate(self):
        er = _make_reward(
            phase_in_step=0, phase_in_pass_rate=0.0,
            alpha_moderate=0.2, max_proof_tokens=512
        )
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm_mod", breadth=2)

        # Proof at max length → factor = max(0.3, 1.0 - 0.2 * 1.0) = 0.8
        result = er.evaluate("stmt", "proof", theorem_id="thm_mod", proof_tokens=512)
        assert result.reward == pytest.approx(0.8)
        assert result.metadata["efficiency_alpha"] == pytest.approx(0.2)

    def test_breadth_less_than_2_still_binary(self):
        er = _make_reward(phase_in_step=0, phase_in_pass_rate=0.0)
        er.update_training_state(step=100, pass_rate=1.0)
        # No history → breadth=0 → alpha_fragile=0.0 → pure binary
        result = er.evaluate("stmt", "proof", theorem_id="new_thm", proof_tokens=500)
        assert result.reward == pytest.approx(1.0)
        assert result.metadata["efficiency_alpha"] == pytest.approx(0.0)


class TestEfficiencyRewardLengthEffect:
    """Short proofs get higher reward than long proofs after phase-in."""

    def _make_high_breadth_tracker(self, tracker: TemperatureBreadthTracker, theorem_id: str):
        for temp in tracker.temperatures:
            for _ in range(tracker.min_attempts):
                tracker.record(theorem_id, temp, success=True)

    def test_short_proof_higher_reward_than_long(self):
        er = _make_reward(phase_in_step=0, phase_in_pass_rate=0.0, alpha_mastered=0.5)
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm1")

        short = er.evaluate("stmt", "proof", theorem_id="thm1", proof_tokens=10)
        long = er.evaluate("stmt", "proof", theorem_id="thm1", proof_tokens=400)

        assert short.reward > long.reward

    def test_zero_length_proof_gets_full_reward(self):
        er = _make_reward(phase_in_step=0, phase_in_pass_rate=0.0, alpha_mastered=0.5)
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm1")

        # length_ratio = 0 → factor = 1.0 - 0 = 1.0
        result = er.evaluate("stmt", "proof", theorem_id="thm1", proof_tokens=1)
        assert result.reward == pytest.approx(1.0, abs=0.05)


class TestEfficiencyRewardFloorClamping:
    """Reward never drops below length_floor for a correct proof."""

    def _make_high_breadth_tracker(self, tracker: TemperatureBreadthTracker, theorem_id: str):
        for temp in tracker.temperatures:
            for _ in range(tracker.min_attempts):
                tracker.record(theorem_id, temp, success=True)

    def test_floor_clamping_at_max_length(self):
        floor = 0.3
        er = _make_reward(
            phase_in_step=0, phase_in_pass_rate=0.0,
            alpha_mastered=0.9, max_proof_tokens=100, length_floor=floor
        )
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm1")

        # With alpha=0.9, proof_tokens=100: raw = 1.0 - 0.9*1.0 = 0.1 < floor
        result = er.evaluate("stmt", "proof", theorem_id="thm1", proof_tokens=100)
        assert result.reward >= floor - 1e-9

    def test_floor_never_exceeded_downward(self):
        floor = 0.4
        er = _make_reward(
            phase_in_step=0, phase_in_pass_rate=0.0,
            alpha_mastered=0.5, max_proof_tokens=512, length_floor=floor
        )
        er.update_training_state(step=100, pass_rate=1.0)
        self._make_high_breadth_tracker(er.breadth_tracker, "thm1")

        for tokens in [1, 100, 256, 512, 600]:
            result = er.evaluate("stmt", "proof", theorem_id="thm1", proof_tokens=tokens)
            assert result.reward >= floor - 1e-9, f"reward {result.reward} below floor at tokens={tokens}"


class TestEfficiencyRewardFailedProof:
    """Failed proofs always get 0.0 regardless of phase or breadth."""

    def test_failed_before_phase_in(self):
        er = _make_reward(base_oracle=AlwaysFailOracle(), phase_in_step=50)
        er.update_training_state(step=10, pass_rate=0.9)
        result = er.evaluate("stmt", "proof", theorem_id="t1")
        assert result.reward == pytest.approx(0.0)
        assert result.verified is False

    def test_failed_after_phase_in(self):
        er = _make_reward(base_oracle=AlwaysFailOracle(), phase_in_step=0, phase_in_pass_rate=0.0)
        er.update_training_state(step=100, pass_rate=1.0)
        result = er.evaluate("stmt", "proof", theorem_id="t1")
        assert result.reward == pytest.approx(0.0)
        assert result.verified is False


class TestEfficiencyRewardUpdateTrainingState:
    """update_training_state() updates internal step and pass_rate."""

    def test_initial_state(self):
        er = _make_reward()
        assert er.current_step == 0
        assert er.current_pass_rate == 0.0

    def test_update_changes_step(self):
        er = _make_reward()
        er.update_training_state(step=75, pass_rate=0.5)
        assert er.current_step == 75

    def test_update_changes_pass_rate(self):
        er = _make_reward()
        er.update_training_state(step=75, pass_rate=0.42)
        assert er.current_pass_rate == pytest.approx(0.42)

    def test_multiple_updates(self):
        er = _make_reward()
        er.update_training_state(step=10, pass_rate=0.1)
        er.update_training_state(step=100, pass_rate=0.9)
        assert er.current_step == 100
        assert er.current_pass_rate == pytest.approx(0.9)


class TestTrajectoryLengthRatio:
    """trajectory_length_ratio() returns correct ratio or None."""

    def test_none_when_no_data(self):
        er = _make_reward()
        assert er.trajectory_length_ratio("unknown") is None

    def test_none_when_only_one_temperature_has_data(self):
        er = _make_reward()
        er.breadth_tracker.record("thm1", 0.2, success=True, proof_tokens=100)
        # Only low-T has data; high-T has none
        assert er.trajectory_length_ratio("thm1") is None

    def test_ratio_computed_correctly(self):
        er = _make_reward()
        tracker = er.breadth_tracker
        # low_t = 0.2, high_t = 1.0
        tracker.record("thm1", 0.2, success=True, proof_tokens=100)
        tracker.record("thm1", 0.2, success=True, proof_tokens=200)
        tracker.record("thm1", 1.0, success=True, proof_tokens=400)

        # mean_low = 150, mean_high = 400 → ratio = 150/400 = 0.375
        ratio = er.trajectory_length_ratio("thm1")
        assert ratio == pytest.approx(150.0 / 400.0)

    def test_ratio_equal_lengths_gives_one(self):
        er = _make_reward()
        tracker = er.breadth_tracker
        tracker.record("thm1", 0.2, success=True, proof_tokens=200)
        tracker.record("thm1", 1.0, success=True, proof_tokens=200)
        assert er.trajectory_length_ratio("thm1") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TrajectoryLengthMetrics.compute()
# ---------------------------------------------------------------------------


class TestTrajectoryLengthMetricsCompute:
    def test_empty_results_returns_defaults(self):
        m = TrajectoryLengthMetrics.compute([], step=5)
        assert m.step == 5
        assert m.pass_rate == 0.0
        assert m.mean_proof_length == 0.0

    def test_all_failed_returns_defaults(self):
        results = [
            RewardResult(reward=0.0, verified=False, metadata={"proof_tokens": 100}),
            RewardResult(reward=0.0, verified=False, metadata={"proof_tokens": 200}),
        ]
        m = TrajectoryLengthMetrics.compute(results, step=10)
        assert m.pass_rate == 0.0
        assert m.mean_proof_length == 0.0

    def test_all_verified_pass_rate_one(self):
        results = [
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 100, "efficiency_alpha": 0.5}),
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 200, "efficiency_alpha": 0.5}),
        ]
        m = TrajectoryLengthMetrics.compute(results, step=10)
        assert m.pass_rate == pytest.approx(1.0)

    def test_mean_proof_length_computed(self):
        results = [
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 100}),
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 300}),
        ]
        m = TrajectoryLengthMetrics.compute(results, step=1)
        assert m.mean_proof_length == pytest.approx(200.0)

    def test_pass_rate_partial(self):
        results = [
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 50}),
            RewardResult(reward=0.0, verified=False, metadata={}),
            RewardResult(reward=0.0, verified=False, metadata={}),
            RewardResult(reward=0.0, verified=False, metadata={}),
        ]
        m = TrajectoryLengthMetrics.compute(results, step=1)
        assert m.pass_rate == pytest.approx(0.25)

    def test_efficiency_count(self):
        results = [
            RewardResult(reward=0.8, verified=True, metadata={"proof_tokens": 100, "efficiency_alpha": 0.4}),
            RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 50, "efficiency_alpha": 0.0}),
        ]
        m = TrajectoryLengthMetrics.compute(results, step=1)
        assert m.theorems_with_efficiency == 1
        assert m.theorems_binary_only == 1

    def test_step_preserved(self):
        results = [RewardResult(reward=1.0, verified=True, metadata={"proof_tokens": 10})]
        m = TrajectoryLengthMetrics.compute(results, step=42)
        assert m.step == 42
