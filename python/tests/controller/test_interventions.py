"""Tests for proofforge.controller.interventions.

Key implementation details:
- InterventionType has 12 members (WAIT, LOG, NO_ACTION, EXPAND_DATASET,
  SWITCH_EFFICIENCY, INCREASE_GROUP_SIZE, REDUCE_LEARNING_RATE,
  ENABLE_SHAPED_WARMUP, INCREASE_TEMPERATURE, CHECKPOINT,
  DIVERSIFY_TACTICS, CONSOLIDATION_HOLD).
- Cooldown: same InterventionType not re-issued within 10 steps.
- Saturation efficiency rule only fires when gradient_saturation() is True
  (reward mean > 0.8, std < 0.1, count >= 10).
- Gradient anomaly rule only fires when gradient_norm mean > 5.0 and count >= 5.
- max_interventions_per_step caps the returned list length.
"""

from __future__ import annotations

import pytest

from proofforge.controller.interventions import (
    Intervention,
    InterventionRules,
    InterventionType,
)
from proofforge.controller.metrics import ControllerMetrics
from proofforge.controller.phase_detector import TrainingPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_for_phase(
    phase: TrainingPhase,
    step: int = 100,
    reward: float = 0.3,
    pass_rate: float | None = None,
    gradient_norm: float | None = None,
    retention_rate: float | None = None,
    tactic_diversity: float | None = None,
    n: int = 15,
) -> ControllerMetrics:
    """Build a ControllerMetrics with enough observations to satisfy window guards."""
    m = ControllerMetrics()
    for i in range(n):
        m.update(
            step=step - n + i,
            reward=reward,
            pass_rate=pass_rate,
            gradient_norm=gradient_norm,
            retention_rate=retention_rate,
            tactic_diversity=tactic_diversity,
        )
    m.step_count = step
    return m


def _saturation_metrics(step: int = 100) -> ControllerMetrics:
    """Metrics that trigger gradient_saturation(): mean > 0.8, std < 0.1."""
    return _metrics_for_phase(TrainingPhase.SATURATION, step=step, reward=0.85, n=15)


# ---------------------------------------------------------------------------
# InterventionType enum
# ---------------------------------------------------------------------------


class TestInterventionTypeEnum:
    def test_has_wait(self):
        assert InterventionType.WAIT.value == "wait"

    def test_has_log(self):
        assert InterventionType.LOG.value == "log"

    def test_has_no_action(self):
        assert InterventionType.NO_ACTION.value == "no_action"

    def test_has_expand_dataset(self):
        assert InterventionType.EXPAND_DATASET.value == "expand_dataset"

    def test_has_switch_efficiency(self):
        assert InterventionType.SWITCH_EFFICIENCY.value == "switch_efficiency"

    def test_has_increase_group_size(self):
        assert InterventionType.INCREASE_GROUP_SIZE.value == "increase_group_size"

    def test_has_reduce_learning_rate(self):
        assert InterventionType.REDUCE_LEARNING_RATE.value == "reduce_learning_rate"

    def test_has_enable_shaped_warmup(self):
        assert InterventionType.ENABLE_SHAPED_WARMUP.value == "enable_shaped_warmup"

    def test_has_increase_temperature(self):
        assert InterventionType.INCREASE_TEMPERATURE.value == "increase_temperature"

    def test_has_checkpoint(self):
        assert InterventionType.CHECKPOINT.value == "checkpoint"

    def test_has_diversify_tactics(self):
        assert InterventionType.DIVERSIFY_TACTICS.value == "diversify_tactics"

    def test_has_consolidation_hold(self):
        assert InterventionType.CONSOLIDATION_HOLD.value == "consolidation_hold"

    def test_exactly_twelve_members(self):
        assert len(InterventionType) == 12


# ---------------------------------------------------------------------------
# Intervention.__str__
# ---------------------------------------------------------------------------


class TestInterventionStr:
    def _make(self, priority: int, itype: InterventionType = InterventionType.WAIT) -> Intervention:
        return Intervention(
            type=itype,
            reason="test reason",
            phase=TrainingPhase.DISRUPTION,
            step=42,
            priority=priority,
        )

    def test_str_priority_0_shows_info(self):
        s = str(self._make(0))
        assert "[INFO]" in s

    def test_str_priority_1_shows_recommend(self):
        s = str(self._make(1))
        assert "[RECOMMEND]" in s

    def test_str_priority_2_shows_critical(self):
        s = str(self._make(2))
        assert "[CRITICAL]" in s

    def test_str_contains_step(self):
        s = str(self._make(0))
        assert "42" in s

    def test_str_contains_phase(self):
        s = str(self._make(0))
        assert "disruption" in s

    def test_str_contains_type_value(self):
        s = str(self._make(0, InterventionType.CHECKPOINT))
        assert "checkpoint" in s

    def test_str_contains_reason(self):
        s = str(self._make(0))
        assert "test reason" in s

    def test_str_unknown_priority_defaults_to_info(self):
        i = Intervention(
            type=InterventionType.WAIT,
            reason="r",
            phase=TrainingPhase.DISRUPTION,
            step=1,
            priority=99,
        )
        s = str(i)
        assert "[INFO]" in s


# ---------------------------------------------------------------------------
# InterventionRules — phase-specific rules
# ---------------------------------------------------------------------------


class TestInterventionRulesDisruption:
    def test_disruption_produces_wait(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.DISRUPTION, reward=0.05)
        result = rules.evaluate(TrainingPhase.DISRUPTION, m)
        types = [i.type for i in result]
        assert InterventionType.WAIT in types

    def test_disruption_wait_is_informational(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.DISRUPTION, reward=0.05)
        result = rules.evaluate(TrainingPhase.DISRUPTION, m)
        wait = next(i for i in result if i.type == InterventionType.WAIT)
        assert wait.priority == 0

    def test_disruption_does_not_produce_checkpoint(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.DISRUPTION, reward=0.05)
        result = rules.evaluate(TrainingPhase.DISRUPTION, m)
        assert all(i.type != InterventionType.CHECKPOINT for i in result)


class TestInterventionRulesAccumulation:
    def _accumulation_metrics(self, with_leading: bool = False) -> ControllerMetrics:
        m = ControllerMetrics()
        for i in range(15):
            reward = 0.12 + i * 0.01 if with_leading else 0.15
            pass_rate = 0.08 if with_leading else None
            m.update(step=i, reward=reward, pass_rate=pass_rate)
        return m

    def test_accumulation_produces_log(self):
        rules = InterventionRules()
        m = self._accumulation_metrics()
        result = rules.evaluate(TrainingPhase.ACCUMULATION, m)
        assert any(i.type == InterventionType.LOG for i in result)

    def test_accumulation_log_includes_leading_indicator_when_active(self):
        rules = InterventionRules()
        m = self._accumulation_metrics(with_leading=True)
        result = rules.evaluate(TrainingPhase.ACCUMULATION, m)
        log = next((i for i in result if i.type == InterventionType.LOG), None)
        assert log is not None
        # leading_indicator is stored in parameters
        assert "leading_indicator" in log.parameters

    def test_accumulation_does_not_produce_stagnation_interventions(self):
        rules = InterventionRules()
        m = self._accumulation_metrics()
        result = rules.evaluate(TrainingPhase.ACCUMULATION, m)
        stagnation_types = {
            InterventionType.INCREASE_GROUP_SIZE,
            InterventionType.REDUCE_LEARNING_RATE,
            InterventionType.ENABLE_SHAPED_WARMUP,
        }
        assert all(i.type not in stagnation_types for i in result)


class TestInterventionRulesBreakout:
    def test_breakout_produces_checkpoint(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        assert any(i.type == InterventionType.CHECKPOINT for i in result)

    def test_breakout_checkpoint_is_auto_apply(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        chk = next(i for i in result if i.type == InterventionType.CHECKPOINT)
        assert chk.auto_apply is True

    def test_breakout_checkpoint_priority_is_1(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        chk = next(i for i in result if i.type == InterventionType.CHECKPOINT)
        assert chk.priority == 1


class TestInterventionRulesConsolidation:
    def test_consolidation_produces_consolidation_hold(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.CONSOLIDATION, reward=0.6, retention_rate=0.85)
        result = rules.evaluate(TrainingPhase.CONSOLIDATION, m)
        assert any(i.type == InterventionType.CONSOLIDATION_HOLD for i in result)

    def test_consolidation_hold_auto_apply(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.CONSOLIDATION, reward=0.6, retention_rate=0.85)
        result = rules.evaluate(TrainingPhase.CONSOLIDATION, m)
        hold = next(i for i in result if i.type == InterventionType.CONSOLIDATION_HOLD)
        assert hold.auto_apply is True

    def test_consolidation_does_not_produce_wait(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.CONSOLIDATION, reward=0.6)
        result = rules.evaluate(TrainingPhase.CONSOLIDATION, m)
        assert all(i.type != InterventionType.WAIT for i in result)


class TestInterventionRulesSaturation:
    def test_saturation_produces_switch_efficiency_when_gradient_saturated(self):
        rules = InterventionRules()
        m = _saturation_metrics()
        assert m.gradient_saturation()  # precondition
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert any(i.type == InterventionType.SWITCH_EFFICIENCY for i in result)

    def test_saturation_produces_expand_dataset(self):
        rules = InterventionRules()
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert any(i.type == InterventionType.EXPAND_DATASET for i in result)

    def test_saturation_switch_efficiency_priority_is_2(self):
        rules = InterventionRules()
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        sw = next(i for i in result if i.type == InterventionType.SWITCH_EFFICIENCY)
        assert sw.priority == 2

    def test_saturation_no_switch_efficiency_when_disabled(self):
        rules = InterventionRules(efficiency_reward_enabled=False)
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert all(i.type != InterventionType.SWITCH_EFFICIENCY for i in result)

    def test_saturation_no_expand_when_expanding_ring_disabled(self):
        rules = InterventionRules(expanding_ring_enabled=False)
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert all(i.type != InterventionType.EXPAND_DATASET for i in result)

    def test_saturation_no_switch_efficiency_without_gradient_saturation(self):
        """SWITCH_EFFICIENCY rule guards on gradient_saturation() being True."""
        rules = InterventionRules()
        # reward=0.5 → gradient_saturation() is False
        m = _metrics_for_phase(TrainingPhase.SATURATION, reward=0.5)
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert all(i.type != InterventionType.SWITCH_EFFICIENCY for i in result)


class TestInterventionRulesStagnation:
    def _stagnation_metrics(self, step: int = 100) -> ControllerMetrics:
        return _metrics_for_phase(TrainingPhase.STAGNATION, step=step, reward=0.10)

    def test_stagnation_produces_increase_group_size(self):
        rules = InterventionRules()
        m = self._stagnation_metrics()
        result = rules.evaluate(TrainingPhase.STAGNATION, m)
        assert any(i.type == InterventionType.INCREASE_GROUP_SIZE for i in result)

    def test_stagnation_produces_reduce_lr(self):
        rules = InterventionRules()
        m = self._stagnation_metrics()
        result = rules.evaluate(TrainingPhase.STAGNATION, m)
        assert any(i.type == InterventionType.REDUCE_LEARNING_RATE for i in result)

    def test_stagnation_produces_enable_shaped(self):
        rules = InterventionRules()
        # Raise max_interventions to 5 so all three stagnation rules fit
        rules.max_interventions_per_step = 5
        m = self._stagnation_metrics()
        result = rules.evaluate(TrainingPhase.STAGNATION, m)
        assert any(i.type == InterventionType.ENABLE_SHAPED_WARMUP for i in result)


# ---------------------------------------------------------------------------
# Cross-phase rules: tactic monoculture and gradient anomaly
# ---------------------------------------------------------------------------


class TestTacticMonocultureRule:
    def test_monoculture_produces_diversify_tactics(self):
        rules = InterventionRules()
        # diversity=1.5 < 3.0 threshold, count >= 5
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, tactic_diversity=1.5)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        assert any(i.type == InterventionType.DIVERSIFY_TACTICS for i in result)

    def test_monoculture_priority_is_2(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, tactic_diversity=1.5)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        div = next(i for i in result if i.type == InterventionType.DIVERSIFY_TACTICS)
        assert div.priority == 2

    def test_no_monoculture_when_diversity_sufficient(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, tactic_diversity=5.0)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        assert all(i.type != InterventionType.DIVERSIFY_TACTICS for i in result)

    def test_monoculture_fires_across_any_phase(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.ACCUMULATION, reward=0.15, tactic_diversity=1.0)
        result = rules.evaluate(TrainingPhase.ACCUMULATION, m)
        assert any(i.type == InterventionType.DIVERSIFY_TACTICS for i in result)


class TestGradientAnomalyRule:
    def test_gradient_anomaly_produces_reduce_lr(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, gradient_norm=6.0)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        assert any(i.type == InterventionType.REDUCE_LEARNING_RATE for i in result)

    def test_gradient_anomaly_reduce_lr_priority_is_2(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, gradient_norm=6.0)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        # Get the reduce_lr that comes from gradient anomaly (priority=2)
        reduce_lrs = [i for i in result if i.type == InterventionType.REDUCE_LEARNING_RATE]
        assert any(i.priority == 2 for i in reduce_lrs)

    def test_no_gradient_anomaly_when_norm_normal(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.BREAKOUT, reward=0.35, gradient_norm=1.5)
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        # Any REDUCE_LEARNING_RATE present should NOT be the gradient anomaly one
        # (gradient_norm 1.5 < 5.0 threshold → anomaly rule returns None)
        reduce_lrs_priority2 = [
            i for i in result
            if i.type == InterventionType.REDUCE_LEARNING_RATE and i.priority == 2
        ]
        assert len(reduce_lrs_priority2) == 0

    def test_no_gradient_anomaly_with_insufficient_data(self):
        """Anomaly rule requires gradient_norm.count >= 5."""
        rules = InterventionRules()
        m = ControllerMetrics()
        # Only 3 gradient_norm observations
        for i in range(3):
            m.update(step=i, reward=0.35, gradient_norm=8.0)
        m.step_count = 100
        result = rules.evaluate(TrainingPhase.BREAKOUT, m)
        priority2_reduce = [
            i for i in result
            if i.type == InterventionType.REDUCE_LEARNING_RATE and i.priority == 2
        ]
        assert len(priority2_reduce) == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_same_intervention_not_repeated_within_10_steps(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.DISRUPTION, step=100, reward=0.05)
        # First call issues WAIT
        result1 = rules.evaluate(TrainingPhase.DISRUPTION, m)
        assert any(i.type == InterventionType.WAIT for i in result1)

        # Second call at same step — within cooldown window of 10
        result2 = rules.evaluate(TrainingPhase.DISRUPTION, m)
        assert all(i.type != InterventionType.WAIT for i in result2)

    def test_intervention_reissued_after_cooldown(self):
        rules = InterventionRules()
        m = _metrics_for_phase(TrainingPhase.DISRUPTION, step=100, reward=0.05)
        rules.evaluate(TrainingPhase.DISRUPTION, m)

        # Advance step_count past cooldown window (10 steps)
        m2 = _metrics_for_phase(TrainingPhase.DISRUPTION, step=111, reward=0.05)
        result = rules.evaluate(TrainingPhase.DISRUPTION, m2)
        assert any(i.type == InterventionType.WAIT for i in result)

    def test_different_intervention_types_not_blocked_by_each_others_cooldown(self):
        rules = InterventionRules(max_interventions_per_step=5)
        m = _metrics_for_phase(TrainingPhase.STAGNATION, step=100, reward=0.10)
        result = rules.evaluate(TrainingPhase.STAGNATION, m)
        types = {i.type for i in result}
        # Multiple distinct stagnation types should all appear on first call
        assert len(types) > 1


# ---------------------------------------------------------------------------
# max_interventions_per_step
# ---------------------------------------------------------------------------


class TestMaxInterventionsPerStep:
    def test_respects_max_interventions_per_step(self):
        rules = InterventionRules(max_interventions_per_step=1)
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        assert len(result) <= 1

    def test_max_interventions_default_is_3(self):
        rules = InterventionRules()
        assert rules.max_interventions_per_step == 3

    def test_result_never_exceeds_max(self):
        for max_n in (1, 2, 3):
            rules = InterventionRules(max_interventions_per_step=max_n)
            m = _saturation_metrics()
            result = rules.evaluate(TrainingPhase.SATURATION, m)
            assert len(result) <= max_n

    def test_interventions_sorted_by_priority_descending(self):
        """Higher-priority interventions are selected first when cap applies."""
        rules = InterventionRules(max_interventions_per_step=2)
        # SATURATION with gradient saturation → SWITCH_EFFICIENCY (p=2) + EXPAND_DATASET (p=1)
        m = _saturation_metrics()
        result = rules.evaluate(TrainingPhase.SATURATION, m)
        priorities = [i.priority for i in result]
        assert priorities == sorted(priorities, reverse=True)
