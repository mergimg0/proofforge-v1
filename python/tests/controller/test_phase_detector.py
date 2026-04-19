"""Tests for proofforge.controller.phase_detector.

Key implementation details that shape the tests:
- detect() increments _stagnation_counter by 1 each call — stagnation needs
  patience calls with low flat reward, not patience updates.
- The reward window is 20; getting mean above a threshold requires filling
  the window with above-threshold values.
- Hysteresis regression check: accumulated→disruption requires
  reward_mean < 0.05 AND trend < -0.005 (declining, not just flat).
"""

from __future__ import annotations

import pytest

from proofforge.controller.metrics import ControllerMetrics
from proofforge.controller.phase_detector import PhaseDetector, PhaseTransition, TrainingPhase


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reach_accumulation(det: PhaseDetector = None) -> tuple[PhaseDetector, ControllerMetrics]:  # type: ignore[assignment]
    """Drive detector to ACCUMULATION phase and return (det, metrics)."""
    if det is None:
        det = PhaseDetector()
    metrics = ControllerMetrics()
    for i in range(10):
        metrics.update(step=i, reward=0.12 + i * 0.01)
    det.detect(metrics)
    assert det.current_phase == TrainingPhase.ACCUMULATION
    return det, metrics


def _reach_breakout(det: PhaseDetector = None) -> tuple[PhaseDetector, ControllerMetrics]:  # type: ignore[assignment]
    """Drive detector to BREAKOUT phase and return (det, metrics).

    Needs enough 0.30-reward pushes so that the window mean clearly exceeds
    the default breakout_reward_threshold of 0.25.
    """
    det, metrics = _reach_accumulation(det)
    # 20 pushes of 0.30 → mean=0.30 > 0.25 threshold
    for i in range(20):
        metrics.update(step=10 + i, reward=0.30)
    det.detect(metrics)
    assert det.current_phase == TrainingPhase.BREAKOUT
    return det, metrics


def _reach_consolidation(det: PhaseDetector = None) -> tuple[PhaseDetector, ControllerMetrics]:  # type: ignore[assignment]
    """Drive detector to CONSOLIDATION phase and return (det, metrics)."""
    det, metrics = _reach_breakout(det)
    for i in range(10):
        metrics.update(step=30 + i, reward=0.55, retention_rate=0.80)
    det.detect(metrics)
    assert det.current_phase == TrainingPhase.CONSOLIDATION
    return det, metrics


# ---------------------------------------------------------------------------
# TrainingPhase enum
# ---------------------------------------------------------------------------


class TestTrainingPhaseEnum:
    def test_has_disruption(self):
        assert TrainingPhase.DISRUPTION.value == "disruption"

    def test_has_accumulation(self):
        assert TrainingPhase.ACCUMULATION.value == "accumulation"

    def test_has_breakout(self):
        assert TrainingPhase.BREAKOUT.value == "breakout"

    def test_has_consolidation(self):
        assert TrainingPhase.CONSOLIDATION.value == "consolidation"

    def test_has_saturation(self):
        assert TrainingPhase.SATURATION.value == "saturation"

    def test_has_stagnation(self):
        assert TrainingPhase.STAGNATION.value == "stagnation"

    def test_exactly_six_values(self):
        assert len(TrainingPhase) == 6


# ---------------------------------------------------------------------------
# PhaseDetector — initial state
# ---------------------------------------------------------------------------


class TestPhaseDetectorInitialState:
    def test_starts_in_disruption(self):
        det = PhaseDetector()
        assert det.current_phase == TrainingPhase.DISRUPTION

    def test_no_transitions_at_start(self):
        det = PhaseDetector()
        assert det.transitions == []

    def test_detect_returns_disruption_below_min_window(self):
        det = PhaseDetector(min_window_size=5)
        metrics = ControllerMetrics()
        for i in range(4):
            metrics.update(step=i, reward=0.3)
        assert det.detect(metrics) == TrainingPhase.DISRUPTION

    def test_detect_returns_disruption_with_single_observation(self):
        det = PhaseDetector()
        metrics = ControllerMetrics()
        metrics.update(step=1, reward=0.5)
        assert det.detect(metrics) == TrainingPhase.DISRUPTION


# ---------------------------------------------------------------------------
# Disruption → Accumulation
# ---------------------------------------------------------------------------


class TestDisruptionToAccumulation:
    def test_transitions_when_reward_above_threshold_with_positive_trend(self):
        det = PhaseDetector(accumulation_reward_threshold=0.10)
        metrics = ControllerMetrics()
        for i in range(10):
            metrics.update(step=i, reward=0.12 + i * 0.01)
        assert det.detect(metrics) == TrainingPhase.ACCUMULATION

    def test_stays_in_disruption_when_reward_below_threshold(self):
        det = PhaseDetector(accumulation_reward_threshold=0.10)
        metrics = ControllerMetrics()
        for i in range(10):
            metrics.update(step=i, reward=0.05)
        assert det.detect(metrics) == TrainingPhase.DISRUPTION

    def test_stays_in_disruption_when_reward_above_threshold_but_trend_flat(self):
        det = PhaseDetector(accumulation_reward_threshold=0.10)
        metrics = ControllerMetrics()
        for i in range(10):
            metrics.update(step=i, reward=0.15)
        # constant values → trend = 0.0, not > 0
        assert det.detect(metrics) == TrainingPhase.DISRUPTION

    def test_transition_recorded_in_transitions_list(self):
        det, _ = _reach_accumulation()
        assert len(det.transitions) == 1
        t = det.transitions[0]
        assert t.from_phase == TrainingPhase.DISRUPTION
        assert t.to_phase == TrainingPhase.ACCUMULATION


# ---------------------------------------------------------------------------
# Accumulation → Breakout
# ---------------------------------------------------------------------------


class TestAccumulationToBreakout:
    def test_transitions_when_reward_mean_crosses_breakout_threshold(self):
        det, _ = _reach_breakout()
        assert det.current_phase == TrainingPhase.BREAKOUT

    def test_transition_recorded(self):
        det, _ = _reach_breakout()
        from_accumulation = [t for t in det.transitions if t.to_phase == TrainingPhase.BREAKOUT]
        assert len(from_accumulation) == 1
        assert from_accumulation[0].from_phase == TrainingPhase.ACCUMULATION

    def test_estimated_breakout_eta_during_accumulation_returns_int(self):
        det, metrics = _reach_accumulation()
        eta = det.estimated_breakout_eta(metrics)
        assert eta is not None
        assert isinstance(eta, int)
        assert eta >= 0

    def test_estimated_breakout_eta_none_outside_accumulation(self):
        det = PhaseDetector()
        metrics = ControllerMetrics()
        # Still in disruption — fewer than 10 rising values
        for i in range(5):
            metrics.update(step=i, reward=0.05)
        assert det.estimated_breakout_eta(metrics) is None

    def test_estimated_breakout_eta_none_when_trend_nonpositive(self):
        """eta returns None when trend ≤ 0 (no progress toward breakout)."""
        det, metrics = _reach_accumulation()
        assert det.current_phase == TrainingPhase.ACCUMULATION
        # Push many flat rewards so trend collapses to ≈0
        for i in range(20):
            metrics.update(step=20 + i, reward=0.15)
        # trend is now effectively 0.0 (constant sequence)
        assert metrics.reward.trend == pytest.approx(0.0, abs=1e-9)
        assert det.estimated_breakout_eta(metrics) is None

    def test_estimated_breakout_eta_zero_when_mean_already_above_threshold(self):
        """When reward mean already exceeds the breakout threshold, eta = 0.

        The implementation returns 0 only when trend > 0 AND remaining <= 0.
        Push a gently rising sequence so trend > 0 while mean > threshold.
        """
        det, metrics = _reach_accumulation()
        # Push gently rising values all above 0.25 so mean > threshold and trend > 0
        for i in range(20):
            metrics.update(step=20 + i, reward=0.30 + i * 0.001)
        # If still accumulation (hysteresis may keep us there), check eta
        if det.current_phase == TrainingPhase.ACCUMULATION:
            assert metrics.reward.mean > det.breakout_reward_threshold
            assert metrics.reward.trend > 0
            assert det.estimated_breakout_eta(metrics) == 0


# ---------------------------------------------------------------------------
# Breakout → Consolidation
# ---------------------------------------------------------------------------


class TestBreakoutToConsolidation:
    def test_transitions_to_consolidation_when_retention_high(self):
        det, metrics = _reach_breakout()
        for i in range(10):
            metrics.update(step=30 + i, reward=0.55, retention_rate=0.80)
        assert det.detect(metrics) == TrainingPhase.CONSOLIDATION

    def test_consolidation_transition_recorded(self):
        det, _ = _reach_consolidation()
        cons = [t for t in det.transitions if t.to_phase == TrainingPhase.CONSOLIDATION]
        assert len(cons) == 1
        assert cons[0].from_phase == TrainingPhase.BREAKOUT


# ---------------------------------------------------------------------------
# Breakout → Saturation (skip consolidation when no retention data)
# ---------------------------------------------------------------------------


class TestBreakoutToSaturationSkipConsolidation:
    def test_reaches_saturation_directly_from_breakout_without_retention(self):
        """Without any retention_rate data, breakout goes straight to saturation."""
        det, metrics = _reach_breakout()
        # No retention_rate pushed at all
        for i in range(20):
            metrics.update(step=30 + i, reward=0.85)
        phase = det.detect(metrics)
        assert phase == TrainingPhase.SATURATION

    def test_saturation_transition_from_breakout_recorded(self):
        det, metrics = _reach_breakout()
        for i in range(20):
            metrics.update(step=30 + i, reward=0.85)
        det.detect(metrics)
        sat = [t for t in det.transitions if t.to_phase == TrainingPhase.SATURATION]
        assert len(sat) == 1
        assert sat[0].from_phase == TrainingPhase.BREAKOUT


# ---------------------------------------------------------------------------
# Consolidation → Saturation
# ---------------------------------------------------------------------------


class TestConsolidationToSaturation:
    def test_transitions_to_saturation_when_reward_high_and_std_low(self):
        det, metrics = _reach_consolidation()
        for i in range(20):
            metrics.update(step=40 + i, reward=0.85, retention_rate=0.95)
        assert det.detect(metrics) == TrainingPhase.SATURATION

    def test_transitions_to_saturation_via_high_flat_pass_rate(self):
        """pass_rate > 0.6 and flat for 10+ steps also triggers saturation."""
        det, metrics = _reach_consolidation()
        # pass_rate.count must reach 10 — push 15 steps
        for i in range(15):
            metrics.update(step=40 + i, reward=0.65, retention_rate=0.85, pass_rate=0.62)
        assert det.detect(metrics) == TrainingPhase.SATURATION


# ---------------------------------------------------------------------------
# Stagnation
# ---------------------------------------------------------------------------


class TestStagnation:
    def test_stagnation_detected_after_patience_detect_calls(self):
        """_stagnation_counter increments once per detect() call, not per push.
        Need stagnation_patience calls with flat-low reward to trigger stagnation.
        """
        det = PhaseDetector(stagnation_patience=50, stagnation_reward_ceiling=0.15)
        metrics = ControllerMetrics()
        # Push enough data so window is full and step_count >= patience
        for i in range(60):
            metrics.update(step=i, reward=0.10)
        # Call detect() 50 times to saturate the counter
        phase = TrainingPhase.DISRUPTION
        for _ in range(50):
            phase = det.detect(metrics)
        assert phase == TrainingPhase.STAGNATION

    def test_stagnation_escape_when_reward_starts_climbing(self):
        det = PhaseDetector(stagnation_patience=50)
        metrics = ControllerMetrics()
        for i in range(60):
            metrics.update(step=i, reward=0.10)
        # Drive counter to saturation
        for _ in range(50):
            det.detect(metrics)
        assert det.current_phase == TrainingPhase.STAGNATION
        # Now push a clearly rising sequence
        for i in range(10):
            metrics.update(step=60 + i, reward=0.10 + i * 0.02)
        assert det.detect(metrics) == TrainingPhase.ACCUMULATION

    def test_stagnation_escape_recorded_as_transition(self):
        det = PhaseDetector(stagnation_patience=50)
        metrics = ControllerMetrics()
        for i in range(60):
            metrics.update(step=i, reward=0.10)
        for _ in range(50):
            det.detect(metrics)
        for i in range(10):
            metrics.update(step=60 + i, reward=0.10 + i * 0.02)
        det.detect(metrics)
        escape = [t for t in det.transitions if t.from_phase == TrainingPhase.STAGNATION]
        assert len(escape) == 1
        assert escape[0].to_phase == TrainingPhase.ACCUMULATION


# ---------------------------------------------------------------------------
# _is_regression
# ---------------------------------------------------------------------------


class TestIsRegression:
    def test_forward_disruption_to_accumulation_not_regression(self):
        assert not PhaseDetector._is_regression(TrainingPhase.DISRUPTION, TrainingPhase.ACCUMULATION)

    def test_forward_accumulation_to_breakout_not_regression(self):
        assert not PhaseDetector._is_regression(TrainingPhase.ACCUMULATION, TrainingPhase.BREAKOUT)

    def test_forward_breakout_to_consolidation_not_regression(self):
        assert not PhaseDetector._is_regression(TrainingPhase.BREAKOUT, TrainingPhase.CONSOLIDATION)

    def test_forward_consolidation_to_saturation_not_regression(self):
        assert not PhaseDetector._is_regression(TrainingPhase.CONSOLIDATION, TrainingPhase.SATURATION)

    def test_backward_accumulation_to_disruption_is_regression(self):
        assert PhaseDetector._is_regression(TrainingPhase.ACCUMULATION, TrainingPhase.DISRUPTION)

    def test_backward_breakout_to_accumulation_is_regression(self):
        assert PhaseDetector._is_regression(TrainingPhase.BREAKOUT, TrainingPhase.ACCUMULATION)

    def test_backward_consolidation_to_breakout_is_regression(self):
        assert PhaseDetector._is_regression(TrainingPhase.CONSOLIDATION, TrainingPhase.BREAKOUT)

    def test_same_phase_not_regression(self):
        assert not PhaseDetector._is_regression(TrainingPhase.ACCUMULATION, TrainingPhase.ACCUMULATION)

    def test_stagnation_order_equals_disruption(self):
        """Both DISRUPTION and STAGNATION have order 0, so going from STAGNATION
        to DISRUPTION is not a regression (same level)."""
        assert not PhaseDetector._is_regression(TrainingPhase.STAGNATION, TrainingPhase.DISRUPTION)


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_regression_blocked_within_min_phase_dwell(self):
        """A brief dip should not cause regression within the dwell window."""
        det = PhaseDetector(min_phase_dwell=10)
        det_used, metrics = _reach_accumulation(det)
        # Immediately after entering accumulation (step ~9), add just 3 bad steps
        for i in range(3):
            metrics.update(step=10 + i, reward=0.01)
        # steps_in_phase ≈ 3, which is < min_phase_dwell=10 → regression blocked
        assert det_used.detect(metrics) == TrainingPhase.ACCUMULATION

    def test_regression_allowed_after_min_phase_dwell(self):
        """After the dwell window, a genuine sustained decline should regress."""
        det = PhaseDetector(min_phase_dwell=5)
        det_used, metrics = _reach_accumulation(det)
        phase_start = det_used.phase_start_step
        # Push a long declining sequence well past dwell window
        # Decline: start 0.10 → end ~0.001, trend < -0.005 and mean < 0.05
        for i in range(20):
            v = max(0.001, 0.10 - i * 0.006)
            metrics.update(step=phase_start + 6 + i, reward=v)
        steps_in = metrics.step_count - det_used.phase_start_step
        assert steps_in > 5  # past dwell window
        assert metrics.reward.mean < 0.05
        assert metrics.reward.trend < -0.005
        assert det_used.detect(metrics) == TrainingPhase.DISRUPTION


# ---------------------------------------------------------------------------
# PhaseTransition dataclass
# ---------------------------------------------------------------------------


class TestPhaseTransition:
    def test_transition_has_expected_fields(self):
        det, _ = _reach_accumulation()
        assert len(det.transitions) == 1
        t = det.transitions[0]
        assert isinstance(t, PhaseTransition)
        for attr in ("from_phase", "to_phase", "step", "trigger", "metrics_snapshot"):
            assert hasattr(t, attr)

    def test_transition_metrics_snapshot_is_dict(self):
        det, _ = _reach_accumulation()
        assert isinstance(det.transitions[0].metrics_snapshot, dict)

    def test_transition_trigger_is_nonempty_str(self):
        det, _ = _reach_accumulation()
        t = det.transitions[0]
        assert isinstance(t.trigger, str)
        assert len(t.trigger) > 0

    def test_transition_step_is_int(self):
        det, _ = _reach_accumulation()
        assert isinstance(det.transitions[0].step, int)

    def test_multiple_transitions_accumulated(self):
        det, _ = _reach_breakout()
        # Should have at least disruption→accumulation and accumulation→breakout
        phases_reached = {t.to_phase for t in det.transitions}
        assert TrainingPhase.ACCUMULATION in phases_reached
        assert TrainingPhase.BREAKOUT in phases_reached
