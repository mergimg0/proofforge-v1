"""Tests for proofforge.controller.metrics."""

from __future__ import annotations

import math

import pytest

from proofforge.controller.metrics import ControllerMetrics, MetricWindow


# ---------------------------------------------------------------------------
# MetricWindow
# ---------------------------------------------------------------------------


class TestMetricWindowBasics:
    def test_empty_window_defaults(self):
        w = MetricWindow()
        assert w.count == 0
        assert w.mean == 0.0
        assert w.std == 0.0
        assert w.trend == 0.0
        assert w.last is None
        assert w.is_full is False

    def test_push_single_value(self):
        w = MetricWindow()
        w.push(5.0)
        assert w.count == 1
        assert w.mean == 5.0
        assert w.last == 5.0
        # std requires at least 2 values
        assert w.std == 0.0
        # trend requires at least 3 values
        assert w.trend == 0.0

    def test_push_five_values_mean(self):
        w = MetricWindow()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.push(v)
        assert w.count == 5
        assert w.mean == pytest.approx(3.0)

    def test_push_twenty_values_fills_window(self):
        w = MetricWindow(max_size=20)
        for i in range(20):
            w.push(float(i))
        assert w.count == 20
        assert w.is_full is True

    def test_window_evicts_oldest_when_full(self):
        w = MetricWindow(max_size=5)
        for i in range(6):
            w.push(float(i))
        # Window should contain [1,2,3,4,5] after evicting 0
        assert w.count == 5
        assert w.last == 5.0
        assert w.mean == pytest.approx(3.0)

    def test_is_full_false_before_capacity(self):
        w = MetricWindow(max_size=20)
        for i in range(19):
            w.push(float(i))
        assert w.is_full is False

    def test_is_full_true_at_capacity(self):
        w = MetricWindow(max_size=20)
        for i in range(20):
            w.push(float(i))
        assert w.is_full is True

    def test_last_returns_most_recent(self):
        w = MetricWindow()
        w.push(1.0)
        w.push(2.0)
        w.push(9.9)
        assert w.last == 9.9

    def test_recent_returns_last_n(self):
        w = MetricWindow()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.push(v)
        assert w.recent(3) == [3.0, 4.0, 5.0]

    def test_recent_n_larger_than_count(self):
        w = MetricWindow()
        w.push(1.0)
        w.push(2.0)
        # Asking for 10 when only 2 exist — should return all
        result = w.recent(10)
        assert result == [1.0, 2.0]

    def test_recent_empty_window(self):
        w = MetricWindow()
        assert w.recent(5) == []


class TestMetricWindowStd:
    def test_std_two_values(self):
        w = MetricWindow()
        w.push(1.0)
        w.push(3.0)
        # population std: mean=2, variance=((1-2)^2+(3-2)^2)/2=1, std=1
        assert w.std == pytest.approx(1.0)

    def test_std_known_sequence(self):
        # [1, 2, 3, 4, 5]: mean=3, variance=(4+1+0+1+4)/5=2, std=√2
        w = MetricWindow()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.push(v)
        assert w.std == pytest.approx(math.sqrt(2))

    def test_std_constant_values(self):
        w = MetricWindow()
        for _ in range(5):
            w.push(7.0)
        assert w.std == pytest.approx(0.0)

    def test_std_single_value_is_zero(self):
        w = MetricWindow()
        w.push(42.0)
        assert w.std == 0.0


class TestMetricWindowTrend:
    def test_trend_ascending_is_positive(self):
        w = MetricWindow()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.push(v)
        assert w.trend > 0.0

    def test_trend_descending_is_negative(self):
        w = MetricWindow()
        for v in [5.0, 4.0, 3.0, 2.0, 1.0]:
            w.push(v)
        assert w.trend < 0.0

    def test_trend_constant_is_zero(self):
        w = MetricWindow()
        for _ in range(10):
            w.push(3.0)
        assert w.trend == pytest.approx(0.0, abs=1e-9)

    def test_trend_requires_at_least_three_values(self):
        w = MetricWindow()
        w.push(1.0)
        w.push(2.0)
        # Only 2 values — should return 0
        assert w.trend == 0.0

    def test_trend_three_values(self):
        w = MetricWindow()
        for v in [1.0, 2.0, 3.0]:
            w.push(v)
        # Perfect line: slope = 1
        assert w.trend == pytest.approx(1.0)

    def test_trend_linear_slope_accuracy(self):
        # y = 2x, expect slope ≈ 2
        w = MetricWindow()
        for i in range(5):
            w.push(2.0 * i)
        assert w.trend == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# ControllerMetrics
# ---------------------------------------------------------------------------


class TestControllerMetricsUpdate:
    def test_update_records_reward(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.5)
        assert m.reward.count == 1
        assert m.reward.last == 0.5

    def test_update_skips_none_pass_rate(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, pass_rate=None)
        assert m.pass_rate.count == 0

    def test_update_records_pass_rate_when_provided(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, pass_rate=0.15)
        assert m.pass_rate.count == 1
        assert m.pass_rate.last == 0.15

    def test_update_skips_none_loss(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, loss=None)
        assert m.loss.count == 0

    def test_update_records_loss_when_provided(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, loss=2.5)
        assert m.loss.count == 1

    def test_update_skips_none_gradient_norm(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, gradient_norm=None)
        assert m.gradient_norm.count == 0

    def test_update_records_gradient_norm(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, gradient_norm=1.2)
        assert m.gradient_norm.count == 1

    def test_update_records_proof_length(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, proof_length=42.0)
        assert m.proof_length.count == 1
        assert m.proof_length.last == 42.0

    def test_update_records_retention_rate(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, retention_rate=0.85)
        assert m.retention_rate.count == 1
        assert m.retention_rate.last == 0.85

    def test_update_records_tactic_diversity(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, tactic_diversity=4.5)
        assert m.tactic_diversity.count == 1
        assert m.tactic_diversity.last == 4.5

    def test_update_step_count_tracked(self):
        m = ControllerMetrics()
        m.update(step=42, reward=0.1)
        assert m.step_count == 42


class TestControllerMetricsSnapshot:
    def test_snapshot_returns_dict(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        snap = m.snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_contains_required_keys(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        snap = m.snapshot()
        required_keys = {
            "step",
            "reward_mean",
            "reward_trend",
            "pass_rate_mean",
            "pass_rate_trend",
            "loss_mean",
            "grad_norm_mean",
            "proof_length_mean",
            "retention_mean",
            "tactic_diversity_mean",
            "leading_indicator",
            "gradient_saturated",
            "stagnated",
            "consolidating",
            "tactic_monoculture",
        }
        assert required_keys.issubset(snap.keys())

    def test_snapshot_step_matches_last_update(self):
        m = ControllerMetrics()
        m.update(step=77, reward=0.4)
        assert m.snapshot()["step"] == 77

    def test_snapshot_loss_mean_none_when_no_data(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        assert m.snapshot()["loss_mean"] is None

    def test_snapshot_loss_mean_present_when_provided(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3, loss=1.5)
        assert m.snapshot()["loss_mean"] is not None

    def test_snapshot_grad_norm_mean_none_when_no_data(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        assert m.snapshot()["grad_norm_mean"] is None

    def test_snapshot_retention_mean_none_when_no_data(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        assert m.snapshot()["retention_mean"] is None

    def test_snapshot_reward_mean_correct_type(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        assert isinstance(m.snapshot()["reward_mean"], float)

    def test_snapshot_gradient_saturated_is_bool(self):
        m = ControllerMetrics()
        m.update(step=1, reward=0.3)
        assert isinstance(m.snapshot()["gradient_saturated"], bool)


class TestControllerMetricsRewardLeadsPassRate:
    def test_returns_none_with_insufficient_reward_data(self):
        m = ControllerMetrics()
        for i in range(4):
            m.update(step=i, reward=0.2 + i * 0.01, pass_rate=0.1)
        assert m.reward_leads_pass_rate() is None

    def test_returns_none_with_insufficient_pass_rate_data(self):
        m = ControllerMetrics()
        for i in range(5):
            m.update(step=i, reward=0.2 + i * 0.01)
        # pass_rate never provided
        assert m.reward_leads_pass_rate() is None

    def test_true_when_reward_trending_up_and_pass_rate_flat(self):
        m = ControllerMetrics()
        # Reward: steadily climbing; pass_rate: constant
        for i in range(10):
            m.update(step=i, reward=0.1 + i * 0.02, pass_rate=0.08)
        result = m.reward_leads_pass_rate()
        assert result is True

    def test_false_when_reward_flat_and_pass_rate_flat(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.2, pass_rate=0.1)
        result = m.reward_leads_pass_rate()
        # Reward trend ≈ 0, not > 0.005
        assert result is False

    def test_false_when_pass_rate_also_trending(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.1 + i * 0.02, pass_rate=0.05 + i * 0.01)
        result = m.reward_leads_pass_rate()
        # pass_rate trend is large → not flat → False
        assert result is False


class TestControllerMetricsGradientSaturation:
    def test_false_with_insufficient_data(self):
        m = ControllerMetrics()
        for i in range(9):
            m.update(step=i, reward=0.9)
        assert m.gradient_saturation() is False

    def test_true_when_high_reward_low_std(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.85)
        # mean=0.85 > 0.8, std=0 < 0.1
        assert m.gradient_saturation() is True

    def test_false_when_reward_high_but_std_too_large(self):
        m = ControllerMetrics()
        rewards = [0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9]
        for i, r in enumerate(rewards):
            m.update(step=i, reward=r)
        assert m.reward.std > 0.1
        assert m.gradient_saturation() is False

    def test_false_when_reward_below_threshold(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.75)
        assert m.gradient_saturation() is False


class TestControllerMetricsStagnation:
    def test_false_with_insufficient_steps(self):
        m = ControllerMetrics()
        for i in range(49):
            m.update(step=i, reward=0.1)
        assert m.stagnation() is False

    def test_false_with_insufficient_reward_window(self):
        # step_count >= 50 but < 10 reward observations
        m = ControllerMetrics()
        m.update(step=50, reward=0.1)
        # Only 1 reward observation despite step=50
        assert m.stagnation() is False

    def test_true_when_low_flat_reward_for_many_steps(self):
        m = ControllerMetrics()
        for i in range(60):
            m.update(step=i, reward=0.1)
        # step_count=59, reward.count=20 (window), mean≈0.1, trend≈0
        assert m.stagnation() is True

    def test_false_when_reward_above_ceiling(self):
        m = ControllerMetrics()
        for i in range(60):
            m.update(step=i, reward=0.2)
        assert m.stagnation() is False

    def test_false_when_reward_trending(self):
        m = ControllerMetrics()
        for i in range(60):
            m.update(step=i, reward=0.05 + i * 0.005)
        assert m.stagnation() is False


class TestControllerMetricsConsolidationDetected:
    def test_false_with_insufficient_retention_data(self):
        m = ControllerMetrics()
        for i in range(4):
            m.update(step=i, reward=0.5, retention_rate=0.9)
        assert m.consolidation_detected() is False

    def test_true_when_retention_high_reward_below_saturation(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.5, retention_rate=0.8)
        assert m.consolidation_detected() is True

    def test_false_when_retention_low(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.5, retention_rate=0.4)
        assert m.consolidation_detected() is False

    def test_false_when_reward_already_saturated(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.85, retention_rate=0.9)
        assert m.consolidation_detected() is False


class TestControllerMetricsTacticMonoculture:
    def test_false_with_insufficient_data(self):
        m = ControllerMetrics()
        for i in range(4):
            m.update(step=i, reward=0.3, tactic_diversity=1.5)
        assert m.tactic_monoculture() is False

    def test_true_when_diversity_below_threshold(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.3, tactic_diversity=2.0)
        assert m.tactic_monoculture() is True

    def test_false_when_diversity_above_threshold(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.3, tactic_diversity=5.0)
        assert m.tactic_monoculture() is False

    def test_boundary_exactly_at_threshold_is_false(self):
        m = ControllerMetrics()
        for i in range(10):
            m.update(step=i, reward=0.3, tactic_diversity=3.0)
        # mean == 3.0 which is NOT < 3.0 → False
        assert m.tactic_monoculture() is False
