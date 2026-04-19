"""Tests for proofforge.curriculum.expanding_ring."""

from __future__ import annotations

import pytest

from proofforge.curriculum.expanding_ring import ExpandingRing, RingConfig


# ---------------------------------------------------------------------------
# RingConfig defaults
# ---------------------------------------------------------------------------


class TestRingConfigDefaults:
    """RingConfig default values match the documented curriculum settings."""

    def test_default_tiers(self):
        cfg = RingConfig()
        assert cfg.tiers == 4

    def test_default_expansion_threshold(self):
        cfg = RingConfig()
        assert cfg.expansion_threshold == pytest.approx(0.7)

    def test_default_contraction_threshold(self):
        cfg = RingConfig()
        assert cfg.contraction_threshold == pytest.approx(0.1)

    def test_default_min_tier_steps(self):
        cfg = RingConfig()
        assert cfg.min_tier_steps == 20

    def test_default_warmup_fraction(self):
        cfg = RingConfig()
        assert cfg.warmup_fraction == pytest.approx(0.3)

    def test_custom_config(self):
        cfg = RingConfig(
            tiers=3,
            expansion_threshold=0.8,
            contraction_threshold=0.05,
            min_tier_steps=10,
            warmup_fraction=0.5,
        )
        assert cfg.tiers == 3
        assert cfg.expansion_threshold == pytest.approx(0.8)
        assert cfg.contraction_threshold == pytest.approx(0.05)
        assert cfg.min_tier_steps == 10
        assert cfg.warmup_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# add_tier / get_active_items
# ---------------------------------------------------------------------------


class TestAddTierAndGetActiveItems:
    """add_tier registers items; get_active_items returns only active tiers."""

    def test_add_tier_stores_items(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a", "b", "c"])
        assert ring.tiers[0] == ["a", "b", "c"]

    def test_get_active_items_returns_tier_zero_only_at_start(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["easy1", "easy2"])
        ring.add_tier(1, ["medium1"])
        # active_tier defaults to 0
        items = ring.get_active_items()
        assert set(items) == {"easy1", "easy2"}

    def test_get_active_items_does_not_include_inactive_tiers(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["t0"])
        ring.add_tier(1, ["t1"])
        ring.add_tier(2, ["t2"])
        items = ring.get_active_items()
        assert "t1" not in items
        assert "t2" not in items

    def test_get_active_items_after_expand_includes_tier_one(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        items = ring.get_active_items()
        assert "a" in items
        assert "b" in items

    def test_get_active_items_empty_tier_zero(self):
        ring = ExpandingRing()
        # No tier added yet
        items = ring.get_active_items()
        assert items == []

    def test_get_active_items_returns_list(self):
        ring = ExpandingRing()
        ring.add_tier(0, [1, 2, 3])
        assert isinstance(ring.get_active_items(), list)


# ---------------------------------------------------------------------------
# get_frontier_items
# ---------------------------------------------------------------------------


class TestGetFrontierItems:
    """get_frontier_items returns only the current active_tier items."""

    def test_frontier_is_tier_zero_initially(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["easy1", "easy2"])
        ring.add_tier(1, ["hard1"])
        assert set(ring.get_frontier_items()) == {"easy1", "easy2"}

    def test_frontier_is_tier_one_after_expand(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["easy"])
        ring.add_tier(1, ["medium"])
        ring.expand()
        assert ring.get_frontier_items() == ["medium"]

    def test_frontier_does_not_include_lower_tiers(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["t0a", "t0b"])
        ring.add_tier(1, ["t1a"])
        ring.expand()
        frontier = ring.get_frontier_items()
        assert "t0a" not in frontier
        assert "t0b" not in frontier

    def test_frontier_empty_when_no_tier_items(self):
        ring = ExpandingRing()
        # active_tier=0 but tier 0 was never added
        assert ring.get_frontier_items() == []

    def test_frontier_returns_list(self):
        ring = ExpandingRing()
        ring.add_tier(0, [1, 2])
        assert isinstance(ring.get_frontier_items(), list)


# ---------------------------------------------------------------------------
# expand / contract
# ---------------------------------------------------------------------------


class TestExpand:
    """expand() moves to the next tier if it exists."""

    def test_expand_returns_true_when_next_tier_exists(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        assert ring.expand() is True

    def test_expand_increments_active_tier(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert ring.active_tier == 1

    def test_expand_returns_false_when_no_next_tier(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        # No tier 1
        assert ring.expand() is False

    def test_expand_does_not_change_tier_when_no_next(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.expand()
        assert ring.active_tier == 0

    def test_expand_records_history(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert len(ring.expansion_history) == 1
        assert ring.expansion_history[0]["action"] == "expand"
        assert ring.expansion_history[0]["from_tier"] == 0
        assert ring.expansion_history[0]["to_tier"] == 1

    def test_expand_twice_reaches_tier_two(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.add_tier(2, ["c"])
        ring.expand()
        ring.expand()
        assert ring.active_tier == 2


class TestContract:
    """contract() moves to the previous tier if not already at 0."""

    def test_contract_returns_false_at_tier_zero(self):
        ring = ExpandingRing()
        assert ring.contract() is False

    def test_contract_does_not_change_tier_at_zero(self):
        ring = ExpandingRing()
        ring.contract()
        assert ring.active_tier == 0

    def test_contract_returns_true_from_tier_one(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert ring.contract() is True

    def test_contract_decrements_active_tier(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        ring.contract()
        assert ring.active_tier == 0

    def test_contract_records_history(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        ring.contract()
        contract_events = [e for e in ring.expansion_history if e["action"] == "contract"]
        assert len(contract_events) == 1
        assert contract_events[0]["from_tier"] == 1
        assert contract_events[0]["to_tier"] == 0


# ---------------------------------------------------------------------------
# should_expand
# ---------------------------------------------------------------------------


class TestShouldExpand:
    """should_expand() is True when pass rate exceeds threshold for enough steps."""

    def _make_ring_with_config(self, min_steps: int = 5) -> ExpandingRing:
        cfg = RingConfig(min_tier_steps=min_steps, expansion_threshold=0.7)
        return ExpandingRing(config=cfg)

    def test_returns_true_when_conditions_met(self):
        ring = self._make_ring_with_config(min_steps=5)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        # Record 5+ steps, all high pass rates
        for _ in range(6):
            ring.record_pass_rate(0, 0.9)
        assert ring.should_expand() is True

    def test_returns_false_insufficient_steps(self):
        ring = self._make_ring_with_config(min_steps=20)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        # Only 3 steps recorded, below min_tier_steps=20
        for _ in range(3):
            ring.record_pass_rate(0, 0.95)
        assert ring.should_expand() is False

    def test_returns_false_when_fewer_than_5_rates(self):
        ring = self._make_ring_with_config(min_steps=5)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        # 5 steps taken but only 4 rates recorded — should_expand checks len(rates) < 5
        for _ in range(4):
            ring.record_pass_rate(0, 0.9)
        # Manually set tier_steps to meet min_steps requirement
        ring.tier_steps[0] = 20
        assert ring.should_expand() is False

    def test_returns_false_when_pass_rate_too_low(self):
        ring = self._make_ring_with_config(min_steps=5)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        for _ in range(6):
            ring.record_pass_rate(0, 0.3)  # below 0.7 threshold
        assert ring.should_expand() is False

    def test_uses_recent_5_rates_not_all(self):
        ring = self._make_ring_with_config(min_steps=5)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        # First 5 rates are low, last 5 are high
        for _ in range(5):
            ring.record_pass_rate(0, 0.1)
        for _ in range(5):
            ring.record_pass_rate(0, 0.9)
        # should_expand uses the most recent 5 rates (all 0.9)
        assert ring.should_expand() is True

    def test_returns_false_with_no_rates_recorded(self):
        ring = self._make_ring_with_config(min_steps=5)
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        assert ring.should_expand() is False


# ---------------------------------------------------------------------------
# should_contract
# ---------------------------------------------------------------------------


class TestShouldContract:
    """should_contract() is True when pass rate falls below contraction threshold."""

    def _make_ring_at_tier_one(self) -> ExpandingRing:
        cfg = RingConfig(contraction_threshold=0.1)
        ring = ExpandingRing(config=cfg)
        ring.add_tier(0, ["easy"])
        ring.add_tier(1, ["hard"])
        ring.expand()
        return ring

    def test_returns_false_at_tier_zero(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        assert ring.should_contract() is False

    def test_returns_true_when_pass_rate_very_low(self):
        ring = self._make_ring_at_tier_one()
        for _ in range(10):
            ring.record_pass_rate(1, 0.05)  # below 0.1 threshold
        assert ring.should_contract() is True

    def test_returns_false_when_not_enough_rates(self):
        ring = self._make_ring_at_tier_one()
        for _ in range(5):
            ring.record_pass_rate(1, 0.05)
        # need at least 10 rates for contraction check
        assert ring.should_contract() is False

    def test_returns_false_when_pass_rate_acceptable(self):
        ring = self._make_ring_at_tier_one()
        for _ in range(10):
            ring.record_pass_rate(1, 0.5)  # above 0.1 threshold
        assert ring.should_contract() is False

    def test_uses_recent_10_rates(self):
        ring = self._make_ring_at_tier_one()
        # Early rates are fine, recent 10 are all terrible
        for _ in range(10):
            ring.record_pass_rate(1, 0.8)
        for _ in range(10):
            ring.record_pass_rate(1, 0.02)
        assert ring.should_contract() is True


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


class TestStep:
    """step() auto-expands or contracts, returning the action taken."""

    def _make_fast_ring(self) -> ExpandingRing:
        """Ring with min_tier_steps=5 so tests don't need 20+ iterations."""
        cfg = RingConfig(
            expansion_threshold=0.7,
            contraction_threshold=0.1,
            min_tier_steps=5,
        )
        return ExpandingRing(config=cfg)

    def test_step_returns_hold_normally(self):
        ring = self._make_fast_ring()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        result = ring.step({0: 0.5})
        assert result == "hold"

    def test_step_returns_expand_when_threshold_met(self):
        ring = self._make_fast_ring()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        # should_expand requires steps >= min_tier_steps (5) AND len(rates) >= 5.
        # record_pass_rate is called once per step, so both counters are equal.
        # The 5th step call brings steps=5 and len(rates)=5, which satisfies
        # both conditions — expansion fires on that call.
        results = [ring.step({0: 0.9}) for _ in range(5)]
        assert "expand" in results

    def test_step_returns_contract_when_failing(self):
        ring = self._make_fast_ring()
        ring.add_tier(0, ["easy"])
        ring.add_tier(1, ["hard"])
        ring.expand()  # move to tier 1
        # should_contract requires len(rates) >= 10 for the active tier.
        # The 10th step call at tier 1 satisfies the condition.
        results = [ring.step({1: 0.02}) for _ in range(10)]
        assert "contract" in results

    def test_step_records_pass_rates(self):
        ring = self._make_fast_ring()
        ring.add_tier(0, ["a"])
        ring.step({0: 0.6})
        assert 0 in ring.tier_pass_rates
        assert 0.6 in ring.tier_pass_rates[0]

    def test_step_records_multiple_tiers(self):
        ring = self._make_fast_ring()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        ring.step({0: 0.5, 1: 0.3})
        assert 0 in ring.tier_pass_rates
        assert 1 in ring.tier_pass_rates


# ---------------------------------------------------------------------------
# record_pass_rate
# ---------------------------------------------------------------------------


class TestRecordPassRate:
    """record_pass_rate() accumulates rates and increments step counters."""

    def test_first_record_creates_list(self):
        ring = ExpandingRing()
        ring.record_pass_rate(0, 0.5)
        assert ring.tier_pass_rates[0] == [0.5]

    def test_multiple_records_accumulate(self):
        ring = ExpandingRing()
        ring.record_pass_rate(0, 0.3)
        ring.record_pass_rate(0, 0.6)
        ring.record_pass_rate(0, 0.9)
        assert ring.tier_pass_rates[0] == [0.3, 0.6, 0.9]

    def test_record_increments_step_counter(self):
        ring = ExpandingRing()
        ring.record_pass_rate(0, 0.5)
        ring.record_pass_rate(0, 0.5)
        assert ring.tier_steps[0] == 2

    def test_record_different_tiers_independently(self):
        ring = ExpandingRing()
        ring.record_pass_rate(0, 0.4)
        ring.record_pass_rate(1, 0.8)
        assert ring.tier_pass_rates[0] == [0.4]
        assert ring.tier_pass_rates[1] == [0.8]

    def test_step_counter_initialises_at_zero_for_tier_zero(self):
        ring = ExpandingRing()
        assert ring.tier_steps[0] == 0

    def test_step_counter_created_for_new_tier(self):
        ring = ExpandingRing()
        ring.record_pass_rate(2, 0.7)
        assert ring.tier_steps[2] == 1


# ---------------------------------------------------------------------------
# expansion_history
# ---------------------------------------------------------------------------


class TestExpansionHistory:
    """expansion_history tracks all expand and contract events."""

    def test_initially_empty(self):
        ring = ExpandingRing()
        assert ring.expansion_history == []

    def test_expand_appends_event(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert len(ring.expansion_history) == 1

    def test_expand_event_has_action_expand(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert ring.expansion_history[0]["action"] == "expand"

    def test_contract_event_has_action_contract(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        ring.contract()
        actions = [e["action"] for e in ring.expansion_history]
        assert "contract" in actions

    def test_multiple_events_recorded_in_order(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.add_tier(2, ["c"])
        ring.expand()
        ring.expand()
        ring.contract()
        assert len(ring.expansion_history) == 3
        assert ring.expansion_history[0]["action"] == "expand"
        assert ring.expansion_history[1]["action"] == "expand"
        assert ring.expansion_history[2]["action"] == "contract"

    def test_failed_expand_does_not_append_event(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        # No tier 1 — expand() returns False
        ring.expand()
        assert len(ring.expansion_history) == 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a dict representing the current curriculum state."""

    def test_returns_dict(self):
        ring = ExpandingRing()
        assert isinstance(ring.summary(), dict)

    def test_active_tier_in_summary(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        assert ring.summary()["active_tier"] == 1

    def test_total_tiers_reflects_registered_tiers(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b", "c"])
        ring.add_tier(2, ["d"])
        assert ring.summary()["total_tiers"] == 3

    def test_items_per_tier_counts_correctly(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a", "b"])
        ring.add_tier(1, ["c"])
        summary = ring.summary()
        assert summary["items_per_tier"][0] == 2
        assert summary["items_per_tier"][1] == 1

    def test_active_items_count_correct(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a", "b"])
        ring.add_tier(1, ["c", "d"])
        ring.expand()
        assert ring.summary()["active_items"] == 4

    def test_expansion_history_in_summary(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.add_tier(1, ["b"])
        ring.expand()
        summary = ring.summary()
        assert "expansion_history" in summary
        assert len(summary["expansion_history"]) == 1

    def test_tier_pass_rates_in_summary(self):
        ring = ExpandingRing()
        ring.add_tier(0, ["a"])
        ring.record_pass_rate(0, 0.8)
        ring.record_pass_rate(0, 0.9)
        summary = ring.summary()
        assert "tier_pass_rates" in summary
        assert 0 in summary["tier_pass_rates"]
