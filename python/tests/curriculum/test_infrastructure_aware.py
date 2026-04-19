"""Tests for proofforge.curriculum.infrastructure_aware."""

from __future__ import annotations

import pytest

from proofforge.curriculum.infrastructure_aware import InfrastructureAwareSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_item(infra: tuple[str, ...]) -> dict:
    """Simple dict-based item carrying required infrastructure."""
    return {"infra": infra}


def get_infra(item: dict) -> tuple[str, ...]:
    return item["infra"]


def mastery_fn_uniform(rate: float):
    """Return a mastery function that gives every pattern the same rate."""
    return lambda _pattern: rate


def mastery_fn_map(rates: dict[str, float]):
    """Return a mastery function backed by an explicit pattern→rate mapping."""
    return lambda p: rates.get(p, 0.0)


def usage_fn_map(usages: dict[str, float]):
    """Return a tactic usage function backed by an explicit pattern→usage mapping."""
    return lambda p: usages.get(p, 0.0)


# ---------------------------------------------------------------------------
# score() — base infrastructure frontier score
# ---------------------------------------------------------------------------


class TestScoreHighForMixedMastery:
    """score() is high when required patterns have mixed mastery levels."""

    def test_mixed_mastery_gives_positive_score(self):
        # one pattern mastered (0.9), one not (0.1) → high std, non-zero mean
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"recursion": 0.9, "iteration": 0.1}),
        )
        item = make_item(("recursion", "iteration"))
        score = sampler.score(item)
        assert score > 0.0

    def test_mixed_mastery_higher_than_uniform(self):
        # Uniform mastery → std=0 → score=0; mixed → score>0
        sampler_uniform = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        sampler_mixed = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"recursion": 0.9, "iteration": 0.05}),
        )
        item = make_item(("recursion", "iteration"))
        assert sampler_mixed.score(item) > sampler_uniform.score(item)


class TestScoreLowForUniformMastery:
    """score() is near zero when all required patterns have the same mastery."""

    def test_all_fully_mastered_score_is_zero(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(1.0),
        )
        item = make_item(("recursion", "iteration", "error_handling"))
        # std([1.0, 1.0, 1.0]) == 0 → score == 0
        assert sampler.score(item) == pytest.approx(0.0, abs=1e-5)

    def test_all_unmastered_score_is_zero(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.0),
        )
        item = make_item(("recursion", "iteration"))
        # std([0.0, 0.0]) == 0 → score == 0
        assert sampler.score(item) == pytest.approx(0.0, abs=1e-5)

    def test_uniform_half_mastery_score_is_zero(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        item = make_item(("a", "b", "c"))
        assert sampler.score(item) == pytest.approx(0.0, abs=1e-5)


class TestScoreWithNoRequiredInfra:
    """score() returns 0.0 when an item has no required infrastructure."""

    def test_empty_infra_returns_zero(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.8),
        )
        item = make_item(())
        assert sampler.score(item) == pytest.approx(0.0)

    def test_empty_infra_returns_zero_regardless_of_mastery(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"recursion": 0.9, "iteration": 0.1}),
        )
        item = make_item(())
        assert sampler.score(item) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score() — diversity bonus (rare tactics, usage < 0.1)
# ---------------------------------------------------------------------------


class TestDiversityBonus:
    """Items requiring rare tactics (usage < 0.1) get the diversity_bonus multiplier."""

    def test_rare_tactic_applies_diversity_bonus(self):
        # "rare_tactic" has usage 0.05 (<0.1) → bonus applied
        sampler_with = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "rare_tactic": 0.1}),
            tactic_usage_fn=usage_fn_map({"common": 0.8, "rare_tactic": 0.05}),
            diversity_bonus=2.0,
        )
        sampler_without = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "rare_tactic": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("common", "rare_tactic"))
        score_with = sampler_with.score(item)
        score_without = sampler_without.score(item)
        # diversity bonus doubles the base score
        assert score_with == pytest.approx(score_without * 2.0, rel=1e-5)

    def test_diversity_bonus_is_exactly_2x(self):
        usages = {"p1": 0.9, "p2": 0.05}  # p2 is rare
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"p1": 0.9, "p2": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
            diversity_bonus=2.0,
        )
        sampler_no_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"p1": 0.9, "p2": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("p1", "p2"))
        assert sampler.score(item) == pytest.approx(
            sampler_no_usage.score(item) * 2.0, rel=1e-5
        )

    def test_no_rare_tactic_no_diversity_bonus(self):
        # All usages above 0.1 → no diversity bonus
        usages = {"p1": 0.6, "p2": 0.5}
        sampler_with_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"p1": 0.9, "p2": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
            diversity_bonus=2.0,
        )
        sampler_no_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"p1": 0.9, "p2": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("p1", "p2"))
        # No rare tactic and not all common → no modifier → same base score
        # (but monoculture penalty also doesn't apply since not all > 0.5)
        assert sampler_with_usage.score(item) == pytest.approx(
            sampler_no_usage.score(item), rel=1e-5
        )


# ---------------------------------------------------------------------------
# score() — monoculture penalty (all common tactics, usage > 0.5)
# ---------------------------------------------------------------------------


class TestMonoculturePenalty:
    """Items requiring only over-represented tactics get the monoculture_penalty multiplier."""

    def test_all_common_tactics_applies_penalty(self):
        # Both tactics have usage > 0.5 → monoculture penalty
        usages = {"simp": 0.9, "rfl": 0.8}
        sampler_with = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"simp": 0.9, "rfl": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
            monoculture_penalty=0.3,
        )
        sampler_without = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"simp": 0.9, "rfl": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("simp", "rfl"))
        score_with = sampler_with.score(item)
        score_without = sampler_without.score(item)
        assert score_with == pytest.approx(score_without * 0.3, rel=1e-5)

    def test_monoculture_penalty_is_exactly_03x(self):
        usages = {"dominant": 0.7, "other": 0.6}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"dominant": 0.9, "other": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
            monoculture_penalty=0.3,
        )
        sampler_no_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"dominant": 0.9, "other": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("dominant", "other"))
        assert sampler.score(item) == pytest.approx(
            sampler_no_usage.score(item) * 0.3, rel=1e-5
        )

    def test_mixed_usage_no_penalty_no_bonus(self):
        # One common (>0.5), one mid-range — not all common, not any rare
        usages = {"common": 0.8, "mid": 0.3}
        sampler_with = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "mid": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
        )
        sampler_no_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "mid": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("common", "mid"))
        assert sampler_with.score(item) == pytest.approx(
            sampler_no_usage.score(item), rel=1e-5
        )

    def test_rare_tactic_overrides_monoculture_check(self):
        # has_rare=True takes precedence over all_common check in the implementation
        usages = {"common": 0.8, "rare": 0.05}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "rare": 0.1}),
            tactic_usage_fn=usage_fn_map(usages),
            diversity_bonus=2.0,
            monoculture_penalty=0.3,
        )
        sampler_no_usage = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"common": 0.9, "rare": 0.1}),
            tactic_usage_fn=None,
        )
        item = make_item(("common", "rare"))
        # has_rare → bonus branch taken, not penalty
        assert sampler.score(item) == pytest.approx(
            sampler_no_usage.score(item) * 2.0, rel=1e-5
        )


# ---------------------------------------------------------------------------
# score() — without tactic_usage_fn: no diversity modifier
# ---------------------------------------------------------------------------


class TestScoreWithoutTacticUsageFn:
    """When tactic_usage_fn is None, no diversity modifier is applied."""

    def test_no_usage_fn_score_equals_base_formula(self):
        import numpy as np
        masteries = {"p1": 0.9, "p2": 0.1}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map(masteries),
            tactic_usage_fn=None,
        )
        item = make_item(("p1", "p2"))
        arr = np.array([0.9, 0.1])
        expected_base = float(np.std(arr) * np.mean(arr))
        # score() clamps to max(base, 1e-6) but base > 0 here
        assert sampler.score(item) == pytest.approx(expected_base, rel=1e-5)

    def test_no_usage_fn_same_as_zero_usage(self):
        # Without usage fn there is no modifier at all — score is purely base
        sampler_none = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"a": 0.8, "b": 0.2}),
            tactic_usage_fn=None,
        )
        item = make_item(("a", "b"))
        score = sampler_none.score(item)
        # Re-compute to confirm no modifier path was taken
        import numpy as np
        arr = np.array([0.8, 0.2])
        assert score == pytest.approx(float(np.std(arr) * np.mean(arr)), rel=1e-5)


# ---------------------------------------------------------------------------
# score() always returns a non-negative float
# ---------------------------------------------------------------------------


class TestScoreNonNegative:
    """score() never returns a negative value or zero for non-empty infra."""

    def test_score_is_non_negative_for_mixed(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"a": 0.9, "b": 0.05}),
        )
        assert sampler.score(make_item(("a", "b"))) >= 0.0

    def test_score_minimum_is_1e6_for_nonempty_infra(self):
        # Implementation clamps to max(base, 1e-6)
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),  # uniform → base=0
        )
        item = make_item(("x", "y"))
        assert sampler.score(item) >= 1e-6


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


class TestSelect:
    """select() returns n items, biased toward high-scoring ones."""

    def test_select_returns_n_items(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"a": 0.9, "b": 0.1}),
        )
        items = [make_item(("a", "b")) for _ in range(10)]
        result = sampler.select(items, n=3)
        assert len(result) == 3

    def test_select_empty_items_returns_empty(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        assert sampler.select([], n=5) == []

    def test_select_n_1_returns_single_item(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"a": 0.9, "b": 0.1}),
        )
        items = [make_item(("a", "b")) for _ in range(5)]
        result = sampler.select(items, n=1)
        assert len(result) == 1

    def test_select_returns_items_from_input(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map({"a": 0.9, "b": 0.1}),
        )
        items = [make_item(("a", "b")) for _ in range(10)]
        result = sampler.select(items, n=4)
        for item in result:
            assert item in items

    def test_select_n_larger_than_items_clips(self):
        # n > len(items): implementation clips to available items
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        items = [make_item(("x",)) for _ in range(3)]
        result = sampler.select(items, n=100)
        # Should not crash and should return at most n items
        assert len(result) <= 100

    def test_select_high_score_items_sampled_more_often(self):
        # Run many selections with n=2; compare frontier (exploitation) picks.
        # With n=2: n_uniform=max(1,int(2*0.1))=1 uniform + 1 frontier.
        # The frontier pick should favor the high-score item.
        import collections
        masteries = {"rare_high": 0.9, "rare_low": 0.1, "common": 0.5}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map(masteries),
            uniform_fraction=0.1,  # 10% uniform, 90% exploitation
        )
        high_item = make_item(("rare_high", "rare_low"))   # high std * mean
        low_item = make_item(("common",))                  # single item → std=0

        items = [high_item, low_item]
        counts: dict = collections.defaultdict(int)
        for _ in range(500):
            selected = sampler.select(items, n=2)
            for s in selected:
                counts[s["infra"]] += 1

        # high_item score >> low_item score → should appear significantly more
        assert counts[("rare_high", "rare_low")] > counts[("common",)], (
            f"Expected high-frontier item to dominate: {dict(counts)}"
        )


# ---------------------------------------------------------------------------
# rank()
# ---------------------------------------------------------------------------


class TestRank:
    """rank() returns items sorted by score descending."""

    def test_rank_returns_list_of_tuples(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        items = [make_item(("a",)), make_item(("b",))]
        result = sampler.rank(items)
        assert isinstance(result, list)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_rank_is_descending_by_score(self):
        masteries = {"high_a": 0.9, "low_a": 0.05, "uniform_b": 0.5, "uniform_c": 0.5}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map(masteries),
        )
        # item_mixed has high std*mean; item_uniform has zero std → score≈0
        item_mixed = make_item(("high_a", "low_a"))
        item_uniform = make_item(("uniform_b", "uniform_c"))
        ranked = sampler.rank([item_uniform, item_mixed])
        # item_mixed should rank first
        assert ranked[0][0] == item_mixed
        assert ranked[1][0] == item_uniform

    def test_rank_scores_are_non_increasing(self):
        masteries = {
            "p1": 0.9, "p2": 0.1,   # high spread
            "p3": 0.7, "p4": 0.3,   # medium spread
            "p5": 0.5, "p6": 0.5,   # no spread
        }
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map(masteries),
        )
        items = [
            make_item(("p5", "p6")),  # uniform
            make_item(("p1", "p2")),  # high spread
            make_item(("p3", "p4")),  # medium spread
        ]
        ranked = sampler.rank(items)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_preserves_all_items(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        items = [make_item((f"p{i}",)) for i in range(5)]
        ranked = sampler.rank(items)
        assert len(ranked) == len(items)
        ranked_items = [item for item, _ in ranked]
        for original in items:
            assert original in ranked_items

    def test_rank_empty_list(self):
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_uniform(0.5),
        )
        assert sampler.rank([]) == []

    def test_rank_scores_match_score_method(self):
        masteries = {"a": 0.8, "b": 0.2, "c": 0.5}
        sampler = InfrastructureAwareSampler(
            get_required_infra=get_infra,
            mastery_fn=mastery_fn_map(masteries),
        )
        items = [make_item(("a", "b")), make_item(("b", "c")), make_item(("a",))]
        ranked = sampler.rank(items)
        for item, score in ranked:
            assert score == pytest.approx(sampler.score(item), rel=1e-6)
