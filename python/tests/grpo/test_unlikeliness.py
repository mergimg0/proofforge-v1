"""Tests for proofforge.grpo.unlikeliness."""

from __future__ import annotations

import pytest
from proofforge.grpo.unlikeliness import unlikeliness_reweight


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestUnlikelinessReweightBasic:
    def test_returns_list(self):
        result = unlikeliness_reweight([1.0, 0.0], [-0.5, -1.0])
        assert isinstance(result, list)

    def test_output_length_matches_input(self):
        rewards = [1.0, 0.0, 1.0]
        log_probs = [-0.1, -0.5, -2.0]
        result = unlikeliness_reweight(rewards, log_probs)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        result = unlikeliness_reweight([], [])
        assert result == []

    def test_all_rewards_still_in_valid_range(self):
        rewards = [1.0, 1.0, 0.0, 1.0]
        log_probs = [-0.1, -0.5, -1.0, -3.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.25)
        for r in result:
            assert r >= 0.0
            assert r <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Rank-based penalisation
# ---------------------------------------------------------------------------


class TestUnlikelinessReweightRanking:
    """Most-probable completion is penalised; least-probable retains full reward."""

    def test_most_likely_penalised(self):
        # rank 0 = most probable (highest log_prob = least negative)
        # factor = 1 - beta_rank * (G - 0) / G = 1 - beta_rank
        rewards = [1.0, 1.0]
        log_probs = [-0.1, -2.0]   # index 0 is most probable
        beta = 0.25
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=beta)
        # Most probable (index 0): factor = 1 - 0.25 * (2 - 0) / 2 = 1 - 0.25 = 0.75
        assert result[0] == pytest.approx(0.75)

    def test_least_likely_retains_full_reward(self):
        # rank G-1 = least probable (lowest log_prob = most negative)
        # factor = 1 - beta_rank * (G - (G-1)) / G = 1 - beta_rank * 1/G
        rewards = [1.0, 1.0]
        log_probs = [-0.1, -2.0]   # index 1 is least probable
        beta = 0.25
        n = 2
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=beta)
        # Least probable (index 1): factor = 1 - 0.25 * (2-1)/2 = 1 - 0.125 = 0.875
        expected = 1.0 * (1.0 - beta * 1 / n)
        assert result[1] == pytest.approx(expected)

    def test_most_likely_reward_lower_than_least_likely(self):
        rewards = [1.0, 1.0, 1.0]
        log_probs = [-0.1, -1.0, -5.0]  # index 0 most probable, index 2 least
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.3)
        assert result[0] < result[2]

    def test_monotone_factor_with_probability(self):
        # Higher probability (less negative log_prob) → lower factor
        rewards = [1.0, 1.0, 1.0]
        log_probs = [-0.1, -1.0, -5.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.25)
        assert result[0] <= result[1] <= result[2]

    def test_three_element_factors(self):
        # G=3, beta=0.25
        # rank 0 (most probable): factor = 1 - 0.25*(3-0)/3 = 1 - 0.25 = 0.75
        # rank 1 (mid):           factor = 1 - 0.25*(3-1)/3 = 1 - 0.1667 ≈ 0.8333
        # rank 2 (least probable):factor = 1 - 0.25*(3-2)/3 = 1 - 0.0833 ≈ 0.9167
        n = 3
        beta = 0.25
        rewards = [1.0, 1.0, 1.0]
        log_probs = [-0.1, -1.0, -5.0]  # index 0 most probable
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=beta)
        assert result[0] == pytest.approx(1.0 - beta * n / n)
        assert result[1] == pytest.approx(1.0 - beta * (n - 1) / n)
        assert result[2] == pytest.approx(1.0 - beta * 1 / n)


# ---------------------------------------------------------------------------
# Zero rewards stay zero
# ---------------------------------------------------------------------------


class TestUnlikelinessReweightZeroRewards:
    def test_zero_reward_stays_zero_regardless_of_rank(self):
        rewards = [0.0, 0.0, 0.0]
        log_probs = [-0.1, -1.0, -5.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.5)
        assert all(r == pytest.approx(0.0) for r in result)

    def test_mixed_rewards_zero_stays_zero(self):
        rewards = [1.0, 0.0, 1.0]
        log_probs = [-0.1, -0.5, -3.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.25)
        assert result[1] == pytest.approx(0.0)

    def test_all_zero_with_high_beta(self):
        rewards = [0.0, 0.0]
        log_probs = [-0.1, -2.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=1.0)
        assert all(r == pytest.approx(0.0) for r in result)


# ---------------------------------------------------------------------------
# beta_rank=0 → no reweighting
# ---------------------------------------------------------------------------


class TestUnlikelinessReweightBetaZero:
    def test_beta_zero_no_change(self):
        rewards = [1.0, 0.0, 1.0]
        log_probs = [-0.1, -0.5, -3.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.0)
        assert result[0] == pytest.approx(rewards[0])
        assert result[1] == pytest.approx(rewards[1])
        assert result[2] == pytest.approx(rewards[2])

    def test_beta_zero_single_element(self):
        result = unlikeliness_reweight([1.0], [-0.5], beta_rank=0.0)
        assert result[0] == pytest.approx(1.0)

    def test_beta_zero_preserves_zeros(self):
        rewards = [0.0, 0.0]
        log_probs = [-0.1, -2.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.0)
        assert all(r == pytest.approx(0.0) for r in result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestUnlikelinessReweightEdgeCases:
    def test_single_element_most_and_least_probable(self):
        # G=1: rank 0 = only element
        # factor = 1 - beta * (1 - 0) / 1 = 1 - beta
        beta = 0.25
        result = unlikeliness_reweight([1.0], [-0.5], beta_rank=beta)
        assert result[0] == pytest.approx(1.0 - beta)

    def test_equal_log_probs_produces_valid_output(self):
        # Ties in log_prob → argsort breaks ties arbitrarily; output still valid
        rewards = [1.0, 1.0, 1.0]
        log_probs = [-1.0, -1.0, -1.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=0.25)
        assert len(result) == 3
        for r in result:
            assert 0.0 <= r <= 1.0 + 1e-9

    def test_large_beta_rank(self):
        # beta_rank=1.0: most probable gets factor = 1 - 1.0 * G/G = 0.0
        rewards = [1.0, 1.0]
        log_probs = [-0.1, -2.0]
        result = unlikeliness_reweight(rewards, log_probs, beta_rank=1.0)
        assert result[0] == pytest.approx(0.0)

    def test_output_is_plain_list_not_numpy(self):
        result = unlikeliness_reweight([1.0, 0.0], [-0.5, -1.5])
        assert isinstance(result, list)
        # Elements should be plain Python floats or at least numeric
        assert isinstance(result[0], (int, float))
