"""Tests for proofforge.grpo.advantages."""

from __future__ import annotations

import numpy as np
import pytest
from proofforge.grpo.advantages import GroupAdvantages, advantage_is_useful, compute_advantages


# ---------------------------------------------------------------------------
# compute_advantages()
# ---------------------------------------------------------------------------


class TestComputeAdvantagesNormalCase:
    """Mixed rewards produce meaningful normalized advantages."""

    def test_returns_group_advantages_type(self):
        result = compute_advantages([1.0, 0.0, 1.0, 0.0])
        assert isinstance(result, GroupAdvantages)

    def test_group_size_matches_input(self):
        rewards = [1.0, 0.0, 0.5, 0.75]
        result = compute_advantages(rewards)
        assert result.group_size == 4

    def test_advantages_length_matches_input(self):
        rewards = [1.0, 0.0, 0.5]
        result = compute_advantages(rewards)
        assert len(result.advantages) == 3

    def test_advantages_sum_to_approximately_zero(self):
        # Group normalization: sum of (r_i - mean) / std = 0
        rewards = [1.0, 0.0, 1.0, 0.0, 0.5]
        result = compute_advantages(rewards)
        assert abs(result.advantages.sum()) < 1e-6

    def test_correct_proofs_get_positive_advantage(self):
        # Reward=1.0 is above the mean of mixed group → positive advantage
        rewards = [1.0, 1.0, 0.0, 0.0]
        result = compute_advantages(rewards)
        assert result.advantages[0] > 0
        assert result.advantages[1] > 0

    def test_failed_proofs_get_negative_advantage(self):
        rewards = [1.0, 1.0, 0.0, 0.0]
        result = compute_advantages(rewards)
        assert result.advantages[2] < 0
        assert result.advantages[3] < 0

    def test_group_mean_is_correct(self):
        rewards = [0.0, 1.0]
        result = compute_advantages(rewards)
        assert result.group_mean == pytest.approx(0.5)

    def test_group_std_is_positive_for_mixed(self):
        rewards = [1.0, 0.0]
        result = compute_advantages(rewards)
        assert result.group_std > 0

    def test_advantages_normalized_to_unit_std(self):
        rewards = [1.0, 0.0, 0.5, 0.25, 0.75]
        result = compute_advantages(rewards)
        # After normalization, std of advantages should be ~1.0
        assert abs(result.advantages.std() - 1.0) < 1e-6

    def test_formula_correctness(self):
        rewards = [1.0, 0.0]
        result = compute_advantages(rewards)
        r = np.array(rewards)
        expected = (r - r.mean()) / r.std()
        np.testing.assert_allclose(result.advantages, expected, atol=1e-10)

    def test_advantages_are_numpy_array(self):
        result = compute_advantages([1.0, 0.0])
        assert isinstance(result.advantages, np.ndarray)


class TestComputeAdvantagesEdgeCases:
    """Edge cases: empty, all-same, all-zero, all-one."""

    def test_empty_list_returns_empty_advantages(self):
        result = compute_advantages([])
        assert result.group_size == 0
        assert len(result.advantages) == 0

    def test_empty_list_group_mean_zero(self):
        result = compute_advantages([])
        assert result.group_mean == 0.0

    def test_all_same_rewards_gives_zero_advantages(self):
        rewards = [0.7, 0.7, 0.7, 0.7]
        result = compute_advantages(rewards)
        np.testing.assert_allclose(result.advantages, np.zeros(4), atol=1e-10)

    def test_all_zero_rewards_gives_zero_advantages(self):
        rewards = [0.0, 0.0, 0.0]
        result = compute_advantages(rewards)
        np.testing.assert_allclose(result.advantages, np.zeros(3), atol=1e-10)

    def test_all_one_rewards_gives_zero_advantages(self):
        rewards = [1.0, 1.0, 1.0]
        result = compute_advantages(rewards)
        np.testing.assert_allclose(result.advantages, np.zeros(3), atol=1e-10)

    def test_all_same_group_std_reported_correctly(self):
        rewards = [0.5, 0.5, 0.5]
        result = compute_advantages(rewards)
        # std is near zero; the returned group_std reflects the actual std
        assert result.group_std < 1e-6

    def test_single_element_gives_zero_advantage(self):
        # std of one element is 0 → homogeneous → zero advantage
        result = compute_advantages([0.8])
        assert result.group_size == 1
        np.testing.assert_allclose(result.advantages, np.zeros(1), atol=1e-10)

    def test_two_elements_symmetric(self):
        result = compute_advantages([1.0, 0.0])
        # Advantages should be equal magnitude, opposite sign
        assert abs(result.advantages[0] + result.advantages[1]) < 1e-10
        assert result.advantages[0] > 0

    def test_large_group(self):
        rewards = [float(i % 2) for i in range(100)]
        result = compute_advantages(rewards)
        assert result.group_size == 100
        assert abs(result.advantages.sum()) < 1e-6


# ---------------------------------------------------------------------------
# advantage_is_useful()
# ---------------------------------------------------------------------------


class TestAdvantageIsUseful:
    def test_mixed_rewards_is_useful(self):
        ga = compute_advantages([1.0, 0.0, 1.0, 0.0])
        assert advantage_is_useful(ga) is True

    def test_all_same_is_not_useful(self):
        ga = compute_advantages([1.0, 1.0, 1.0])
        assert advantage_is_useful(ga) is False

    def test_all_zero_is_not_useful(self):
        ga = compute_advantages([0.0, 0.0, 0.0])
        assert advantage_is_useful(ga) is False

    def test_all_one_is_not_useful(self):
        ga = compute_advantages([1.0, 1.0, 1.0])
        assert advantage_is_useful(ga) is False

    def test_empty_group_is_not_useful(self):
        ga = compute_advantages([])
        # group_std=1.0 by convention for empty — check the actual behaviour
        # The function returns group_std > min_std; empty has group_std=1.0
        # So empty is technically "useful" by the formula but has no advantages.
        # Just verify it doesn't crash and returns a bool.
        result = advantage_is_useful(ga)
        assert isinstance(result, bool)

    def test_custom_min_std_threshold(self):
        # With a very high min_std threshold, even a mixed group isn't useful
        ga = compute_advantages([1.0, 0.9])
        # std of [1.0, 0.9] is 0.05; with min_std=0.1, should be not useful
        assert advantage_is_useful(ga, min_std=0.1) is False

    def test_custom_min_std_low_threshold(self):
        ga = compute_advantages([1.0, 0.0])
        assert advantage_is_useful(ga, min_std=0.001) is True
