"""Tests for tactic diversity bonus in /reward endpoint.

Bug 4: The DIVERSIFY_TACTICS intervention fires but has no effect on the
reward signal. The diversity bonus must only apply to CORRECT proofs.
"""

from __future__ import annotations

import json

import pytest

from train_server import apply_tactic_diversity_bonus, app


class TestDiversityBonusCorrectness:
    """OT gate: diversity bonus must NOT apply to wrong proofs."""

    def test_bonus_only_on_correct_proofs(self):
        rewards = [[0.7, 0.0, 1.0]]  # omega=correct, cases=WRONG, simp=correct
        proofs = [["omega", "cases h", "simp"]]
        tactic_usage = {"simp": 0.9, "omega": 0.08, "cases": 0.02}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        # omega (correct + rare, 0.7) should get bonus → 0.85
        assert rewards[0][0] == pytest.approx(0.85)
        # cases (WRONG + rare) must NOT get bonus
        assert rewards[0][1] == 0.0
        # simp (correct + common) → no bonus (usage 0.9 >= 0.1)
        assert rewards[0][2] == 1.0

    def test_no_bonus_without_tactic_usage(self):
        rewards = [[1.0, 0.0]]
        proofs = [["omega", "simp"]]

        apply_tactic_diversity_bonus(rewards, proofs, None)

        assert rewards[0] == [1.0, 0.0]

    def test_no_bonus_for_common_tactics(self):
        rewards = [[1.0]]
        proofs = [["simp"]]
        tactic_usage = {"simp": 0.9}  # very common

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        assert rewards[0][0] == 1.0  # no bonus

    def test_bonus_for_rare_correct_tactic(self):
        rewards = [[0.8]]
        proofs = [["ring"]]
        tactic_usage = {"ring": 0.05, "simp": 0.9}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        assert rewards[0][0] == pytest.approx(0.95)  # 0.8 + 0.15

    def test_bonus_capped_at_one(self):
        rewards = [[0.95]]
        proofs = [["ring"]]
        tactic_usage = {"ring": 0.01}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        assert rewards[0][0] == 1.0  # min(1.0, 0.95 + 0.15)

    def test_empty_proof_skipped(self):
        rewards = [[1.0, 0.5]]
        proofs = [["", "omega"]]
        tactic_usage = {"omega": 0.05}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        assert rewards[0][0] == 1.0  # empty proof unchanged
        assert rewards[0][1] == pytest.approx(0.65)  # 0.5 + 0.15

    def test_unknown_tactic_treated_as_common(self):
        """Tactics not in tactic_usage default to 1.0 (common) — no bonus."""
        rewards = [[1.0]]
        proofs = [["unfold_mystery"]]
        tactic_usage = {"simp": 0.9}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        assert rewards[0][0] == 1.0  # unknown → default 1.0 → no bonus


class TestDiversityBonusInvariant:
    """Invariant: every proof with adv > 0 must have reward > 0."""

    def test_wrong_proof_never_gets_positive_reward(self):
        """No matter the tactic rarity, wrong proofs stay at 0."""
        rewards = [[0.0, 0.0, 0.0]]
        proofs = [["omega", "cases h with x", "ring"]]
        tactic_usage = {"omega": 0.01, "cases": 0.01, "ring": 0.01}

        apply_tactic_diversity_bonus(rewards, proofs, tactic_usage)

        for r in rewards[0]:
            assert r == 0.0, "Wrong proof must never receive diversity bonus"


class TestDiversityBonusIntegration:
    """Integration test: POST to /reward endpoint with tactic_usage."""

    def test_reward_endpoint_applies_diversity_bonus(self):
        """The live /reward endpoint applies the bonus, not just the extracted function."""
        client = app.test_client()
        response = client.post("/reward", json={
            "statements": ["theorem t : 1 + 1 = 2 := by"],
            "proofs": [["omega", "cases h", "simp"]],
            "rewards": [[0.8, 0.0, 0.8]],
            "tactic_usage": {"simp": 0.9, "omega": 0.02, "cases": 0.01},
        })
        data = json.loads(response.data)
        advantages = data["advantages"][0]
        # omega (correct + rare) should have highest advantage
        # cases (wrong + rare) should have negative advantage
        assert advantages[1] < 0, "Wrong proof must have negative advantage"
        assert advantages[0] > advantages[2], "Rare correct tactic should beat common correct tactic"
