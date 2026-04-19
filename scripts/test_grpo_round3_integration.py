"""Integration tests for grpo_round3_integration.py.

Tests the bridge between the proofforge library modules (efficiency, controller)
and the training script. No GPU required — uses mock oracles and synthetic data.
"""

import pytest
from grpo_round3_integration import (
    create_efficiency_oracle,
    create_controller,
    compute_efficiency_rewards,
    controller_step,
    should_switch_to_efficiency,
    should_expand_dataset,
    tactic_diversity_warning,
    RetentionTracker,
    count_distinct_tactics,
    extract_first_tactic,
)


class TestEfficiencyOracleCreation:
    def test_default_config(self):
        oracle = create_efficiency_oracle()
        assert oracle.config.alpha_mastered == 0.5
        assert oracle.config.phase_in_step == 50

    def test_custom_config(self):
        oracle = create_efficiency_oracle(alpha_mastered=0.3, phase_in_step=100)
        assert oracle.config.alpha_mastered == 0.3
        assert oracle.config.phase_in_step == 100

    def test_binary_before_phase_in(self):
        oracle = create_efficiency_oracle(phase_in_step=50)
        thm = {"id": "test", "statement": "theorem t : True := by"}
        comps = [{"text": "trivial", "token_ids": [1, 2, 3]}]
        rewards = compute_efficiency_rewards(
            oracle, thm, comps, [True], [1.0], step=10, pass_rate=0.5
        )
        assert rewards == [1.0], "Before phase-in, should return binary reward"

    def test_zero_for_unverified(self):
        oracle = create_efficiency_oracle(phase_in_step=0)
        thm = {"id": "test", "statement": "theorem t : True := by"}
        comps = [{"text": "sorry", "token_ids": [1, 2, 3]}]
        rewards = compute_efficiency_rewards(
            oracle, thm, comps, [False], [0.0], step=100, pass_rate=0.5
        )
        assert rewards == [0.0], "Unverified should always return 0.0"

    def test_efficiency_depresses_long_proofs(self):
        """After phase-in with mastered theorem, long proofs get < 1.0."""
        oracle = create_efficiency_oracle(phase_in_step=0, phase_in_pass_rate=0.0)
        thm = {"id": "t1", "statement": "theorem t : True := by"}
        # Feed breadth data: 4+ attempts at 3 temperatures
        for _ in range(5):
            for temp in [0.3, 0.7, 1.2]:
                oracle.breadth_tracker.record("t1", temp, True, 10)
        # Now breadth >= 3 → alpha_mastered = 0.5
        long_comp = [{"text": "x " * 200, "token_ids": list(range(200))}]
        rewards = compute_efficiency_rewards(
            oracle, thm, long_comp, [True], [1.0], step=100, pass_rate=0.5
        )
        assert 0.0 < rewards[0] < 1.0, f"Long proof should be depressed, got {rewards[0]}"

    def test_multiple_completions(self):
        oracle = create_efficiency_oracle(phase_in_step=50)
        thm = {"id": "test", "statement": "theorem t : True := by"}
        comps = [
            {"text": "trivial", "token_ids": [1]},
            {"text": "sorry", "token_ids": [2]},
            {"text": "simp", "token_ids": [3]},
        ]
        rewards = compute_efficiency_rewards(
            oracle, thm, comps, [True, False, True], [1.0, 0.0, 1.0],
            step=10, pass_rate=0.3
        )
        assert len(rewards) == 3
        assert rewards[1] == 0.0


class TestControllerCreation:
    def test_create_default(self):
        ctrl = create_controller()
        assert ctrl.config.verbose is True

    def test_create_with_output_dir(self, tmp_path):
        ctrl = create_controller(output_dir=str(tmp_path))
        assert ctrl.config.log_dir is not None

    def test_controller_step_returns_list(self):
        ctrl = create_controller()
        interventions = controller_step(ctrl, step=1, reward=0.5, pass_rate=0.3)
        assert isinstance(interventions, list)

    def test_controller_handles_none_params(self):
        ctrl = create_controller()
        interventions = controller_step(
            ctrl, step=1, reward=0.1,
            pass_rate=None, loss=None,
            proof_length=None, retention_rate=None,
            tactic_diversity=None,
        )
        assert isinstance(interventions, list)


class TestRetentionTracker:
    def test_first_rotation_zero(self):
        rt = RetentionTracker()
        rt.record("t1", True)
        rate = rt.compute_and_rotate()
        assert rate == 0.0, "First rotation has no previous solved set"

    def test_full_retention(self):
        rt = RetentionTracker()
        rt.record("t1", True)
        rt.record("t2", True)
        rt.compute_and_rotate()
        rt.record("t1", True)
        rt.record("t2", True)
        rate = rt.compute_and_rotate()
        assert rate == 1.0

    def test_partial_retention(self):
        rt = RetentionTracker()
        rt.record("t1", True)
        rt.record("t2", True)
        rt.compute_and_rotate()
        rt.record("t1", True)
        # t2 not solved this round
        rate = rt.compute_and_rotate()
        assert rate == 0.5

    def test_zero_retention(self):
        rt = RetentionTracker()
        rt.record("t1", True)
        rt.record("t2", True)
        rt.compute_and_rotate()
        # Neither solved this round
        rt.record("t3", True)
        rate = rt.compute_and_rotate()
        assert rate == 0.0

    def test_unsolved_not_tracked(self):
        rt = RetentionTracker()
        rt.record("t1", True)
        rt.record("t2", False)  # not solved
        rt.compute_and_rotate()
        rt.record("t1", True)
        rate = rt.compute_and_rotate()
        assert rate == 1.0, "Only solved theorems should be tracked"


class TestTacticExtraction:
    def test_simp(self):
        assert extract_first_tactic("simp [add_comm]") == "simp"

    def test_omega(self):
        assert extract_first_tactic("  omega") == "omega"

    def test_unknown(self):
        assert extract_first_tactic("foobar") == "unknown"

    def test_count_distinct(self):
        comps = [
            {"text": "simp"},
            {"text": "omega"},
            {"text": "simp"},
        ]
        verified = [True, True, False]
        assert count_distinct_tactics(comps, verified) == 2

    def test_count_zero_when_no_verified(self):
        comps = [{"text": "simp"}, {"text": "omega"}]
        verified = [False, False]
        assert count_distinct_tactics(comps, verified) == 0


class TestInterventionChecks:
    def test_no_efficiency_switch_when_absent(self):
        ctrl = create_controller()
        interventions = controller_step(ctrl, step=1, reward=0.1)
        assert not should_switch_to_efficiency(interventions)

    def test_no_expand_when_not_saturated(self):
        ctrl = create_controller()
        interventions = controller_step(ctrl, step=1, reward=0.1)
        assert not should_expand_dataset(interventions)

    def test_no_tactic_warning_initially(self):
        ctrl = create_controller()
        interventions = controller_step(ctrl, step=1, reward=0.1)
        assert tactic_diversity_warning(interventions) is None
