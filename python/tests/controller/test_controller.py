"""Tests for proofforge.controller.controller.

Key implementation details:
- AdaptiveController.__post_init__ builds metrics/phase_detector/rules from config.
- update() returns list[Intervention] and auto-calls on_checkpoint during BREAKOUT
  only when config.auto_apply_checkpoints=True AND the callback is registered.
- current_phase and phase_transitions are properties delegating to phase_detector.
- replay_from_log() reads JSONL {step, reward, pass_rate?, loss?, gradient_norm?}.
- The stagnation_counter increments once per detect() call; the controller calls
  detect() once per update() call, so stagnation needs patience update() calls.
- Reaching SATURATION in lifecycle tests requires enough same-value pushes so the
  rolling window mean exceeds the threshold.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proofforge.controller.controller import AdaptiveController, ControllerConfig
from proofforge.controller.interventions import Intervention, InterventionType
from proofforge.controller.phase_detector import TrainingPhase


# ---------------------------------------------------------------------------
# ControllerConfig defaults
# ---------------------------------------------------------------------------


class TestControllerConfigDefaults:
    def test_default_accumulation_threshold(self):
        c = ControllerConfig()
        assert c.accumulation_reward_threshold == pytest.approx(0.10)

    def test_default_breakout_threshold(self):
        c = ControllerConfig()
        assert c.breakout_reward_threshold == pytest.approx(0.25)

    def test_default_saturation_threshold(self):
        c = ControllerConfig()
        assert c.saturation_reward_threshold == pytest.approx(0.80)

    def test_default_saturation_std_threshold(self):
        c = ControllerConfig()
        assert c.saturation_std_threshold == pytest.approx(0.10)

    def test_default_stagnation_patience(self):
        c = ControllerConfig()
        assert c.stagnation_patience == 50

    def test_default_reward_window_size(self):
        c = ControllerConfig()
        assert c.reward_window_size == 20

    def test_default_auto_apply_checkpoints(self):
        c = ControllerConfig()
        assert c.auto_apply_checkpoints is True

    def test_default_auto_apply_efficiency(self):
        c = ControllerConfig()
        assert c.auto_apply_efficiency is False

    def test_default_max_interventions_per_step(self):
        c = ControllerConfig()
        assert c.max_interventions_per_step == 3

    def test_default_log_dir_is_none(self):
        c = ControllerConfig()
        assert c.log_dir is None

    def test_default_verbose(self):
        c = ControllerConfig()
        assert c.verbose is True


# ---------------------------------------------------------------------------
# AdaptiveController construction
# ---------------------------------------------------------------------------


class TestAdaptiveControllerConstruction:
    def test_starts_in_disruption(self):
        ctrl = AdaptiveController()
        assert ctrl.current_phase == TrainingPhase.DISRUPTION

    def test_no_transitions_at_start(self):
        ctrl = AdaptiveController()
        assert ctrl.phase_transitions == []

    def test_metrics_initialized(self):
        ctrl = AdaptiveController()
        assert ctrl.metrics is not None

    def test_phase_detector_initialized(self):
        ctrl = AdaptiveController()
        assert ctrl.phase_detector is not None

    def test_rules_initialized(self):
        ctrl = AdaptiveController()
        assert ctrl.rules is not None

    def test_config_thresholds_passed_to_phase_detector(self):
        cfg = ControllerConfig(breakout_reward_threshold=0.40)
        ctrl = AdaptiveController(config=cfg)
        assert ctrl.phase_detector.breakout_reward_threshold == pytest.approx(0.40)

    def test_config_max_interventions_passed_to_rules(self):
        cfg = ControllerConfig(max_interventions_per_step=1)
        ctrl = AdaptiveController(config=cfg)
        assert ctrl.rules.max_interventions_per_step == 1


# ---------------------------------------------------------------------------
# update() basics
# ---------------------------------------------------------------------------


class TestAdaptiveControllerUpdate:
    def test_update_returns_list(self):
        ctrl = AdaptiveController()
        result = ctrl.update(step=1, reward=0.1)
        assert isinstance(result, list)

    def test_update_list_items_are_interventions(self):
        ctrl = AdaptiveController()
        result = ctrl.update(step=1, reward=0.1)
        for item in result:
            assert isinstance(item, Intervention)

    def test_update_records_reward_in_metrics(self):
        ctrl = AdaptiveController()
        ctrl.update(step=5, reward=0.3)
        assert ctrl.metrics.reward.last == pytest.approx(0.3)

    def test_update_forwards_pass_rate_to_metrics(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, pass_rate=0.15)
        assert ctrl.metrics.pass_rate.count == 1
        assert ctrl.metrics.pass_rate.last == pytest.approx(0.15)

    def test_update_skips_none_pass_rate(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, pass_rate=None)
        assert ctrl.metrics.pass_rate.count == 0

    def test_update_forwards_retention_rate(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, retention_rate=0.75)
        assert ctrl.metrics.retention_rate.last == pytest.approx(0.75)

    def test_update_forwards_tactic_diversity(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, tactic_diversity=4.0)
        assert ctrl.metrics.tactic_diversity.last == pytest.approx(4.0)

    def test_update_forwards_loss(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, loss=1.5)
        assert ctrl.metrics.loss.last == pytest.approx(1.5)

    def test_update_forwards_gradient_norm(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, gradient_norm=2.1)
        assert ctrl.metrics.gradient_norm.last == pytest.approx(2.1)

    def test_update_forwards_proof_length(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.2, proof_length=128.0)
        assert ctrl.metrics.proof_length.last == pytest.approx(128.0)


# ---------------------------------------------------------------------------
# current_phase updates via update()
# ---------------------------------------------------------------------------


class TestCurrentPhaseUpdates:
    def test_stays_disruption_with_low_reward(self):
        ctrl = AdaptiveController()
        for i in range(10):
            ctrl.update(step=i, reward=0.05)
        assert ctrl.current_phase == TrainingPhase.DISRUPTION

    def test_transitions_to_accumulation(self):
        ctrl = AdaptiveController()
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        assert ctrl.current_phase == TrainingPhase.ACCUMULATION

    def test_transitions_to_breakout(self):
        ctrl = AdaptiveController()
        # disruption → accumulation
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        # accumulation → breakout (need mean > 0.25)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        assert ctrl.current_phase == TrainingPhase.BREAKOUT

    def test_transitions_to_saturation(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        for i in range(20):
            ctrl.update(step=30 + i, reward=0.85)
        assert ctrl.current_phase == TrainingPhase.SATURATION


# ---------------------------------------------------------------------------
# phase_transitions recording
# ---------------------------------------------------------------------------


class TestPhaseTransitionsRecording:
    def test_phase_transitions_is_list(self):
        ctrl = AdaptiveController()
        assert isinstance(ctrl.phase_transitions, list)

    def test_phase_transitions_records_disruption_to_accumulation(self):
        ctrl = AdaptiveController()
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        transitions = ctrl.phase_transitions
        assert len(transitions) >= 1
        assert transitions[0].from_phase == TrainingPhase.DISRUPTION
        assert transitions[0].to_phase == TrainingPhase.ACCUMULATION

    def test_phase_transitions_records_multiple_transitions(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        assert len(ctrl.phase_transitions) >= 2


# ---------------------------------------------------------------------------
# Full lifecycle: disruption → accumulation → breakout → saturation
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_full_lifecycle_phases_visited(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        phases_seen = set()

        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
            phases_seen.add(ctrl.current_phase)

        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
            phases_seen.add(ctrl.current_phase)

        for i in range(20):
            ctrl.update(step=30 + i, reward=0.85)
            phases_seen.add(ctrl.current_phase)

        assert TrainingPhase.DISRUPTION in phases_seen
        assert TrainingPhase.ACCUMULATION in phases_seen
        assert TrainingPhase.BREAKOUT in phases_seen
        assert TrainingPhase.SATURATION in phases_seen

    def test_full_lifecycle_update_returns_interventions_each_step(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        all_interventions = []
        for i in range(50):
            reward = 0.05 + i * 0.015
            result = ctrl.update(step=i, reward=reward)
            all_interventions.extend(result)
        # Some steps should have produced interventions
        assert len(all_interventions) > 0

    def test_full_lifecycle_transitions_count(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        for i in range(20):
            ctrl.update(step=30 + i, reward=0.85)
        # At minimum: disruption→accumulation, accumulation→breakout, breakout→saturation
        assert len(ctrl.phase_transitions) >= 3


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_dict(self):
        ctrl = AdaptiveController()
        assert isinstance(ctrl.summary(), dict)

    def test_summary_contains_required_keys(self):
        ctrl = AdaptiveController()
        s = ctrl.summary()
        required = {
            "final_phase",
            "total_steps",
            "total_interventions",
            "phase_transitions",
            "final_metrics",
            "elapsed_s",
        }
        assert required.issubset(s.keys())

    def test_summary_final_phase_is_string(self):
        ctrl = AdaptiveController()
        assert isinstance(ctrl.summary()["final_phase"], str)

    def test_summary_phase_transitions_is_list(self):
        ctrl = AdaptiveController()
        assert isinstance(ctrl.summary()["phase_transitions"], list)

    def test_summary_final_metrics_is_dict(self):
        ctrl = AdaptiveController()
        assert isinstance(ctrl.summary()["final_metrics"], dict)

    def test_summary_total_interventions_increases(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.05)
        # DISRUPTION phase produces WAIT each step (but cooldown caps at 1 per 10 steps)
        assert ctrl.summary()["total_interventions"] >= 1

    def test_summary_reflects_current_phase(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        assert ctrl.summary()["final_phase"] == ctrl.current_phase.value

    def test_summary_phase_transitions_entries_have_correct_keys(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        s = ctrl.summary()
        if s["phase_transitions"]:
            t = s["phase_transitions"][0]
            assert "from" in t
            assert "to" in t
            assert "step" in t
            assert "trigger" in t


# ---------------------------------------------------------------------------
# on_checkpoint callback
# ---------------------------------------------------------------------------


class TestOnCheckpointCallback:
    def test_on_checkpoint_fires_during_breakout(self):
        fired_steps = []

        cfg = ControllerConfig(auto_apply_checkpoints=True, verbose=False)
        ctrl = AdaptiveController(config=cfg, on_checkpoint=lambda s: fired_steps.append(s))

        # Drive to BREAKOUT
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)

        assert ctrl.current_phase == TrainingPhase.BREAKOUT
        # Continue in breakout so checkpoint interventions fire and auto-apply
        for i in range(15):
            ctrl.update(step=30 + i, reward=0.30)

        assert len(fired_steps) > 0

    def test_on_checkpoint_not_fired_when_auto_apply_disabled(self):
        fired_steps = []

        cfg = ControllerConfig(auto_apply_checkpoints=False, verbose=False)
        ctrl = AdaptiveController(config=cfg, on_checkpoint=lambda s: fired_steps.append(s))

        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(30):
            ctrl.update(step=10 + i, reward=0.30)

        assert len(fired_steps) == 0

    def test_on_checkpoint_not_fired_when_no_callback_registered(self):
        # Should not raise even with auto_apply_checkpoints=True and no callback
        cfg = ControllerConfig(auto_apply_checkpoints=True, verbose=False)
        ctrl = AdaptiveController(config=cfg)
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        # No assertion needed — just must not raise


# ---------------------------------------------------------------------------
# replay_from_log()
# ---------------------------------------------------------------------------


class TestReplayFromLog:
    def _write_log(self, entries: list[dict], path: Path) -> None:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_replay_returns_controller(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log([{"step": 1, "reward": 0.1}], log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert isinstance(ctrl, AdaptiveController)

    def test_replay_processes_all_entries(self):
        entries = [{"step": i, "reward": 0.12 + i * 0.01} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.metrics.reward.count > 0

    def test_replay_detects_accumulation_phase(self):
        entries = [{"step": i, "reward": 0.12 + i * 0.01} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.current_phase == TrainingPhase.ACCUMULATION

    def test_replay_handles_pass_rate_field(self):
        entries = [
            {"step": i, "reward": 0.15, "pass_rate": 0.08}
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.metrics.pass_rate.count > 0

    def test_replay_handles_loss_field(self):
        entries = [{"step": i, "reward": 0.15, "loss": 2.0 - i * 0.1} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.metrics.loss.count > 0

    def test_replay_handles_gradient_norm_field(self):
        entries = [{"step": i, "reward": 0.15, "gradient_norm": 1.5} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.metrics.gradient_norm.count > 0

    def test_replay_records_transitions(self):
        # Enough data to trigger at least disruption→accumulation
        entries = [{"step": i, "reward": 0.12 + i * 0.01} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert len(ctrl.phase_transitions) >= 1

    def test_replay_empty_log_stays_disruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log([], log)
            ctrl = AdaptiveController.replay_from_log(str(log))
            assert ctrl.current_phase == TrainingPhase.DISRUPTION

    def test_replay_full_lifecycle(self):
        entries = (
            [{"step": i, "reward": 0.12 + i * 0.01} for i in range(10)]
            + [{"step": 10 + i, "reward": 0.30} for i in range(20)]
            + [{"step": 30 + i, "reward": 0.85} for i in range(20)]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "run.jsonl"
            self._write_log(entries, log)
            ctrl = AdaptiveController.replay_from_log(str(log))
        assert ctrl.current_phase == TrainingPhase.SATURATION
        phases = {t.to_phase for t in ctrl.phase_transitions}
        assert TrainingPhase.ACCUMULATION in phases
        assert TrainingPhase.BREAKOUT in phases
        assert TrainingPhase.SATURATION in phases


# ---------------------------------------------------------------------------
# retention_rate and tactic_diversity forwarded to metrics
# ---------------------------------------------------------------------------


class TestNewParamsForwarding:
    def test_retention_rate_forwarded(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.3, retention_rate=0.85)
        assert ctrl.metrics.retention_rate.count == 1
        assert ctrl.metrics.retention_rate.last == pytest.approx(0.85)

    def test_retention_rate_none_not_recorded(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.3, retention_rate=None)
        assert ctrl.metrics.retention_rate.count == 0

    def test_tactic_diversity_forwarded(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.3, tactic_diversity=3.5)
        assert ctrl.metrics.tactic_diversity.count == 1
        assert ctrl.metrics.tactic_diversity.last == pytest.approx(3.5)

    def test_tactic_diversity_none_not_recorded(self):
        ctrl = AdaptiveController()
        ctrl.update(step=1, reward=0.3, tactic_diversity=None)
        assert ctrl.metrics.tactic_diversity.count == 0

    def test_retention_rate_influences_consolidation_detection(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        # Reach breakout first
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30)
        assert ctrl.current_phase == TrainingPhase.BREAKOUT
        # Add retention data to trigger consolidation
        for i in range(10):
            ctrl.update(step=30 + i, reward=0.55, retention_rate=0.80)
        assert ctrl.current_phase == TrainingPhase.CONSOLIDATION

    def test_tactic_diversity_low_triggers_diversify_intervention(self):
        ctrl = AdaptiveController(config=ControllerConfig(verbose=False))
        # Get to breakout with low diversity
        for i in range(10):
            ctrl.update(step=i, reward=0.12 + i * 0.01, tactic_diversity=1.5)
        for i in range(20):
            ctrl.update(step=10 + i, reward=0.30, tactic_diversity=1.5)
        assert ctrl.current_phase == TrainingPhase.BREAKOUT
        # One more step with low diversity to trigger monoculture rule
        interventions = ctrl.update(step=31, reward=0.30, tactic_diversity=1.5)
        # Check that DIVERSIFY_TACTICS was issued at some point
        all_types = {i.type for i in ctrl.rules.intervention_history}
        assert InterventionType.DIVERSIFY_TACTICS in all_types
