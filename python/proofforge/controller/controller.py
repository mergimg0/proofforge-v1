"""Adaptive Training Controller (App 8) — the main orchestrator.

Monitors training metrics in real time, detects phase transitions, and
recommends (or auto-applies) interventions based on the three-phase model.

Architecture:
  TrainingLoop → controller.update(step, reward, ...) → interventions
                                                         ↓
                                     auto-apply OR surface to user

Integration: the controller is called once per training step with the
latest metrics. It returns a list of interventions (possibly empty).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from proofforge.controller.metrics import ControllerMetrics
from proofforge.controller.phase_detector import PhaseDetector, TrainingPhase
from proofforge.controller.interventions import (
    Intervention,
    InterventionRules,
    InterventionType,
)


@dataclass
class ControllerConfig:
    """Configuration for the adaptive training controller."""

    # Phase detection thresholds
    accumulation_reward_threshold: float = 0.10
    breakout_reward_threshold: float = 0.25
    saturation_reward_threshold: float = 0.80
    saturation_std_threshold: float = 0.10
    stagnation_reward_ceiling: float = 0.15
    stagnation_patience: int = 50

    # Metric windows
    reward_window_size: int = 20
    pass_rate_window_size: int = 20

    # Intervention policy
    auto_apply_checkpoints: bool = True
    auto_apply_efficiency: bool = False
    expanding_ring_enabled: bool = True
    efficiency_reward_enabled: bool = True
    max_interventions_per_step: int = 3

    # Logging
    log_dir: Optional[str] = None
    log_every_n_steps: int = 5
    verbose: bool = True


@dataclass
class AdaptiveController:
    """The adaptive training controller daemon (App 8).

    Central orchestrator that combines:
    - ControllerMetrics: rolling-window statistics
    - PhaseDetector: three-phase transition detection
    - InterventionRules: phase-specific intervention logic

    Usage:
        controller = AdaptiveController(config)
        for step in training_loop:
            reward = compute_reward(...)
            interventions = controller.update(step, reward, pass_rate=pr)
            for i in interventions:
                if i.auto_apply:
                    apply_intervention(i)
                else:
                    log_recommendation(i)
    """

    config: ControllerConfig = field(default_factory=ControllerConfig)

    metrics: ControllerMetrics = field(init=False)
    phase_detector: PhaseDetector = field(init=False)
    rules: InterventionRules = field(init=False)

    _log_file: Optional[Path] = field(default=None, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)
    _total_interventions: int = field(default=0, init=False)

    # Tactic monoculture state (for /reward diversity bonus wiring)
    tactic_monoculture_active: bool = field(default=False, init=False)
    current_tactic_usage: dict[str, float] = field(default_factory=dict, init=False)

    # Callback hooks for auto-apply
    on_checkpoint: Optional[Callable[[int], None]] = None
    on_switch_efficiency: Optional[Callable[[float], None]] = None
    on_expand_dataset: Optional[Callable[[], None]] = None

    def __post_init__(self):
        self.metrics = ControllerMetrics(
            reward=__import__("proofforge.controller.metrics", fromlist=["MetricWindow"]).MetricWindow(
                max_size=self.config.reward_window_size
            ),
            pass_rate=__import__("proofforge.controller.metrics", fromlist=["MetricWindow"]).MetricWindow(
                max_size=self.config.pass_rate_window_size
            ),
        )

        self.phase_detector = PhaseDetector(
            accumulation_reward_threshold=self.config.accumulation_reward_threshold,
            breakout_reward_threshold=self.config.breakout_reward_threshold,
            saturation_reward_threshold=self.config.saturation_reward_threshold,
            saturation_std_threshold=self.config.saturation_std_threshold,
            stagnation_reward_ceiling=self.config.stagnation_reward_ceiling,
            stagnation_patience=self.config.stagnation_patience,
        )

        self.rules = InterventionRules(
            expanding_ring_enabled=self.config.expanding_ring_enabled,
            efficiency_reward_enabled=self.config.efficiency_reward_enabled,
            max_interventions_per_step=self.config.max_interventions_per_step,
        )

        if self.config.log_dir:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path / "controller.jsonl"

    def update(
        self,
        step: int,
        reward: float,
        pass_rate: Optional[float] = None,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        proof_length: Optional[float] = None,
        retention_rate: Optional[float] = None,
        tactic_diversity: Optional[float] = None,
    ) -> list[Intervention]:
        """Process one training step. Returns recommended interventions.

        This is the main entry point, called once per GRPO step.
        """
        # 1. Update metrics
        self.metrics.update(
            step=step,
            reward=reward,
            pass_rate=pass_rate,
            loss=loss,
            gradient_norm=gradient_norm,
            proof_length=proof_length,
            retention_rate=retention_rate,
            tactic_diversity=tactic_diversity,
        )

        # 2. Detect phase
        phase = self.phase_detector.detect(self.metrics)

        # 3. Evaluate intervention rules
        interventions = self.rules.evaluate(phase, self.metrics)

        # 4. Auto-apply where configured
        for intervention in interventions:
            if intervention.auto_apply:
                self._auto_apply(intervention)

        # Track tactic monoculture state for /reward diversity bonus
        self.tactic_monoculture_active = any(
            i.type == InterventionType.DIVERSIFY_TACTICS for i in interventions
        )
        if self.tactic_monoculture_active:
            params = next(
                (i.parameters for i in interventions
                 if i.type == InterventionType.DIVERSIFY_TACTICS),
                {},
            )
            # Compute tactic usage from tactic_diversity metric window
            # The intervention's parameters carry tactic_diversity as a float;
            # full per-tactic counts must come from the caller's batch data.
            self.current_tactic_usage = params.get("tactic_usage", {})

        self._total_interventions += len(interventions)

        # 5. Periodic logging
        if step % self.config.log_every_n_steps == 0:
            self._log_state(step, phase, interventions)

        return interventions

    def _auto_apply(self, intervention: Intervention) -> None:
        """Auto-apply an intervention via registered callbacks."""
        if intervention.type == InterventionType.CHECKPOINT:
            if self.config.auto_apply_checkpoints and self.on_checkpoint:
                self.on_checkpoint(intervention.step)

        elif intervention.type == InterventionType.SWITCH_EFFICIENCY:
            if self.config.auto_apply_efficiency and self.on_switch_efficiency:
                alpha = intervention.parameters.get("recommended_alpha", 0.3)
                self.on_switch_efficiency(alpha)

        elif intervention.type == InterventionType.EXPAND_DATASET:
            if self.on_expand_dataset:
                self.on_expand_dataset()

    def _log_state(
        self, step: int, phase: TrainingPhase, interventions: list[Intervention]
    ) -> None:
        """Log controller state for offline analysis."""
        entry = {
            "step": step,
            "phase": phase.value,
            "metrics": self.metrics.snapshot(),
            "interventions": [
                {"type": i.type.value, "reason": i.reason, "priority": i.priority}
                for i in interventions
            ],
            "elapsed_s": round(time.monotonic() - self._start_time, 1),
        }

        if self.config.verbose:
            phase_str = phase.value.upper()
            reward_str = f"reward={self.metrics.reward.mean:.3f}"
            pr_str = f"pr={self.metrics.pass_rate.mean:.3f}" if self.metrics.pass_rate.count > 0 else "pr=N/A"
            intervention_str = ", ".join(i.type.value for i in interventions) if interventions else "none"
            print(f"  [Controller] Step {step} | {phase_str} | {reward_str} | {pr_str} | interventions: {intervention_str}")

        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    @property
    def current_phase(self) -> TrainingPhase:
        return self.phase_detector.current_phase

    @property
    def phase_transitions(self) -> list:
        return self.phase_detector.transitions

    def summary(self) -> dict:
        """Full controller summary for end-of-training reporting."""
        return {
            "final_phase": self.current_phase.value,
            "total_steps": self.metrics.step_count,
            "total_interventions": self._total_interventions,
            "phase_transitions": [
                {
                    "from": t.from_phase.value,
                    "to": t.to_phase.value,
                    "step": t.step,
                    "trigger": t.trigger,
                }
                for t in self.phase_detector.transitions
            ],
            "final_metrics": self.metrics.snapshot(),
            "elapsed_s": round(time.monotonic() - self._start_time, 1),
        }

    @classmethod
    def replay_from_log(cls, log_path: str) -> "AdaptiveController":
        """Replay a training log through the controller for validation.

        Reads a JSONL file with {step, reward, pass_rate, ...} entries
        and replays them through a fresh controller. Use this to validate
        phase detection against manually-observed transitions.
        """
        controller = cls()
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                controller.update(
                    step=entry.get("step", 0),
                    reward=entry.get("reward", 0.0),
                    pass_rate=entry.get("pass_rate"),
                    loss=entry.get("loss"),
                    gradient_norm=entry.get("gradient_norm"),
                )
        return controller
