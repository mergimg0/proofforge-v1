"""Intervention rules: what to do at each training phase.

Maps (phase, diagnostic) → recommended intervention. The controller
applies these rules automatically or surfaces them as recommendations.

Intervention types:
  WAIT       — Do nothing. Log the phase. (Disruption)
  LOG        — Record progress. Estimate breakout ETA. (Accumulation)
  NO_ACTION  — Optimal zone. Don't touch anything. (Breakout)
  EXPAND     — Add harder theorems / expand dataset. (Saturation)
  EFFICIENCY — Switch to efficiency reward (App 6). (Saturation)
  TUNE       — Adjust learning rate, group size, etc. (Stagnation)
  SHAPED     — Temporarily enable shaped reward warmup. (Stagnation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from proofforge.controller.phase_detector import TrainingPhase
from proofforge.controller.metrics import ControllerMetrics


class InterventionType(Enum):
    """Categories of training interventions."""
    WAIT = "wait"
    LOG = "log"
    NO_ACTION = "no_action"
    EXPAND_DATASET = "expand_dataset"
    SWITCH_EFFICIENCY = "switch_efficiency"
    INCREASE_GROUP_SIZE = "increase_group_size"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    ENABLE_SHAPED_WARMUP = "enable_shaped_warmup"
    INCREASE_TEMPERATURE = "increase_temperature"
    CHECKPOINT = "checkpoint"
    DIVERSIFY_TACTICS = "diversify_tactics"
    CONSOLIDATION_HOLD = "consolidation_hold"


@dataclass
class Intervention:
    """A recommended training intervention."""

    type: InterventionType
    reason: str
    phase: TrainingPhase
    step: int
    priority: int = 0
    """Higher = more urgent. 0 = informational, 1 = recommended, 2 = critical."""

    parameters: dict = field(default_factory=dict)
    """Intervention-specific parameters (e.g., new_lr, new_group_size)."""

    auto_apply: bool = False
    """Whether this intervention can be applied automatically."""

    def __str__(self) -> str:
        urgency = {0: "INFO", 1: "RECOMMEND", 2: "CRITICAL"}
        return f"[{urgency.get(self.priority, 'INFO')}] Step {self.step} ({self.phase.value}): {self.type.value} — {self.reason}"


class InterventionRules:
    """Rule engine mapping (phase, metrics) → interventions.

    Each rule is a method that checks conditions and returns an
    Intervention or None. The engine runs all rules and returns
    the highest-priority interventions.
    """

    def __init__(
        self,
        expanding_ring_enabled: bool = True,
        efficiency_reward_enabled: bool = True,
        max_interventions_per_step: int = 3,
    ):
        self.expanding_ring_enabled = expanding_ring_enabled
        self.efficiency_reward_enabled = efficiency_reward_enabled
        self.max_interventions_per_step = max_interventions_per_step
        self.intervention_history: list[Intervention] = []
        self._last_intervention_step: dict[InterventionType, int] = {}

    def evaluate(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> list[Intervention]:
        """Run all rules and return prioritized interventions."""
        candidates = []

        for rule in self._all_rules():
            result = rule(phase, metrics)
            if result is not None:
                # Cooldown: don't repeat same intervention type within 10 steps
                last_step = self._last_intervention_step.get(result.type, -100)
                if metrics.step_count - last_step >= 10:
                    candidates.append(result)

        candidates.sort(key=lambda i: -i.priority)
        selected = candidates[: self.max_interventions_per_step]

        for intervention in selected:
            self.intervention_history.append(intervention)
            self._last_intervention_step[intervention.type] = metrics.step_count

        return selected

    def _all_rules(self):
        return [
            self._rule_disruption_patience,
            self._rule_accumulation_logging,
            self._rule_breakout_checkpoint,
            self._rule_consolidation_hold,
            self._rule_tactic_monoculture,
            self._rule_saturation_efficiency,
            self._rule_saturation_expand,
            self._rule_stagnation_group_size,
            self._rule_stagnation_lr,
            self._rule_stagnation_shaped,
            self._rule_gradient_anomaly,
        ]

    def _rule_disruption_patience(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """During disruption: WAIT. Log reassurance."""
        if phase != TrainingPhase.DISRUPTION:
            return None
        return Intervention(
            type=InterventionType.WAIT,
            reason="Disruption phase — GRPO reshaping distribution. Pass rate drop is expected. Patience required.",
            phase=phase,
            step=metrics.step_count,
            priority=0,
        )

    def _rule_accumulation_logging(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """During accumulation: log progress, estimate breakout ETA."""
        if phase != TrainingPhase.ACCUMULATION:
            return None

        reward_trend = metrics.reward.trend
        leading = metrics.reward_leads_pass_rate()

        reason_parts = [f"Infrastructure building. Reward trend: {reward_trend:.4f}."]
        if leading:
            reason_parts.append("LEADING INDICATOR ACTIVE: reward climbing while pass rate flat — breakout approaching.")

        return Intervention(
            type=InterventionType.LOG,
            reason=" ".join(reason_parts),
            phase=phase,
            step=metrics.step_count,
            priority=0,
            parameters={"reward_trend": reward_trend, "leading_indicator": leading},
        )

    def _rule_breakout_checkpoint(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """During breakout: checkpoint aggressively."""
        if phase != TrainingPhase.BREAKOUT:
            return None
        return Intervention(
            type=InterventionType.CHECKPOINT,
            reason=f"OPTIMAL ZONE. Pass rate rising. Checkpoint at step {metrics.step_count}.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            auto_apply=True,
        )

    def _rule_saturation_efficiency(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """At saturation: switch to efficiency reward (App 6)."""
        if phase != TrainingPhase.SATURATION:
            return None
        if not self.efficiency_reward_enabled:
            return None
        if not metrics.gradient_saturation():
            return None

        return Intervention(
            type=InterventionType.SWITCH_EFFICIENCY,
            reason=f"Gradient saturation detected (reward mean {metrics.reward.mean:.3f}, std {metrics.reward.std:.3f}). Switch to efficiency reward to restore gradient signal.",
            phase=phase,
            step=metrics.step_count,
            priority=2,
            parameters={"recommended_alpha": 0.3},
            auto_apply=False,
        )

    def _rule_saturation_expand(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """At saturation: expand theorem set via expanding ring."""
        if phase != TrainingPhase.SATURATION:
            return None
        if not self.expanding_ring_enabled:
            return None

        return Intervention(
            type=InterventionType.EXPAND_DATASET,
            reason="Saturation detected. Activate expanding ring curriculum — add harder theorems from Level 1-2 to keep theorems in frontier zone.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            auto_apply=False,
        )

    def _rule_stagnation_group_size(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """At stagnation: try increasing group size for better advantage signal."""
        if phase != TrainingPhase.STAGNATION:
            return None

        return Intervention(
            type=InterventionType.INCREASE_GROUP_SIZE,
            reason=f"Stagnation detected (reward mean {metrics.reward.mean:.3f} for {metrics.step_count} steps). Increase group_size for richer advantage signal.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            parameters={"recommended_group_size": 32},
        )

    def _rule_stagnation_lr(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """At stagnation: reduce learning rate."""
        if phase != TrainingPhase.STAGNATION:
            return None

        return Intervention(
            type=InterventionType.REDUCE_LEARNING_RATE,
            reason="Stagnation persists. Reduce learning rate to allow finer gradient steps.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            parameters={"recommended_lr_factor": 0.5},
        )

    def _rule_stagnation_shaped(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """At stagnation: enable shaped reward warmup to bootstrap."""
        if phase != TrainingPhase.STAGNATION:
            return None

        return Intervention(
            type=InterventionType.ENABLE_SHAPED_WARMUP,
            reason="Stagnation persists. Enable shaped reward warmup (partial credit from failure analysis) to provide gradient signal from incorrect proofs.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            parameters={"warmup_steps": 30},
        )

    def _rule_consolidation_hold(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """During consolidation: hold steady and checkpoint.

        The model is cementing learned infrastructure — retention is jumping
        from ~9% to ~95%. Don't interfere. Checkpoint aggressively because
        this is the phase where fragile proofs become robust.
        """
        if phase != TrainingPhase.CONSOLIDATION:
            return None

        retention_str = ""
        if metrics.retention_rate.count > 0:
            retention_str = f" Retention: {metrics.retention_rate.mean:.1%}."

        return Intervention(
            type=InterventionType.CONSOLIDATION_HOLD,
            reason=f"CONSOLIDATION — theorems stabilizing from fragile to robust.{retention_str} Hold steady. Checkpoint aggressively.",
            phase=phase,
            step=metrics.step_count,
            priority=1,
            auto_apply=True,
        )

    def _rule_tactic_monoculture(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """Detect and flag tactic monoculture (simp domination).

        At C150, 67% of first tactics were simp. The model never learned
        cases/omega/specialized lemmas. Flag when tactic diversity is
        dangerously low so the training loop can add diversity pressure.
        """
        if not metrics.tactic_monoculture():
            return None

        return Intervention(
            type=InterventionType.DIVERSIFY_TACTICS,
            reason=f"TACTIC MONOCULTURE: diversity={metrics.tactic_diversity.mean:.1f} distinct tactics. Risk of simp domination — 7 unsolved theorems require cases/omega. Increase temperature or add tactic-forcing curriculum.",
            phase=phase,
            step=metrics.step_count,
            priority=2,
            parameters={
                "tactic_diversity": metrics.tactic_diversity.mean,
                "recommended_actions": [
                    "increase sampling temperature for exploration",
                    "add curriculum items requiring non-simp tactics",
                    "apply diversity bonus in reward",
                ],
            },
        )

    def _rule_gradient_anomaly(
        self, phase: TrainingPhase, metrics: ControllerMetrics
    ) -> Optional[Intervention]:
        """Detect gradient norm anomalies (SOS bounded-step proxy)."""
        if metrics.gradient_norm.count < 5:
            return None

        if metrics.gradient_norm.mean > 5.0:
            return Intervention(
                type=InterventionType.REDUCE_LEARNING_RATE,
                reason=f"Gradient norm anomaly: mean {metrics.gradient_norm.mean:.2f} exceeds safety threshold. Reduce learning rate to maintain bounded step.",
                phase=phase,
                step=metrics.step_count,
                priority=2,
                parameters={"recommended_lr_factor": 0.3},
            )
        return None
