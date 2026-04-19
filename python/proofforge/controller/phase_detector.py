"""Three-phase training dynamics detector.

Detects the phase of GRPO training from metric streams:

  DISRUPTION → ACCUMULATION → BREAKOUT → SATURATION
                                            ↓
                                        STAGNATION (if stuck)

Phase transitions are detected from the running reward average (leading
indicator) and pass rate (lagging indicator). The key insight: reward
average leads pass rate by ~10-20 steps.

From the ProofForge training run:
  - Steps 0-10:  Disruption (pass rate drops as GRPO reshapes distribution)
  - Steps 10-35: Accumulation (reward climbing, pass rate flat at ~8%)
  - Steps 35-50: Breakout predicted by reward trend
  - Steps 50+:   Breakout confirmed (pass rate 8% → 23% → 43% → 61%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from proofforge.controller.metrics import ControllerMetrics


class TrainingPhase(Enum):
    """The six phases of GRPO training with binary verification."""

    DISRUPTION = "disruption"
    """Initial phase: GRPO reshapes the model's distribution.
    Pass rate may DROP. This is expected, not failure."""

    ACCUMULATION = "accumulation"
    """Infrastructure building: reward climbing but pass rate flat.
    The model is learning tactics/patterns that haven't yet combined
    into successful proofs. Patience required."""

    BREAKOUT = "breakout"
    """Infrastructure reaches critical mass: multiple theorem categories
    crack simultaneously. Pass rate rises rapidly. OPTIMAL ZONE."""

    CONSOLIDATION = "consolidation"
    """Previously-solved theorems stop flickering and become reliably solved.
    Retention rate jumps from ~9% to ~95%. The model is CEMENTING learned
    infrastructure. This is the paper's sharpest finding: the transition
    from fragile to robust proofs."""

    SATURATION = "saturation"
    """Most theorems solved reliably. Gradient signal collapses because
    GRPO groups have mostly-successful completions. Need efficiency
    reward (App 6) or dataset expansion."""

    STAGNATION = "stagnation"
    """Stuck: reward flat below useful threshold for many steps.
    Requires intervention (learning rate, group size, reward shaping)."""


@dataclass
class PhaseTransition:
    """Record of a detected phase transition."""
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    step: int
    trigger: str
    metrics_snapshot: dict


@dataclass
class PhaseDetector:
    """Detects training phase from metric streams.

    Uses the three-phase model from the ProofForge paper with
    empirically-validated thresholds from the 2% → 61% training run.

    Thresholds are configurable because different model/dataset
    combinations may have different breakout points.
    """

    accumulation_reward_threshold: float = 0.10
    """Reward mean above this + positive trend = disruption → accumulation."""

    breakout_reward_threshold: float = 0.25
    """Reward mean above this = accumulation → breakout."""

    consolidation_retention_threshold: float = 0.70
    """Retention rate above this during breakout = breakout → consolidation.
    Based on the 9.1% → 95.2% retention jump observed at steps 30-75."""

    saturation_reward_threshold: float = 0.80
    """Reward mean above this consistently = consolidation → saturation."""

    saturation_std_threshold: float = 0.10
    """Reward std below this (with high mean) confirms saturation."""

    stagnation_reward_ceiling: float = 0.15
    """Reward mean below this for stagnation_patience steps = stagnation."""

    stagnation_patience: int = 50
    """Steps of low reward before declaring stagnation."""

    min_window_size: int = 5
    """Minimum observations before attempting phase detection."""

    min_phase_dwell: int = 10
    """Minimum steps in a phase before allowing regression (hysteresis).
    Prevents oscillation from noisy rewards near phase boundaries."""

    reward_bias_correction: float = 0.0
    """Additive correction for reward mean when efficiency weighting and/or
    biased sampling depress observed rewards relative to true capability.
    Set to ~0.15 when both App 6 (efficiency) and App 2 (frontier sampling) are active.
    Set to ~0.08 when only App 6 is active."""

    current_phase: TrainingPhase = TrainingPhase.DISRUPTION
    phase_start_step: int = 0
    transitions: list[PhaseTransition] = field(default_factory=list)
    _stagnation_counter: int = 0

    def detect(self, metrics: ControllerMetrics) -> TrainingPhase:
        """Detect current phase from accumulated metrics.

        Returns the detected phase. If a transition occurred, it's
        recorded in self.transitions.
        """
        if metrics.reward.count < self.min_window_size:
            return self.current_phase

        new_phase = self._classify(metrics)

        # Hysteresis: prevent regression within min_phase_dwell steps
        if new_phase != self.current_phase:
            steps_in_phase = metrics.step_count - self.phase_start_step
            is_regression = self._is_regression(self.current_phase, new_phase)
            if is_regression and steps_in_phase < self.min_phase_dwell:
                return self.current_phase

        if new_phase != self.current_phase:
            transition = PhaseTransition(
                from_phase=self.current_phase,
                to_phase=new_phase,
                step=metrics.step_count,
                trigger=self._transition_reason(self.current_phase, new_phase, metrics),
                metrics_snapshot=metrics.snapshot(),
            )
            self.transitions.append(transition)
            self.phase_start_step = metrics.step_count
            self.current_phase = new_phase

        return self.current_phase

    @staticmethod
    def _is_regression(current: TrainingPhase, proposed: TrainingPhase) -> bool:
        """Is this a backward phase transition?

        Forward: disruption → accumulation → breakout → consolidation → saturation
        Backward: anything moving left in the sequence above.
        """
        order = {
            TrainingPhase.DISRUPTION: 0,
            TrainingPhase.STAGNATION: 0,
            TrainingPhase.ACCUMULATION: 1,
            TrainingPhase.BREAKOUT: 2,
            TrainingPhase.CONSOLIDATION: 3,
            TrainingPhase.SATURATION: 4,
        }
        return order.get(proposed, 0) < order.get(current, 0)

    def _classify(self, metrics: ControllerMetrics) -> TrainingPhase:
        """Core classification logic."""
        reward_mean = metrics.reward.mean + self.reward_bias_correction
        reward_trend = metrics.reward.trend
        reward_std = metrics.reward.std

        # Check stagnation first (can happen from any phase)
        if reward_mean < self.stagnation_reward_ceiling and abs(reward_trend) < 0.001:
            self._stagnation_counter += 1
            if self._stagnation_counter >= self.stagnation_patience:
                return TrainingPhase.STAGNATION
        else:
            self._stagnation_counter = 0

        # Phase-specific transitions
        if self.current_phase == TrainingPhase.DISRUPTION:
            if reward_mean > self.accumulation_reward_threshold and reward_trend > 0:
                return TrainingPhase.ACCUMULATION
            return TrainingPhase.DISRUPTION

        elif self.current_phase == TrainingPhase.ACCUMULATION:
            if reward_mean > self.breakout_reward_threshold:
                return TrainingPhase.BREAKOUT
            # Can regress to disruption if reward drops
            if reward_mean < 0.05 and reward_trend < -0.005:
                return TrainingPhase.DISRUPTION
            return TrainingPhase.ACCUMULATION

        elif self.current_phase == TrainingPhase.BREAKOUT:
            # Check consolidation: retention rate climbing = theorems stabilizing
            if metrics.consolidation_detected():
                return TrainingPhase.CONSOLIDATION
            # Saturation: reward AND pass rate both high, or pass rate plateauing high
            reward_saturated = reward_mean > self.saturation_reward_threshold and reward_std < self.saturation_std_threshold
            passrate_saturated = (
                metrics.pass_rate.count >= 10
                and metrics.pass_rate.mean > 0.6
                and abs(metrics.pass_rate.trend) < 0.002
            )
            if reward_saturated or passrate_saturated:
                return TrainingPhase.SATURATION
            # Can regress to accumulation if breakout stalls SIGNIFICANTLY
            # (reward must drop well below breakout threshold, not just noise)
            if reward_mean < self.breakout_reward_threshold * 0.8 and reward_trend < -0.01:
                return TrainingPhase.ACCUMULATION
            return TrainingPhase.BREAKOUT

        elif self.current_phase == TrainingPhase.CONSOLIDATION:
            # Consolidation → saturation when gradient signal collapses
            reward_saturated = reward_mean > self.saturation_reward_threshold and reward_std < self.saturation_std_threshold
            passrate_saturated = (
                metrics.pass_rate.count >= 10
                and metrics.pass_rate.mean > 0.6
                and abs(metrics.pass_rate.trend) < 0.002
            )
            if reward_saturated or passrate_saturated:
                return TrainingPhase.SATURATION
            # Can regress to breakout if retention drops
            if metrics.retention_rate.count >= 5 and metrics.retention_rate.mean < 0.5:
                return TrainingPhase.BREAKOUT
            return TrainingPhase.CONSOLIDATION

        elif self.current_phase == TrainingPhase.SATURATION:
            # Saturation is absorbing unless reward drops significantly
            if reward_mean < self.breakout_reward_threshold:
                return TrainingPhase.BREAKOUT
            return TrainingPhase.SATURATION

        elif self.current_phase == TrainingPhase.STAGNATION:
            # Can escape stagnation if reward starts climbing
            if reward_trend > 0.005:
                return TrainingPhase.ACCUMULATION
            return TrainingPhase.STAGNATION

        return self.current_phase

    def _transition_reason(
        self, from_phase: TrainingPhase, to_phase: TrainingPhase, metrics: ControllerMetrics
    ) -> str:
        """Human-readable reason for a phase transition."""
        snap = metrics.snapshot()
        retention_str = f", retention={snap.get('retention_mean', 'N/A')}" if snap.get('retention_mean') is not None else ""
        reasons = {
            (TrainingPhase.DISRUPTION, TrainingPhase.ACCUMULATION):
                f"Reward mean {snap['reward_mean']:.3f} > {self.accumulation_reward_threshold} with positive trend {snap['reward_trend']:.4f}",
            (TrainingPhase.ACCUMULATION, TrainingPhase.BREAKOUT):
                f"Reward mean {snap['reward_mean']:.3f} crossed breakout threshold {self.breakout_reward_threshold}",
            (TrainingPhase.BREAKOUT, TrainingPhase.CONSOLIDATION):
                f"Retention rate climbing above {self.consolidation_retention_threshold}{retention_str} — theorems stabilizing from fragile to robust",
            (TrainingPhase.CONSOLIDATION, TrainingPhase.SATURATION):
                f"Reward mean {snap['reward_mean']:.3f} > {self.saturation_reward_threshold} with low std — gradient saturation",
            (TrainingPhase.BREAKOUT, TrainingPhase.SATURATION):
                f"Reward mean {snap['reward_mean']:.3f} > {self.saturation_reward_threshold} with low std — gradient saturation (no retention data)",
            (TrainingPhase.DISRUPTION, TrainingPhase.STAGNATION):
                f"Reward flat at {snap['reward_mean']:.3f} for {self.stagnation_patience}+ steps",
        }
        key = (from_phase, to_phase)
        return reasons.get(key, f"Transition {from_phase.value} → {to_phase.value} at step {snap['step']}")

    def estimated_breakout_eta(self, metrics: ControllerMetrics) -> Optional[int]:
        """Estimate steps until breakout based on reward trend.

        Only meaningful during ACCUMULATION phase.
        Returns None if not in accumulation or trend is non-positive.
        """
        if self.current_phase != TrainingPhase.ACCUMULATION:
            return None

        reward_trend = metrics.reward.trend
        if reward_trend <= 0:
            return None

        remaining = self.breakout_reward_threshold - metrics.reward.mean
        if remaining <= 0:
            return 0

        return int(remaining / reward_trend)
