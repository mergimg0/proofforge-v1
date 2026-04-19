"""Three-Phase Training Dynamics Analyzer (App 5).

Validates the three-phase model against observed training data.
Produces annotated phase labels for each training step and
diagnostic reports for the methodology paper.

From the ProofForge 2% → 61% training run:
  Phase 1 (disruption):    Steps 0-10,  pass rate drops
  Phase 2 (accumulation):  Steps 10-35, reward climbing, pass rate flat ~8%
  Phase 3 (breakout):      Steps 35+,   pass rate 8% → 23% → 43% → 61%

Key claim: The running reward average is the LEADING indicator.
If it's climbing while pass rate is flat, infrastructure is building
and breakout is coming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from proofforge.controller.phase_detector import TrainingPhase


@dataclass(frozen=True)
class PhaseAnnotation:
    """Phase label for a single training step."""
    step: int
    phase: TrainingPhase
    reward_mean: float
    pass_rate: float
    reward_trend: float
    leading_indicator: bool


@dataclass
class ThreePhaseAnalyzer:
    """Offline analyzer for three-phase training dynamics.

    Takes a complete training log and produces phase annotations,
    transition points, and diagnostic metrics for the methodology paper.
    """

    steps: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    pass_rates: list[float] = field(default_factory=list)

    # Analysis parameters
    reward_window: int = 10
    """Window size for rolling reward average."""

    disruption_threshold: float = 0.05
    """Pass rate below this = disruption."""

    accumulation_reward_slope: float = 0.003
    """Reward trend above this during flat pass rate = accumulation."""

    breakout_pass_rate_slope: float = 0.01
    """Pass rate trend above this = breakout."""

    def add_step(self, step: int, reward: float, pass_rate: float) -> None:
        self.steps.append(step)
        self.rewards.append(reward)
        self.pass_rates.append(pass_rate)

    def annotate(self) -> list[PhaseAnnotation]:
        """Produce phase annotations for all recorded steps."""
        if len(self.steps) < self.reward_window:
            return []

        annotations = []
        rewards_arr = np.array(self.rewards)
        pass_rates_arr = np.array(self.pass_rates)

        for i in range(self.reward_window, len(self.steps)):
            window = rewards_arr[max(0, i - self.reward_window):i]
            pr_window = pass_rates_arr[max(0, i - self.reward_window):i]

            reward_mean = float(np.mean(window))
            reward_trend = self._linear_trend(window)
            pr_mean = float(np.mean(pr_window))
            pr_trend = self._linear_trend(pr_window)

            leading = reward_trend > self.accumulation_reward_slope and abs(pr_trend) < self.breakout_pass_rate_slope

            phase = self._classify_phase(reward_mean, reward_trend, pr_mean, pr_trend)

            annotations.append(PhaseAnnotation(
                step=self.steps[i],
                phase=phase,
                reward_mean=round(reward_mean, 4),
                pass_rate=round(pr_mean, 4),
                reward_trend=round(reward_trend, 6),
                leading_indicator=leading,
            ))

        return annotations

    def _classify_phase(
        self, reward_mean: float, reward_trend: float, pr_mean: float, pr_trend: float
    ) -> TrainingPhase:
        """Classify phase from windowed metrics."""
        if pr_mean < self.disruption_threshold and reward_mean < 0.1:
            return TrainingPhase.DISRUPTION

        if reward_mean > 0.8:
            return TrainingPhase.SATURATION

        if pr_trend > self.breakout_pass_rate_slope and reward_mean > 0.2:
            return TrainingPhase.BREAKOUT

        if reward_trend > self.accumulation_reward_slope:
            return TrainingPhase.ACCUMULATION

        if reward_mean < 0.15 and abs(reward_trend) < 0.001:
            return TrainingPhase.STAGNATION

        return TrainingPhase.ACCUMULATION

    @staticmethod
    def _linear_trend(values: np.ndarray) -> float:
        """Compute linear trend (slope) of a sequence."""
        n = len(values)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = values.mean()
        num = np.sum((x - x_mean) * (values - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den < 1e-12:
            return 0.0
        return float(num / den)

    def find_transitions(self) -> list[dict]:
        """Find phase transition points in the annotated data."""
        annotations = self.annotate()
        if len(annotations) < 2:
            return []

        transitions = []
        for i in range(1, len(annotations)):
            if annotations[i].phase != annotations[i - 1].phase:
                transitions.append({
                    "step": annotations[i].step,
                    "from": annotations[i - 1].phase.value,
                    "to": annotations[i].phase.value,
                    "reward_mean": annotations[i].reward_mean,
                    "pass_rate": annotations[i].pass_rate,
                    "leading_indicator": annotations[i].leading_indicator,
                })

        return transitions

    def leading_indicator_report(self) -> dict:
        """Analyze the leading indicator pattern.

        Key question: Does the reward average lead the pass rate?
        If yes, by how many steps?
        """
        annotations = self.annotate()
        if not annotations:
            return {"sufficient_data": False}

        # Find first step where reward trend > threshold
        reward_climb_step = None
        for a in annotations:
            if a.reward_trend > self.accumulation_reward_slope:
                reward_climb_step = a.step
                break

        # Find first step where pass rate trend > threshold
        pr_climb_step = None
        for a in annotations:
            if a.phase == TrainingPhase.BREAKOUT:
                pr_climb_step = a.step
                break

        if reward_climb_step is not None and pr_climb_step is not None:
            lead_steps = pr_climb_step - reward_climb_step
        else:
            lead_steps = None

        return {
            "sufficient_data": True,
            "reward_climb_step": reward_climb_step,
            "pass_rate_breakout_step": pr_climb_step,
            "lead_steps": lead_steps,
            "lead_confirmed": lead_steps is not None and lead_steps > 0,
            "total_steps_analyzed": len(annotations),
        }

    def methodology_report(self) -> dict:
        """Full report for the three-phase methodology paper (App 5)."""
        annotations = self.annotate()
        transitions = self.find_transitions()
        leading = self.leading_indicator_report()

        phase_counts = {}
        for a in annotations:
            phase_counts[a.phase.value] = phase_counts.get(a.phase.value, 0) + 1

        return {
            "title": "Three-Phase GRPO Training Analysis",
            "total_steps": len(self.steps),
            "annotated_steps": len(annotations),
            "phase_distribution": phase_counts,
            "transitions": transitions,
            "leading_indicator": leading,
            "final_pass_rate": self.pass_rates[-1] if self.pass_rates else None,
            "final_reward": self.rewards[-1] if self.rewards else None,
            "key_claim": (
                "The disruption phase is not failure. "
                "The accumulation plateau is not a ceiling. "
                "The running reward average is the leading indicator."
            ),
        }
