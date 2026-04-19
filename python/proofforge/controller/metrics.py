"""Metric computation and windowed statistics for the training controller.

Provides rolling-window statistics that power phase detection and
intervention decisions. All computations are O(1) amortized via
deque-based sliding windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricWindow:
    """Fixed-size sliding window with O(1) statistics."""

    max_size: int = 20
    _values: deque = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self):
        self._values = deque(maxlen=self.max_size)

    def push(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self._values) / len(self._values)
        return variance ** 0.5

    @property
    def trend(self) -> float:
        """Linear trend (slope) via least-squares regression.

        Positive = improving, negative = declining, near-zero = flat.
        """
        n = len(self._values)
        if n < 3:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = self.mean

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(self._values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-12:
            return 0.0
        return numerator / denominator

    @property
    def last(self) -> Optional[float]:
        return self._values[-1] if self._values else None

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def is_full(self) -> bool:
        return len(self._values) >= self.max_size

    def recent(self, n: int) -> list[float]:
        """Last n values."""
        return list(self._values)[-n:]


@dataclass
class ControllerMetrics:
    """Aggregated training metrics for the adaptive controller.

    Tracks rolling windows for:
    - reward: GRPO reward signal (LEADING indicator)
    - pass_rate: theorem pass rate (LAGGING indicator)
    - loss: training loss
    - gradient_norm: gradient magnitude (bounded step proxy)
    - proof_length: mean proof token count (efficiency metric)
    """

    reward: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    pass_rate: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    loss: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    gradient_norm: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    proof_length: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    retention_rate: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    """Retention rate: fraction of previously-solved theorems still solved.
    The 9.1% → 95.2% retention jump is the consolidation signal."""

    tactic_diversity: MetricWindow = field(default_factory=lambda: MetricWindow(max_size=20))
    """Tactic diversity: number of distinct first-tactics used across proofs.
    Low diversity = simp monoculture."""

    step_count: int = 0

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
    ) -> None:
        """Record metrics for a training step."""
        self.step_count = step
        self.reward.push(reward)
        if pass_rate is not None:
            self.pass_rate.push(pass_rate)
        if loss is not None:
            self.loss.push(loss)
        if gradient_norm is not None:
            self.gradient_norm.push(gradient_norm)
        if proof_length is not None:
            self.proof_length.push(proof_length)
        if retention_rate is not None:
            self.retention_rate.push(retention_rate)
        if tactic_diversity is not None:
            self.tactic_diversity.push(tactic_diversity)

    def reward_leads_pass_rate(self) -> Optional[bool]:
        """Detect if reward average is climbing while pass rate is flat.

        This is the KEY leading indicator pattern: reward average started
        climbing at step 35; pass rate broke out at step 50. A positive
        reward trend with flat pass rate = infrastructure is building.
        """
        if self.reward.count < 5 or self.pass_rate.count < 5:
            return None

        reward_trend = self.reward.trend
        pass_rate_trend = self.pass_rate.trend

        return reward_trend > 0.005 and abs(pass_rate_trend) < 0.003

    def gradient_saturation(self) -> bool:
        """Detect gradient saturation: reward > 0.8 consistently.

        When most GRPO groups have 6-7/8 successes, advantage separation
        collapses. The model can't distinguish "good" from "better".
        """
        if self.reward.count < 10:
            return False
        return self.reward.mean > 0.8 and self.reward.std < 0.1

    def stagnation(self, threshold_steps: int = 50) -> bool:
        """Detect stagnation: reward flat below 0.15 for many steps."""
        if self.step_count < threshold_steps:
            return False
        if self.reward.count < 10:
            return False
        return self.reward.mean < 0.15 and abs(self.reward.trend) < 0.001

    def consolidation_detected(self) -> bool:
        """Detect consolidation: retention has jumped to high and stable.

        The consolidation phase is when previously-solved theorems STOP
        flickering and become reliably solved. The paper's sharpest finding:
        retention jumped from 9.1% to 95.2% between steps 30-75.

        Triggers when retention_rate mean > 0.7 (theorems stable) AND
        reward is not yet in saturation territory (< 0.8). This
        distinguishes consolidation (high retention, still improving)
        from saturation (high retention, gradient collapsed).
        """
        if self.retention_rate.count < 5:
            return False
        return (
            self.retention_rate.mean > 0.7
            and self.reward.mean < 0.8  # not yet saturated
        )

    def tactic_monoculture(self, min_diversity: float = 3.0) -> bool:
        """Detect tactic monoculture: diversity below threshold.

        At C150, 67% of first tactics were simp. Diversity < 3 distinct
        tactics across a group signals monoculture risk.
        """
        if self.tactic_diversity.count < 5:
            return False
        return self.tactic_diversity.mean < min_diversity

    def snapshot(self) -> dict:
        """Current state for logging/diagnostics."""
        return {
            "step": self.step_count,
            "reward_mean": round(self.reward.mean, 4),
            "reward_trend": round(self.reward.trend, 6),
            "pass_rate_mean": round(self.pass_rate.mean, 4),
            "pass_rate_trend": round(self.pass_rate.trend, 6),
            "loss_mean": round(self.loss.mean, 4) if self.loss.count > 0 else None,
            "grad_norm_mean": round(self.gradient_norm.mean, 4) if self.gradient_norm.count > 0 else None,
            "proof_length_mean": round(self.proof_length.mean, 1) if self.proof_length.count > 0 else None,
            "retention_mean": round(self.retention_rate.mean, 4) if self.retention_rate.count > 0 else None,
            "tactic_diversity_mean": round(self.tactic_diversity.mean, 2) if self.tactic_diversity.count > 0 else None,
            "leading_indicator": self.reward_leads_pass_rate(),
            "gradient_saturated": self.gradient_saturation(),
            "stagnated": self.stagnation(),
            "consolidating": self.consolidation_detected(),
            "tactic_monoculture": self.tactic_monoculture(),
        }
