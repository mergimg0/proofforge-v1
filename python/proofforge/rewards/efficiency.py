"""App 6: Trajectory Length as Training Signal — efficiency-weighted reward.

Replaces binary reward with efficiency-weighted reward for solved theorems:

    reward(proof) = lean_verified(proof) * (1.0 - α * length / max_length)

Where α is adaptive per-theorem based on temperature breadth (mastery depth):
  - breadth >= 3: α = 0.5 (strong efficiency pressure on mastered theorems)
  - breadth >= 2: α = 0.2 (moderate pressure)
  - breadth < 2:  α = 0.0 (pure binary — don't destabilize fragile capabilities)

Phase-in: binary-only for first N steps (bootstrap), then switch.

This preserves SOS monotone improvement: for any fixed theorem, the efficiency
reward is a monotone function of proof quality (shorter correct proofs always
score higher), and incorrect proofs still get 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from proofforge.rewards.base import RewardOracle, RewardResult


@dataclass
class TemperatureBreadthTracker:
    """Track at how many temperatures each theorem is solved reliably.

    Temperature breadth measures internalization depth:
    - breadth 1: solved only at one specific temperature (lucky sampling)
    - breadth 2: solved at 2 temperatures (partial internalization)
    - breadth 3+: solved across temperatures (deep internalization)

    The model has LEARNED the proof strategy, not just found it by chance.
    """

    temperatures: tuple[float, ...] = (0.2, 0.6, 1.0)
    """Temperature levels to test breadth at."""

    success_counts: dict[str, dict[float, int]] = field(default_factory=dict)
    """theorem_id -> {temperature -> success_count}"""

    attempt_counts: dict[str, dict[float, int]] = field(default_factory=dict)
    """theorem_id -> {temperature -> attempt_count}"""

    proof_lengths: dict[str, dict[float, list[int]]] = field(default_factory=dict)
    """theorem_id -> {temperature -> [proof_token_lengths]} for successful proofs.
    Tracks proof lengths per temperature for trajectory length ratio computation."""

    breadth_threshold: float = 0.5
    """Minimum pass rate at a temperature to count as "solved at this temp"."""

    min_attempts: int = 4
    """Minimum attempts at a temperature before counting it."""

    def record(
        self, theorem_id: str, temperature: float, success: bool, proof_tokens: int = 0
    ) -> None:
        """Record a proof attempt at a given temperature."""
        if theorem_id not in self.success_counts:
            self.success_counts[theorem_id] = {}
            self.attempt_counts[theorem_id] = {}
            self.proof_lengths[theorem_id] = {}

        t_key = self._snap_temperature(temperature)
        self.success_counts[theorem_id][t_key] = (
            self.success_counts[theorem_id].get(t_key, 0) + (1 if success else 0)
        )
        self.attempt_counts[theorem_id][t_key] = (
            self.attempt_counts[theorem_id].get(t_key, 0) + 1
        )

        if success and proof_tokens > 0:
            if t_key not in self.proof_lengths[theorem_id]:
                self.proof_lengths[theorem_id][t_key] = []
            self.proof_lengths[theorem_id][t_key].append(proof_tokens)

    def breadth(self, theorem_id: str) -> int:
        """How many temperatures does this theorem pass at reliably?"""
        if theorem_id not in self.success_counts:
            return 0

        count = 0
        for temp in self.temperatures:
            t_key = self._snap_temperature(temp)
            attempts = self.attempt_counts.get(theorem_id, {}).get(t_key, 0)
            successes = self.success_counts.get(theorem_id, {}).get(t_key, 0)

            if attempts >= self.min_attempts:
                rate = successes / attempts
                if rate >= self.breadth_threshold:
                    count += 1
        return count

    def _snap_temperature(self, temp: float) -> float:
        """Snap to nearest tracked temperature."""
        return min(self.temperatures, key=lambda t: abs(t - temp))

    def mastery_summary(self) -> dict[str, dict]:
        """Return mastery info for all tracked theorems."""
        result = {}
        for thm_id in self.success_counts:
            b = self.breadth(thm_id)
            per_temp = {}
            for temp in self.temperatures:
                t_key = self._snap_temperature(temp)
                att = self.attempt_counts.get(thm_id, {}).get(t_key, 0)
                suc = self.success_counts.get(thm_id, {}).get(t_key, 0)
                per_temp[t_key] = {"attempts": att, "successes": suc, "rate": suc / att if att > 0 else 0.0}
            result[thm_id] = {"breadth": b, "temperatures": per_temp}
        return result


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency-weighted reward."""

    alpha_mastered: float = 0.5
    """Efficiency pressure for fully mastered theorems (breadth >= 3)."""

    alpha_moderate: float = 0.2
    """Efficiency pressure for moderately mastered theorems (breadth >= 2)."""

    alpha_fragile: float = 0.0
    """Efficiency pressure for fragile theorems (breadth < 2). Should be 0."""

    max_proof_tokens: int = 512
    """Maximum proof length in tokens for normalization."""

    phase_in_step: int = 50
    """Training step at which to switch from binary to efficiency reward."""

    phase_in_pass_rate: float = 0.20
    """Minimum pass rate before switching to efficiency reward."""

    length_floor: float = 0.3
    """Minimum reward for a correct proof regardless of length.
    Prevents the model from being penalized too harshly for long-but-correct proofs."""


@dataclass
class EfficiencyReward(RewardOracle):
    """Efficiency-weighted reward oracle (App 6).

    Wraps a base oracle (typically LeanOracle) and modulates the reward
    by proof length for mastered theorems.

    The key insight: once a theorem is reliably solved (breadth >= 2),
    gradient saturation kills further learning. The efficiency signal
    restores gradient by differentiating short (good) from long (okay) proofs.

    SOS preservation: For a fixed theorem, efficiency reward is:
      - 0.0 for incorrect proofs (same as binary)
      - monotone decreasing in proof length for correct proofs
      - bounded in [config.length_floor, 1.0] for correct proofs

    Since the evaluator E = mean reward over theorems, and each theorem's
    reward is a monotone function of proof quality, the overall evaluator
    is still monotone under the GRPO update. Convergence is preserved.
    """

    base_oracle: RewardOracle = None  # type: ignore[assignment]
    """The underlying verifier (LeanOracle for theorem proving)."""

    config: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    """Efficiency reward configuration."""

    breadth_tracker: TemperatureBreadthTracker = field(
        default_factory=TemperatureBreadthTracker
    )
    """Per-theorem temperature breadth tracking."""

    current_step: int = 0
    """Current training step (for phase-in logic)."""

    current_pass_rate: float = 0.0
    """Current overall pass rate (for phase-in logic)."""

    def _compute_alpha(self, theorem_id: str) -> float:
        """Per-theorem adaptive alpha based on temperature breadth.

        Returns 0.0 during bootstrap phase (before phase_in_step or below
        phase_in_pass_rate). After phase-in, returns alpha proportional
        to mastery depth.
        """
        if self.current_step < self.config.phase_in_step:
            return 0.0
        if self.current_pass_rate < self.config.phase_in_pass_rate:
            return 0.0

        b = self.breadth_tracker.breadth(theorem_id)
        if b >= 3:
            return self.config.alpha_mastered
        elif b >= 2:
            return self.config.alpha_moderate
        else:
            return self.config.alpha_fragile

    def _efficiency_factor(self, alpha: float, proof_length: int) -> float:
        """Compute efficiency scaling factor.

        Returns a value in [length_floor, 1.0]:
          factor = max(length_floor, 1.0 - alpha * length / max_length)

        Short proofs → factor ≈ 1.0
        Long proofs → factor → length_floor (never 0, to avoid discouraging
        correct-but-verbose proofs entirely)
        """
        if alpha == 0.0:
            return 1.0

        length_ratio = min(proof_length / self.config.max_proof_tokens, 1.0)
        raw = 1.0 - alpha * length_ratio
        return max(raw, self.config.length_floor)

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        """Evaluate with efficiency weighting.

        Args:
            statement: Theorem statement
            solution: Proof text
            theorem_id: Identifier for breadth tracking (default: hash of statement)
            temperature: Sampling temperature used (for breadth recording)
            proof_tokens: Token count of proof (default: estimated from chars)
        """
        theorem_id = kwargs.get("theorem_id", str(hash(statement)))
        temperature = kwargs.get("temperature", None)
        proof_tokens = kwargs.get("proof_tokens", self._estimate_tokens(solution))

        base_result = self.base_oracle.evaluate(statement, solution, **kwargs)

        if temperature is not None:
            self.breadth_tracker.record(
                theorem_id, temperature, base_result.verified, proof_tokens
            )

        if not base_result.verified:
            return RewardResult(
                reward=0.0,
                verified=False,
                metadata={
                    **base_result.metadata,
                    "efficiency_alpha": 0.0,
                    "efficiency_factor": 0.0,
                    "proof_tokens": proof_tokens,
                    "reward_mode": "efficiency",
                },
            )

        alpha = self._compute_alpha(theorem_id)
        factor = self._efficiency_factor(alpha, proof_tokens)
        reward = base_result.reward * factor

        return RewardResult(
            reward=reward,
            verified=True,
            metadata={
                **base_result.metadata,
                "efficiency_alpha": alpha,
                "efficiency_factor": factor,
                "proof_tokens": proof_tokens,
                "theorem_breadth": self.breadth_tracker.breadth(theorem_id),
                "reward_mode": "efficiency" if alpha > 0 else "binary",
            },
        )

    def update_training_state(self, step: int, pass_rate: float) -> None:
        """Called by the training loop to update phase-in state."""
        self.current_step = step
        self.current_pass_rate = pass_rate

    def trajectory_length_ratio(self, theorem_id: str) -> Optional[float]:
        """Compute low-T / high-T proof length ratio.

        This metric should decrease toward 1.0 as proof strategies are
        promoted from secondary (high-T) to primary (low-T) modes.

        Uses the lowest and highest tracked temperatures from the
        breadth tracker's proof_lengths data.

        Returns None if insufficient data at both temperature extremes.
        """
        thm_lengths = self.breadth_tracker.proof_lengths.get(theorem_id, {})
        if not thm_lengths:
            return None

        temps = self.breadth_tracker.temperatures
        low_t = temps[0]   # lowest temperature
        high_t = temps[-1]  # highest temperature

        low_t_key = self.breadth_tracker._snap_temperature(low_t)
        high_t_key = self.breadth_tracker._snap_temperature(high_t)

        low_lengths = thm_lengths.get(low_t_key, [])
        high_lengths = thm_lengths.get(high_t_key, [])

        if not low_lengths or not high_lengths:
            return None

        mean_low = sum(low_lengths) / len(low_lengths)
        mean_high = sum(high_lengths) / len(high_lengths)
        return mean_low / max(mean_high, 1.0)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for code/math."""
        return max(1, len(text) // 4)


@dataclass
class TrajectoryLengthMetrics:
    """Aggregate metrics for monitoring App 6 effectiveness.

    Track these per checkpoint:
    - mean_proof_length: should decrease if efficiency signal works
    - trajectory_length_ratio: should approach 1.0
    - pass_rate: must not decrease (SOS monotone improvement)
    """

    step: int = 0
    mean_proof_length: float = 0.0
    median_proof_length: float = 0.0
    trajectory_length_ratio: float = 0.0
    pass_rate: float = 0.0
    theorems_with_efficiency: int = 0
    theorems_binary_only: int = 0

    @staticmethod
    def compute(
        results: list[RewardResult],
        step: int,
    ) -> TrajectoryLengthMetrics:
        """Compute metrics from a batch of reward results."""
        verified = [r for r in results if r.verified]
        if not verified:
            return TrajectoryLengthMetrics(step=step)

        lengths = [r.metadata.get("proof_tokens", 0) for r in verified]
        lengths_sorted = sorted(lengths)

        eff_count = sum(
            1 for r in verified if r.metadata.get("efficiency_alpha", 0) > 0
        )

        return TrajectoryLengthMetrics(
            step=step,
            mean_proof_length=sum(lengths) / len(lengths),
            median_proof_length=lengths_sorted[len(lengths_sorted) // 2],
            pass_rate=len(verified) / len(results),
            theorems_with_efficiency=eff_count,
            theorems_binary_only=len(verified) - eff_count,
        )
