"""Idea Reactor: Shaped reward with partial credit from failure-mode analysis.

Assigns partial credit based on HOW a proof fails, creating gradient signal
even from incorrect proofs. This is the "reward shaping" alternative being
verified by Orange Team.

Failure-mode taxonomy (from grpo_shaped_reward.py observations):
  1.0  — Lean accepts (full reward)
  0.4  — Valid tactic syntax but wrong theorem applied
  0.3  — Correct proof structure but type mismatch
  0.2  — Parseable Lean but tactic failure
  0.1  — Has Lean-like structure but syntax errors
  0.05 — Lean keywords present but garbled
  0.0  — No recognizable Lean content

Risk: Shaped rewards can create local optima where the model produces
syntactically-valid-but-wrong proofs. Mitigation: use shaped reward only
for warmup (first N steps), then switch to binary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from proofforge.rewards.base import RewardOracle, RewardResult


class FailureMode(IntEnum):
    """Proof failure taxonomy, ordered by proximity to correctness."""
    VERIFIED = 6       # Lean accepted
    WRONG_THEOREM = 5  # Valid tactic, wrong application
    TYPE_MISMATCH = 4  # Right structure, type error
    TACTIC_FAIL = 3    # Parseable but tactic doesn't work
    SYNTAX_ERROR = 2   # Lean-like but won't parse
    GARBLED = 1        # Some Lean keywords but broken
    NO_CONTENT = 0     # Nothing recognizable


# Partial credit per failure mode
SHAPED_REWARDS: dict[FailureMode, float] = {
    FailureMode.VERIFIED: 1.0,
    FailureMode.WRONG_THEOREM: 0.4,
    FailureMode.TYPE_MISMATCH: 0.3,
    FailureMode.TACTIC_FAIL: 0.2,
    FailureMode.SYNTAX_ERROR: 0.1,
    FailureMode.GARBLED: 0.05,
    FailureMode.NO_CONTENT: 0.0,
}

# Lean 4 tactic keywords for structural detection
LEAN_TACTICS = frozenset({
    "simp", "rfl", "exact", "apply", "intro", "intros", "cases", "induction",
    "rw", "rewrite", "have", "let", "show", "calc", "omega", "linarith",
    "norm_num", "ring", "field_simp", "push_neg", "contrapose", "by_contra",
    "trivial", "decide", "native_decide", "aesop", "tauto", "constructor",
    "ext", "funext", "congr", "refine", "use", "exists",
})

# Patterns indicating structural proof elements
PROOF_STRUCTURE_RE = re.compile(
    r"(?:by\s|:=\s*by\b|theorem\s|lemma\s|example\s|def\s)",
    re.IGNORECASE,
)

TYPE_MISMATCH_RE = re.compile(
    r"(?:type mismatch|has type|expected type|is not definitionally equal)",
    re.IGNORECASE,
)


def classify_failure(proof_text: str, lean_error: str) -> FailureMode:
    """Classify a proof failure into the taxonomy.

    Uses both the proof text (structural analysis) and the Lean error
    message (semantic analysis) for classification.
    """
    if not proof_text or not proof_text.strip():
        return FailureMode.NO_CONTENT

    text_lower = proof_text.lower().strip()
    error_lower = lean_error.lower() if lean_error else ""

    tactic_count = sum(1 for t in LEAN_TACTICS if t in text_lower)

    if tactic_count == 0 and not PROOF_STRUCTURE_RE.search(proof_text):
        if any(kw in text_lower for kw in ("theorem", "lemma", "by", "sorry")):
            return FailureMode.GARBLED
        return FailureMode.NO_CONTENT

    if not PROOF_STRUCTURE_RE.search(proof_text) and tactic_count <= 1:
        return FailureMode.GARBLED

    if "unknown identifier" in error_lower or "unknown tactic" in error_lower:
        return FailureMode.SYNTAX_ERROR

    if "tactic" in error_lower and ("failed" in error_lower or "no goals" in error_lower):
        return FailureMode.TACTIC_FAIL

    if TYPE_MISMATCH_RE.search(error_lower):
        return FailureMode.TYPE_MISMATCH

    if "application type mismatch" in error_lower or "incorrect number of arguments" in error_lower:
        return FailureMode.WRONG_THEOREM

    if tactic_count >= 2:
        return FailureMode.TACTIC_FAIL

    return FailureMode.SYNTAX_ERROR


@dataclass
class ShapedConfig:
    """Configuration for shaped reward."""

    warmup_steps: int = 50
    """Use shaped reward for first N steps, then switch to binary."""

    custom_rewards: dict[FailureMode, float] = field(default_factory=lambda: dict(SHAPED_REWARDS))
    """Override default partial credit per failure mode."""

    blend_factor: float = 0.0
    """After warmup, optionally blend: (1-blend)*binary + blend*shaped.
    0.0 = pure binary after warmup (recommended).
    0.1 = 90% binary + 10% shaped (gentle shaping)."""


@dataclass
class ShapedReward(RewardOracle):
    """Shaped reward oracle with partial credit from failure analysis.

    Wraps a base oracle and adds partial credit based on failure-mode
    classification. During warmup, provides full shaped signal. After
    warmup, switches to binary (or blended, per config).

    SOS note: Shaped rewards do NOT preserve strict SOS monotone improvement
    because partial credit introduces approximation error. This is a
    HEURISTIC bootstrap — the binary oracle is the formal SOS. Use shaped
    only during warmup to bootstrap, then switch to binary for convergence
    guarantees.
    """

    base_oracle: RewardOracle = None  # type: ignore[assignment]
    config: ShapedConfig = field(default_factory=ShapedConfig)
    current_step: int = 0

    failure_counts: dict[FailureMode, int] = field(
        default_factory=lambda: {mode: 0 for mode in FailureMode}
    )
    """Running count of failure modes for diagnostics."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        base_result = self.base_oracle.evaluate(statement, solution, **kwargs)

        if base_result.verified:
            self.failure_counts[FailureMode.VERIFIED] += 1
            return RewardResult(
                reward=1.0,
                verified=True,
                metadata={
                    **base_result.metadata,
                    "failure_mode": FailureMode.VERIFIED.name,
                    "shaped_reward": 1.0,
                    "binary_reward": 1.0,
                    "reward_source": "shaped" if self._in_warmup() else "binary",
                },
            )

        lean_error = base_result.metadata.get("error", "")
        failure_mode = classify_failure(solution, lean_error)
        self.failure_counts[failure_mode] += 1

        shaped_r = self.config.custom_rewards.get(failure_mode, 0.0)

        if self._in_warmup():
            reward = shaped_r
            source = "shaped"
        elif self.config.blend_factor > 0:
            reward = (1 - self.config.blend_factor) * 0.0 + self.config.blend_factor * shaped_r
            source = "blended"
        else:
            reward = 0.0
            source = "binary"

        return RewardResult(
            reward=reward,
            verified=False,
            metadata={
                **base_result.metadata,
                "failure_mode": failure_mode.name,
                "shaped_reward": shaped_r,
                "binary_reward": 0.0,
                "reward_source": source,
            },
        )

    def _in_warmup(self) -> bool:
        return self.current_step < self.config.warmup_steps

    def update_step(self, step: int) -> None:
        self.current_step = step

    def failure_distribution(self) -> dict[str, int]:
        """Return human-readable failure mode distribution."""
        return {mode.name: self.failure_counts[mode] for mode in FailureMode}
