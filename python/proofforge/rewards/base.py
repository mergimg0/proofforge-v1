"""Abstract reward oracle protocol.

Every reward oracle in ProofForge produces a RewardResult containing:
  - reward: float in [0, 1]
  - metadata: dict with oracle-specific diagnostic info

The protocol is intentionally minimal so it can wrap:
  - Lean 4 type checker (binary: 0 or 1)
  - Test suite execution (binary: all pass or not)
  - Efficiency-weighted (continuous: correctness * efficiency factor)
  - Shaped (continuous: partial credit from failure-mode analysis)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardResult:
    """Result of a reward oracle evaluation."""

    reward: float
    """Scalar reward in [0, 1]. For binary oracles: exactly 0.0 or 1.0."""

    verified: bool
    """Whether the underlying verifier accepted the solution."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Oracle-specific diagnostics (proof length, error category, etc.)."""

    def __post_init__(self):
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError(f"Reward must be in [0, 1], got {self.reward}")


class RewardOracle(ABC):
    """Protocol for all ProofForge reward oracles.

    Implementations must be deterministic given the same input — the SOS
    monotone improvement property depends on reward consistency.
    """

    @abstractmethod
    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        """Evaluate a candidate solution against its specification.

        Args:
            statement: The problem statement (theorem, code task, etc.)
            solution: The candidate solution (proof, code, etc.)
            **kwargs: Oracle-specific parameters

        Returns:
            RewardResult with reward in [0, 1] and diagnostic metadata.
        """

    def batch_evaluate(
        self, pairs: list[tuple[str, str]], **kwargs
    ) -> list[RewardResult]:
        """Evaluate multiple (statement, solution) pairs.

        Default implementation is sequential. Subclasses can override
        for parallel execution (e.g., parallel Lean workers).
        """
        return [self.evaluate(stmt, sol, **kwargs) for stmt, sol in pairs]
