"""App 2: Infrastructure-Aware GRPO Sampling.

Instead of sampling theorems/tasks uniformly, bias toward tasks at the
infrastructure frontier: tasks where some required patterns are mastered
and others aren't. This accelerates the self-accelerating loop by
deliberately presenting frontier tasks instead of waiting for random
sampling to hit them.

Scoring: score = std(masteries) * mean(masteries)
  - Peak when mix of mastered (>0.5) and unmastered (<0.2) patterns
  - Zero when all patterns are at similar mastery levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, Callable

import numpy as np

T = TypeVar("T")


@dataclass
class InfrastructureAwareSampler(Generic[T]):
    """Infrastructure-aware task sampler for GRPO training.

    Selects tasks that require infrastructure the model has PARTIALLY
    learned — maximizing the gradient signal from each training step.

    Includes tactic diversity pressure: tasks requiring under-represented
    tactics get a bonus, tasks solvable by the dominant tactic (simp)
    get a penalty. This prevents monoculture.

    Generic over item type T (theorems, code tasks, etc.).
    """

    get_required_infra: Callable[[T], tuple[str, ...]]
    """Extract required infrastructure patterns from an item."""

    mastery_fn: Callable[[str], float]
    """Current mastery level for a pattern name."""

    tactic_usage_fn: Callable[[str], float] | None = None
    """Optional: returns usage frequency for a tactic (0-1).
    High frequency = over-represented (simp). Low = under-represented (cases).
    When provided, diversity pressure is applied."""

    uniform_fraction: float = 0.3
    """Fraction of samples drawn uniformly (exploration)."""

    diversity_bonus: float = 2.0
    """Multiplier for tasks requiring under-represented tactics.
    Applied when a required tactic has usage frequency < 0.1."""

    monoculture_penalty: float = 0.3
    """Multiplier for tasks solvable entirely by over-represented tactics.
    Applied when ALL required tactics have usage frequency > 0.5."""

    def score(self, item: T) -> float:
        """Score an item by infrastructure frontier value + diversity pressure.

        Base score: std(masteries) * mean(masteries)
        Diversity modifier:
          - Bonus if item requires under-represented tactics
          - Penalty if item only needs over-represented tactics (simp)
        """
        required = self.get_required_infra(item)
        if not required:
            return 0.0

        masteries = np.array([self.mastery_fn(p) for p in required])
        if len(masteries) == 0:
            return 0.0

        base_score = float(np.std(masteries) * np.mean(masteries))

        # Apply diversity pressure if tactic usage data is available
        if self.tactic_usage_fn is not None:
            usages = [self.tactic_usage_fn(p) for p in required]
            has_rare = any(u < 0.1 for u in usages)
            all_common = all(u > 0.5 for u in usages)

            if has_rare:
                base_score *= self.diversity_bonus
            elif all_common:
                base_score *= self.monoculture_penalty

        return max(base_score, 1e-6)  # never fully zero — allow exploration

    def select(self, items: list[T], n: int = 1) -> list[T]:
        """Select n items biased toward the infrastructure frontier.

        With probability uniform_fraction, samples uniformly (exploration).
        Otherwise, samples proportional to frontier score (exploitation).
        """
        if not items:
            return []

        rng = np.random.default_rng()

        # Split budget between exploration and exploitation
        n_uniform = max(1, int(n * self.uniform_fraction))
        n_frontier = n - n_uniform

        selected = []

        # Uniform exploration
        if n_uniform > 0:
            uniform_idx = rng.choice(len(items), size=min(n_uniform, len(items)), replace=False)
            selected.extend(items[i] for i in uniform_idx)

        # Frontier exploitation
        if n_frontier > 0:
            scores = np.array([self.score(item) for item in items])
            total = scores.sum()

            if total > 0:
                probs = scores / total
            else:
                probs = np.ones(len(items)) / len(items)

            frontier_idx = rng.choice(
                len(items), size=min(n_frontier, len(items)), replace=True, p=probs
            )
            selected.extend(items[i] for i in frontier_idx)

        return selected[:n]

    def rank(self, items: list[T]) -> list[tuple[T, float]]:
        """Rank all items by frontier score (descending)."""
        scored = [(item, self.score(item)) for item in items]
        scored.sort(key=lambda x: -x[1])
        return scored
