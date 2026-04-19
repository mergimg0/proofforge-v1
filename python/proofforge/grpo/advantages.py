"""GRPO advantage computation via group normalization.

The key innovation: no learned value function.

    A_i = (r_i - mean(r)) / std(r)

Advantages are relative to the group, not to a learned baseline.
This eliminates approximation error from V(s) estimation.

On tasks with binary verification (Lean type-checking, test suites),
this makes GRPO a concrete SOS — no axioms needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupAdvantages:
    """GRPO advantages for a group of proof/code attempts."""
    advantages: np.ndarray
    group_mean: float
    group_std: float
    group_size: int


def compute_advantages(rewards: list[float]) -> GroupAdvantages:
    """Compute GRPO group-normalized advantages.

    A_i = (r_i - mean(r)) / max(std(r), ε)

    Returns zero advantages for homogeneous groups (all same reward)
    since there is no gradient signal.
    """
    n = len(rewards)
    if n == 0:
        return GroupAdvantages(
            advantages=np.array([]),
            group_mean=0.0,
            group_std=1.0,
            group_size=0,
        )

    r = np.array(rewards)
    mean = float(r.mean())
    std = float(r.std())

    if std < 1e-8:
        return GroupAdvantages(
            advantages=np.zeros(n),
            group_mean=mean,
            group_std=std,
            group_size=n,
        )

    advantages = (r - mean) / std

    return GroupAdvantages(
        advantages=advantages,
        group_mean=mean,
        group_std=std,
        group_size=n,
    )


def advantage_is_useful(advantages: GroupAdvantages, min_std: float = 0.01) -> bool:
    """Check if a group produces useful gradient signal.

    Groups where all rewards are identical (all pass or all fail)
    produce zero advantage and zero gradient. Skip these for efficiency.
    """
    return advantages.group_std > min_std
