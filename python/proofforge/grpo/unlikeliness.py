"""Round 3: Unlikeliness reward reweighting.

Re-weights rewards so that rare, lower-probability successes get amplified
while common high-probability successes are discounted. Prevents mode
collapse and encourages exploration.

Formula:
    r_i_unlike = r_i * (1 - beta_rank * (G - rank_i) / G)

Where rank_i is by log-probability (rank 0 = most probable).

Effect:
    - Most likely completion (rank 0): penalized by (1 - beta_rank)
    - Least likely completion (rank G-1): retains full reward
    - Mid-rank: intermediate scaling
"""

from __future__ import annotations

import numpy as np


def unlikeliness_reweight(
    rewards: list[float],
    log_probs: list[float],
    beta_rank: float = 0.25,
) -> list[float]:
    """Apply unlikeliness reweighting to GRPO rewards.

    Args:
        rewards: Binary rewards (0.0 or 1.0) per completion.
        log_probs: Log-probability of each completion under current policy.
        beta_rank: Reweighting strength (0 = no reweighting, 1 = full).

    Returns:
        Reweighted rewards. Rare correct proofs get higher reward.
    """
    n = len(rewards)
    if n == 0:
        return []

    # Rank by log-probability (descending: rank 0 = most probable)
    indices = np.argsort(log_probs)[::-1]
    ranks = np.empty(n, dtype=int)
    for rank, idx in enumerate(indices):
        ranks[idx] = rank

    reweighted = []
    for i in range(n):
        factor = 1.0 - beta_rank * (n - ranks[i]) / n
        reweighted.append(rewards[i] * factor)

    return reweighted
