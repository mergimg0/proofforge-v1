"""GRPO engine: group relative policy optimization for ProofForge.

Core algorithm that eliminates the value function entirely.
Advantages are computed by group normalization of sampled rewards.
"""

from proofforge.grpo.advantages import compute_advantages, GroupAdvantages
from proofforge.grpo.unlikeliness import unlikeliness_reweight

__all__ = [
    "compute_advantages",
    "GroupAdvantages",
    "unlikeliness_reweight",
]
