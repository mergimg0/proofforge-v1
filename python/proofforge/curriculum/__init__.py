"""Curriculum strategies for GRPO training.

Implements task selection strategies that keep theorems/tasks in the
"frontier zone" — where the model has partial infrastructure and gradient
signal is richest.
"""

from proofforge.curriculum.expanding_ring import ExpandingRing, RingConfig
from proofforge.curriculum.infrastructure_aware import InfrastructureAwareSampler

__all__ = [
    "ExpandingRing",
    "RingConfig",
    "InfrastructureAwareSampler",
]
