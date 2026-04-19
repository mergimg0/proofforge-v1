"""Reward oracles for ProofForge GRPO training.

All reward oracles implement the RewardOracle protocol: given a candidate
solution and a verification mechanism, return a scalar reward in [0, 1].

The binary Lean oracle is the CONCRETE SOS — no approximation error.
Shaped/efficiency rewards preserve SOS monotone improvement by construction.
"""

from proofforge.rewards.base import RewardOracle, RewardResult
from proofforge.rewards.efficiency import EfficiencyReward, EfficiencyConfig
from proofforge.rewards.lean_oracle import LeanOracle
from proofforge.rewards.test_suite import TestSuiteOracle
from proofforge.rewards.shaped import ShapedReward, ShapedConfig

__all__ = [
    "RewardOracle",
    "RewardResult",
    "EfficiencyReward",
    "EfficiencyConfig",
    "LeanOracle",
    "TestSuiteOracle",
    "ShapedReward",
    "ShapedConfig",
]
