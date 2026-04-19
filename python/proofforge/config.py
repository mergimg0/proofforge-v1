"""Unified configuration for ProofForge applications.

Single entry point for configuring all applications. Loads from YAML/JSON
and provides typed access to per-application configs.

Usage:
    config = ProofForgeConfig.from_yaml("config.yaml")
    efficiency_reward = EfficiencyReward(
        base_oracle=LeanOracle(workers=config.lean.workers),
        config=config.efficiency,
    )
    controller = AdaptiveController(config=config.controller)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from proofforge.rewards.efficiency import EfficiencyConfig
from proofforge.rewards.shaped import ShapedConfig
from proofforge.controller.controller import ControllerConfig
from proofforge.curriculum.expanding_ring import RingConfig
from proofforge.code_verify.task_loader import TaskLevel


@dataclass
class LeanConfig:
    """Lean 4 oracle configuration."""
    timeout: int = 30
    workers: int = 8


@dataclass
class GRPOTrainingConfig:
    """GRPO training loop configuration."""
    group_size: int = 16
    learning_rate: float = 1e-6
    kl_beta: float = 0.01
    epsilon_clip: float = 0.2
    max_steps: int = 200
    checkpoint_schedule: list[int] = field(
        default_factory=lambda: [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    )
    temperatures: list[float] = field(
        default_factory=lambda: [0.2, 0.6, 1.0]
    )
    unlikeliness_beta: float = 0.25


@dataclass
class CodeVerifyConfig:
    """App 1: Code verification pipeline configuration."""
    initial_level: TaskLevel = TaskLevel.EXPRESSION
    level_up_threshold: float = 0.6
    infrastructure_aware: bool = True
    frontier_weight: float = 0.7
    test_timeout: int = 30
    test_memory_mb: int = 256
    test_workers: int = 4
    custom_tasks_path: Optional[str] = None


@dataclass
class ProofForgeConfig:
    """Unified configuration for all ProofForge applications."""

    # Core training
    grpo: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)

    # App 6: Efficiency reward
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)

    # App 8: Adaptive controller
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    # App 1: Code verification
    code_verify: CodeVerifyConfig = field(default_factory=CodeVerifyConfig)

    # Idea Reactor: Shaped reward
    shaped: ShapedConfig = field(default_factory=ShapedConfig)

    # Curriculum
    ring: RingConfig = field(default_factory=RingConfig)

    # Output
    output_dir: str = "./proofforge_output"
    seed: int = 42

    @classmethod
    def from_json(cls, path: str) -> "ProofForgeConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            raw = json.load(f)
        return cls._from_dict(raw)

    @classmethod
    def from_yaml(cls, path: str) -> "ProofForgeConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "ProofForgeConfig":
        """Build config from a nested dict."""
        config = cls()

        if "grpo" in raw:
            for k, v in raw["grpo"].items():
                if hasattr(config.grpo, k):
                    setattr(config.grpo, k, v)

        if "lean" in raw:
            for k, v in raw["lean"].items():
                if hasattr(config.lean, k):
                    setattr(config.lean, k, v)

        if "efficiency" in raw:
            for k, v in raw["efficiency"].items():
                if hasattr(config.efficiency, k):
                    setattr(config.efficiency, k, v)

        if "controller" in raw:
            for k, v in raw["controller"].items():
                if hasattr(config.controller, k):
                    setattr(config.controller, k, v)

        if "code_verify" in raw:
            for k, v in raw["code_verify"].items():
                if hasattr(config.code_verify, k):
                    if k == "initial_level":
                        v = TaskLevel(v)
                    setattr(config.code_verify, k, v)

        if "shaped" in raw:
            for k, v in raw["shaped"].items():
                if hasattr(config.shaped, k):
                    setattr(config.shaped, k, v)

        if "ring" in raw:
            for k, v in raw["ring"].items():
                if hasattr(config.ring, k):
                    setattr(config.ring, k, v)

        for key in ("output_dir", "seed"):
            if key in raw:
                setattr(config, key, raw[key])

        return config

    def to_dict(self) -> dict:
        """Serialize config to a dict for logging."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
