"""Code verification GRPO pipeline (App 1).

Orchestrates the full code verification training loop:
  1. Select tasks from fractal curriculum
  2. Generate code solutions (LLM inference)
  3. Verify via test suite execution (binary reward)
  4. Compute GRPO advantages
  5. Update policy
  6. Track infrastructure patterns (tactic registry)

Expected three-phase dynamics:
  Phase 1 (disruption): Code quality drops as GRPO reshapes distribution
  Phase 2 (accumulation): Model learns patterns (try/except, list comps, etc.)
  Phase 3 (breakout): Pattern infrastructure hits critical mass
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from proofforge.rewards.test_suite import TestSuiteOracle
from proofforge.code_verify.task_loader import CodeTask, TaskLevel, TaskLoader
from proofforge.code_verify.tactic_registry import TacticRegistry
from proofforge.controller.controller import AdaptiveController, ControllerConfig


@dataclass
class PipelineConfig:
    """Configuration for the code verification pipeline."""

    # GRPO parameters
    group_size: int = 16
    """Number of code solutions generated per task."""

    max_tokens: int = 512
    """Maximum token length for generated code."""

    temperatures: tuple[float, ...] = (0.2, 0.6, 1.0)
    """Sampling temperatures for diversity."""

    # Curriculum
    initial_level: TaskLevel = TaskLevel.EXPRESSION
    """Start with easiest tasks."""

    level_up_threshold: float = 0.6
    """Pass rate above this at current level triggers level-up."""

    # Infrastructure-aware sampling (App 2)
    infrastructure_aware: bool = True
    """Use tactic registry to bias task selection toward frontier."""

    frontier_weight: float = 0.7
    """Fraction of tasks sampled from frontier vs. uniform."""

    # Controller (App 8)
    controller_enabled: bool = True
    """Enable adaptive training controller."""

    # Output
    output_dir: Optional[str] = None
    log_every_n_steps: int = 5


@dataclass
class StepResult:
    """Result of one GRPO training step."""
    step: int
    task_id: str
    level: TaskLevel
    group_size: int
    rewards: list[float]
    pass_rate: float
    mean_reward: float
    patterns_detected: dict[str, int]
    duration_ms: int


@dataclass
class CodeVerificationPipeline:
    """Full code verification GRPO pipeline (App 1).

    This is the code-domain realization of the ProofForge pattern.
    Instead of Lean proofs, the model generates Python code.
    Instead of the type-checker, test suites provide binary reward.

    The infrastructure learning hypothesis: the model will learn
    coding PATTERNS (error handling, iteration, API usage) that
    transfer across tasks, producing the same three-phase dynamics.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)

    task_loader: TaskLoader = field(default_factory=TaskLoader)
    oracle: TestSuiteOracle = field(default_factory=TestSuiteOracle)
    registry: TacticRegistry = field(default_factory=TacticRegistry)
    controller: Optional[AdaptiveController] = field(default=None, init=False)

    # State
    current_level: TaskLevel = field(init=False)
    step_history: list[StepResult] = field(default_factory=list)
    _level_pass_rates: dict[TaskLevel, list[float]] = field(default_factory=dict)

    # Callback: generate code solutions (pluggable LLM interface)
    generate_fn: Optional[Callable] = None
    """Signature: generate_fn(prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]"""

    def __post_init__(self):
        self.current_level = self.config.initial_level

        # Register all tasks with the oracle
        for task in self.task_loader.all_tasks:
            self.oracle.register_task(task.task_id, task.test_code)

        # Initialize controller if enabled
        if self.config.controller_enabled:
            self.controller = AdaptiveController(
                config=ControllerConfig(
                    log_dir=self.config.output_dir,
                    verbose=True,
                )
            )

    def select_task(self) -> CodeTask:
        """Select next training task using curriculum + infrastructure awareness.

        Priority:
        1. If infrastructure_aware: bias toward frontier tasks
        2. Otherwise: sample from current difficulty level
        """
        import random

        level_tasks = self.task_loader.by_level(self.current_level)

        if self.config.infrastructure_aware and random.random() < self.config.frontier_weight:
            masteries = self.registry.all_masteries()
            frontier = self.task_loader.frontier_tasks(masteries)
            # Filter to current level or one level up
            eligible = [
                t for t in frontier
                if t.level <= min(self.current_level + 1, TaskLevel.MODULE)
            ]
            if eligible:
                # Score by infrastructure frontier value
                scored = [
                    (t, self.registry.infrastructure_score(t.required_infrastructure))
                    for t in eligible
                ]
                scored.sort(key=lambda x: -x[1])
                return scored[0][0]

        if level_tasks:
            return random.choice(level_tasks)

        return random.choice(self.task_loader.all_tasks)

    def run_step(self, step: int) -> StepResult:
        """Execute one GRPO training step.

        1. Select task
        2. Generate solutions
        3. Verify with test suite
        4. Track patterns
        5. Update controller
        6. Check level-up
        """
        start = time.monotonic()

        task = self.select_task()
        temperature = self.config.temperatures[step % len(self.config.temperatures)]

        # Generate solutions
        if self.generate_fn is not None:
            solutions = self.generate_fn(
                task.prompt,
                self.config.group_size,
                temperature,
                self.config.max_tokens,
            )
        else:
            solutions = [f"# placeholder solution {i}" for i in range(self.config.group_size)]

        # Verify each solution
        pairs = [(task.prompt, sol) for sol in solutions]
        results = self.oracle.batch_evaluate(
            pairs, task_ids=[task.task_id] * len(pairs)
        )

        # Track infrastructure patterns
        pattern_counts: dict[str, int] = {}
        for sol, result in zip(solutions, results):
            detected = self.registry.record(sol, result.verified)
            for p in detected:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1

        # Compute metrics
        rewards = [r.reward for r in results]
        pass_rate = sum(1 for r in results if r.verified) / len(results)
        mean_reward = sum(rewards) / len(rewards)

        # Update controller
        if self.controller is not None:
            self.controller.update(
                step=step,
                reward=mean_reward,
                pass_rate=pass_rate,
            )

        # Track level pass rates
        if self.current_level not in self._level_pass_rates:
            self._level_pass_rates[self.current_level] = []
        self._level_pass_rates[self.current_level].append(pass_rate)

        # Check level-up
        self._check_level_up()

        duration_ms = int((time.monotonic() - start) * 1000)

        step_result = StepResult(
            step=step,
            task_id=task.task_id,
            level=task.level,
            group_size=len(solutions),
            rewards=rewards,
            pass_rate=pass_rate,
            mean_reward=mean_reward,
            patterns_detected=pattern_counts,
            duration_ms=duration_ms,
        )
        self.step_history.append(step_result)

        return step_result

    def _check_level_up(self) -> None:
        """Promote to next curriculum level if pass rate is high enough."""
        rates = self._level_pass_rates.get(self.current_level, [])
        if len(rates) < 10:
            return

        recent = rates[-10:]
        avg_rate = sum(recent) / len(recent)

        if avg_rate >= self.config.level_up_threshold:
            next_level = min(self.current_level + 1, TaskLevel.MODULE)
            if next_level != self.current_level:
                self.current_level = TaskLevel(next_level)

    def summary(self) -> dict:
        """Pipeline execution summary."""
        return {
            "total_steps": len(self.step_history),
            "current_level": self.current_level.name,
            "level_pass_rates": {
                level.name: (
                    round(sum(rates) / len(rates), 3) if rates else 0.0
                )
                for level, rates in self._level_pass_rates.items()
            },
            "pattern_masteries": self.registry.all_masteries(),
            "frontier_patterns": self.registry.frontier_patterns(),
            "controller": self.controller.summary() if self.controller else None,
        }
