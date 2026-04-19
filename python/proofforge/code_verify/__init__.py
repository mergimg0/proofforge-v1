"""App 1: Self-Bootstrapping Code Verification Pipeline.

Applies the ProofForge pattern to code: model generates code, test suite
verifies it, binary reward feeds back through GRPO. Infrastructure learning
predicts the model will learn coding PATTERNS (error handling, iteration,
API usage) that transfer across tasks.

Fractal curriculum: Level 0 (single expression) → Level 1 (single function) →
Level 2 (multi-function) → Level 3 (module-level).
"""

from proofforge.code_verify.pipeline import CodeVerificationPipeline, PipelineConfig
from proofforge.code_verify.task_loader import CodeTask, TaskLevel, TaskLoader
from proofforge.code_verify.tactic_registry import TacticRegistry

__all__ = [
    "CodeVerificationPipeline",
    "PipelineConfig",
    "CodeTask",
    "TaskLevel",
    "TaskLoader",
    "TacticRegistry",
]
