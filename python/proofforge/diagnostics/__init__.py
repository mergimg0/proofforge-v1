"""Diagnostics and analysis modules for GRPO training.

Includes Idea Reactor alternatives being verified by Orange Team:
  - CGLE diagnostic: Complex Ginzburg-Landau analysis of training dynamics
  - Symbolic regression: Fit scaling laws to pass rate curves (App 7)
  - Training dynamics: Three-phase model validation (App 5)
"""

from proofforge.diagnostics.symbolic_regression import ScalingLawFitter, SigmoidModel
from proofforge.diagnostics.training_dynamics import ThreePhaseAnalyzer, PhaseAnnotation

__all__ = [
    "ScalingLawFitter",
    "SigmoidModel",
    "ThreePhaseAnalyzer",
    "PhaseAnnotation",
]
