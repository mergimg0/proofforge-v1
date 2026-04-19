"""App 8: Adaptive Training Controller.

Automated monitoring daemon that detects phase transitions, reward saturation,
and capability plateaus in GRPO training. Makes intervention decisions based
on the three-phase model (disruption → accumulation → breakout → saturation).

The running reward average LEADS the pass rate by ~10-20 steps. The controller
uses this lead time to predict phase transitions before they happen.
"""

from proofforge.controller.phase_detector import PhaseDetector, TrainingPhase
from proofforge.controller.controller import AdaptiveController, ControllerConfig
from proofforge.controller.interventions import Intervention, InterventionType
from proofforge.controller.metrics import ControllerMetrics, MetricWindow

__all__ = [
    "PhaseDetector",
    "TrainingPhase",
    "AdaptiveController",
    "ControllerConfig",
    "Intervention",
    "InterventionType",
    "ControllerMetrics",
    "MetricWindow",
]
