"""Extended analyses subpackage for GRPO training dynamics."""

from .temperature import analysis_temperature_breadth
from .tactics import extract_first_tactic, analysis_tactic_distribution
from .length_ratio import analysis_length_ratio
from .composition import detect_composition, analysis_tactic_composition
from .regression import analysis_regression
from .sigmoid import sigmoid, analysis_sigmoid_fit
from .__main__ import main

__all__ = [
    "analysis_temperature_breadth",
    "extract_first_tactic",
    "analysis_tactic_distribution",
    "analysis_length_ratio",
    "detect_composition",
    "analysis_tactic_composition",
    "analysis_regression",
    "sigmoid",
    "analysis_sigmoid_fit",
    "main",
]
