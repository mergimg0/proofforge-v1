"""Infrastructure pattern tracking for App 2: Infrastructure-Aware GRPO.

Tracks which coding patterns (analogous to Lean tactics) appear in
successful code solutions. Used to select training tasks at the
infrastructure frontier — tasks where some required patterns are
mastered and others aren't.

This is the code-domain analog of the TacticRegistry from App 2:
instead of Lean tactics (simp, rfl, induction), we track coding
patterns (error_handling, recursion, list_comprehension).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import ast
import re


# Infrastructure patterns detectable from code AST
PATTERN_DETECTORS: dict[str, type] = {}


def _detect_patterns(code: str) -> set[str]:
    """Detect infrastructure patterns in Python code via AST analysis."""
    patterns: set[str] = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return patterns

    for node in ast.walk(tree):
        # Iteration patterns
        if isinstance(node, ast.For):
            patterns.add("iteration")
        if isinstance(node, ast.While):
            patterns.add("iteration")
            patterns.add("while_loop")
        if isinstance(node, ast.ListComp):
            patterns.add("list_comprehension")
        if isinstance(node, ast.DictComp):
            patterns.add("dict_comprehension")
        if isinstance(node, ast.GeneratorExp):
            patterns.add("generator")

        # Error handling
        if isinstance(node, ast.Try):
            patterns.add("error_handling")
        if isinstance(node, ast.Raise):
            patterns.add("error_handling")

        # Function/class definition
        if isinstance(node, ast.FunctionDef):
            patterns.add("function_definition")
            if any(isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod", "property")
                   for d in node.decorator_list):
                patterns.add("decorators")
        if isinstance(node, ast.ClassDef):
            patterns.add("class_definition")

        # Recursion detection (function calls own name)
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id == func_name:
                        patterns.add("recursion")

        # Data structure operations
        if isinstance(node, ast.Subscript):
            patterns.add("index_arithmetic")
        if isinstance(node, ast.Slice):
            patterns.add("slicing")

        # Comparison and conditionals
        if isinstance(node, ast.If):
            patterns.add("conditional")
        if isinstance(node, ast.Compare):
            patterns.add("comparison")
        if isinstance(node, ast.BoolOp):
            patterns.add("boolean_logic")

        # String operations (detected by method calls)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in ("split", "join", "strip", "replace", "find", "startswith", "endswith"):
                patterns.add("string_ops")
            if method in ("append", "extend", "insert", "pop", "remove", "sort"):
                patterns.add("list_ops")
            if method in ("get", "keys", "values", "items", "update", "setdefault"):
                patterns.add("dict_ops")

        # Modular arithmetic
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            patterns.add("modular_arithmetic")

    # Type checking patterns (isinstance, type annotations)
    code_str = code
    if "isinstance(" in code_str:
        patterns.add("type_checking")
    if re.search(r"->|:\s*(int|str|float|bool|list|dict|tuple|Optional)", code_str):
        patterns.add("type_annotations")

    return patterns


@dataclass
class TacticRegistry:
    """Track coding pattern (tactic) success rates.

    Records which patterns appear in successful vs. failed code solutions.
    Provides mastery scores per pattern for infrastructure-aware sampling.
    """

    pattern_attempts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    pattern_successes: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record(self, code: str, passed: bool) -> set[str]:
        """Record patterns found in a code solution attempt.

        Returns the set of detected patterns.
        """
        patterns = _detect_patterns(code)

        for pattern in patterns:
            self.pattern_attempts[pattern] += 1
            if passed:
                self.pattern_successes[pattern] += 1

        return patterns

    def mastery(self, pattern: str) -> float:
        """Success rate for a specific pattern. 0.0 if never seen."""
        attempts = self.pattern_attempts.get(pattern, 0)
        if attempts == 0:
            return 0.0
        return self.pattern_successes.get(pattern, 0) / attempts

    def all_masteries(self) -> dict[str, float]:
        """Mastery scores for all observed patterns."""
        return {p: self.mastery(p) for p in self.pattern_attempts}

    def frontier_patterns(self, low: float = 0.2, high: float = 0.8) -> list[str]:
        """Patterns at the frontier: partially mastered.

        Patterns with mastery between low and high are the ones where
        additional training has the most impact — the model knows SOMETHING
        but hasn't fully internalized the pattern.
        """
        return [
            p for p in self.pattern_attempts
            if low <= self.mastery(p) <= high
        ]

    def infrastructure_score(self, required: tuple[str, ...]) -> float:
        """Score a task's infrastructure requirements against current mastery.

        High score = mix of mastered and unmastered patterns (frontier).
        Used by infrastructure-aware sampling (App 2).

        Score = std(masteries) * mean(masteries)
        Peak when some patterns are mastered (>0.5) and others aren't (<0.2).
        """
        if not required:
            return 0.0

        import numpy as np
        masteries = [self.mastery(p) for p in required]
        if not masteries:
            return 0.0

        return float(np.std(masteries) * np.mean(masteries))

    def summary(self) -> dict[str, dict]:
        """Full summary of pattern tracking."""
        return {
            pattern: {
                "attempts": self.pattern_attempts[pattern],
                "successes": self.pattern_successes.get(pattern, 0),
                "mastery": round(self.mastery(pattern), 3),
            }
            for pattern in sorted(self.pattern_attempts)
        }
