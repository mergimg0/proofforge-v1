"""Tests for proofforge.code_verify.tactic_registry."""

from __future__ import annotations

import pytest

from proofforge.code_verify.tactic_registry import (
    TacticRegistry,
    _detect_patterns,
)


# ---------------------------------------------------------------------------
# _detect_patterns — pattern detection via AST analysis
# ---------------------------------------------------------------------------


class TestDetectPatternsIteration:
    """For-loop code is detected as 'iteration'."""

    def test_for_loop_detects_iteration(self):
        code = "for i in range(10):\n    print(i)"
        patterns = _detect_patterns(code)
        assert "iteration" in patterns

    def test_while_loop_detects_iteration(self):
        code = "while x > 0:\n    x -= 1"
        patterns = _detect_patterns(code)
        assert "iteration" in patterns

    def test_while_loop_also_detects_while_loop(self):
        code = "while True:\n    break"
        patterns = _detect_patterns(code)
        assert "while_loop" in patterns

    def test_for_loop_does_not_detect_while_loop(self):
        code = "for i in range(5):\n    pass"
        patterns = _detect_patterns(code)
        assert "while_loop" not in patterns


class TestDetectPatternsRecursion:
    """Recursive function calls are detected as 'recursion'."""

    def test_recursive_function_detects_recursion(self):
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        patterns = _detect_patterns(code)
        assert "recursion" in patterns

    def test_non_recursive_function_no_recursion(self):
        code = "def add(a, b):\n    return a + b\n"
        patterns = _detect_patterns(code)
        assert "recursion" not in patterns

    def test_mutual_recursion_not_detected_for_other_name(self):
        # Only self-calls are detected; calling a different function is not recursion
        code = (
            "def is_even(n):\n"
            "    if n == 0:\n"
            "        return True\n"
            "    return is_odd(n - 1)\n"
            "def is_odd(n):\n"
            "    if n == 0:\n"
            "        return False\n"
            "    return is_even(n - 1)\n"
        )
        patterns = _detect_patterns(code)
        # Neither function calls itself directly, so 'recursion' is not detected
        assert "recursion" not in patterns


class TestDetectPatternsErrorHandling:
    """Try/except and raise statements are detected as 'error_handling'."""

    def test_try_except_detects_error_handling(self):
        code = "try:\n    x = int('abc')\nexcept ValueError:\n    x = 0\n"
        patterns = _detect_patterns(code)
        assert "error_handling" in patterns

    def test_raise_detects_error_handling(self):
        code = "def f(x):\n    if x < 0:\n        raise ValueError('negative')\n"
        patterns = _detect_patterns(code)
        assert "error_handling" in patterns

    def test_no_exception_handling_absent(self):
        code = "x = 1 + 2\n"
        patterns = _detect_patterns(code)
        assert "error_handling" not in patterns


class TestDetectPatternsClassDefinition:
    """Class definitions are detected as 'class_definition'."""

    def test_class_definition_detected(self):
        code = "class MyClass:\n    pass\n"
        patterns = _detect_patterns(code)
        assert "class_definition" in patterns

    def test_class_with_methods_detected(self):
        code = (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "    def push(self, x):\n"
            "        self.items.append(x)\n"
        )
        patterns = _detect_patterns(code)
        assert "class_definition" in patterns

    def test_no_class_keyword_no_class_definition(self):
        code = "def f():\n    return 42\n"
        patterns = _detect_patterns(code)
        assert "class_definition" not in patterns


class TestDetectPatternsListComprehension:
    """List comprehensions are detected as 'list_comprehension'."""

    def test_list_comprehension_detected(self):
        code = "result = [x * 2 for x in range(10)]\n"
        patterns = _detect_patterns(code)
        assert "list_comprehension" in patterns

    def test_list_comp_with_condition_detected(self):
        code = "evens = [x for x in range(20) if x % 2 == 0]\n"
        patterns = _detect_patterns(code)
        assert "list_comprehension" in patterns

    def test_dict_comprehension_is_not_list_comprehension(self):
        code = "d = {k: v for k, v in items}\n"
        patterns = _detect_patterns(code)
        assert "list_comprehension" not in patterns
        assert "dict_comprehension" in patterns

    def test_no_comprehension_absent(self):
        code = "x = list(range(10))\n"
        patterns = _detect_patterns(code)
        assert "list_comprehension" not in patterns


class TestDetectPatternsEmptyOrInvalid:
    """Empty or syntactically invalid code yields an empty pattern set."""

    def test_empty_string_returns_empty_set(self):
        patterns = _detect_patterns("")
        assert patterns == set()

    def test_whitespace_only_returns_empty_set(self):
        patterns = _detect_patterns("   \n  \t  \n")
        assert patterns == set()

    def test_syntax_error_returns_empty_set(self):
        patterns = _detect_patterns("def (broken syntax !!!")
        assert patterns == set()

    def test_invalid_code_does_not_raise(self):
        # Must not propagate SyntaxError
        try:
            result = _detect_patterns("class : pass")
            assert isinstance(result, set)
        except SyntaxError:
            pytest.fail("_detect_patterns should not raise SyntaxError")


class TestDetectPatternsTypeChecking:
    """isinstance() usage is detected as 'type_checking'."""

    def test_isinstance_detected(self):
        code = "if isinstance(x, int):\n    pass\n"
        patterns = _detect_patterns(code)
        assert "type_checking" in patterns

    def test_isinstance_in_condition_detected(self):
        code = (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        )
        patterns = _detect_patterns(code)
        assert "type_checking" in patterns

    def test_no_isinstance_no_type_checking(self):
        code = "x = type(42)\n"
        patterns = _detect_patterns(code)
        assert "type_checking" not in patterns


class TestDetectPatternsModularArithmetic:
    """The modulo operator (%) is detected as 'modular_arithmetic'."""

    def test_modulo_detected(self):
        code = "remainder = n % 2\n"
        patterns = _detect_patterns(code)
        assert "modular_arithmetic" in patterns

    def test_modulo_in_condition_detected(self):
        code = "def is_even(n):\n    return n % 2 == 0\n"
        patterns = _detect_patterns(code)
        assert "modular_arithmetic" in patterns

    def test_no_modulo_absent(self):
        code = "x = 10 + 5\ny = x * 3\n"
        patterns = _detect_patterns(code)
        assert "modular_arithmetic" not in patterns

    def test_modulo_in_prime_check_detected(self):
        code = (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, n):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        )
        patterns = _detect_patterns(code)
        assert "modular_arithmetic" in patterns


class TestDetectPatternsReturnType:
    """_detect_patterns always returns a set."""

    def test_returns_set_for_valid_code(self):
        assert isinstance(_detect_patterns("x = 1"), set)

    def test_returns_set_for_empty_code(self):
        assert isinstance(_detect_patterns(""), set)

    def test_returns_set_for_invalid_code(self):
        assert isinstance(_detect_patterns("!!!"), set)


# ---------------------------------------------------------------------------
# TacticRegistry — record, mastery, all_masteries, frontier_patterns,
#                  infrastructure_score, summary
# ---------------------------------------------------------------------------


class TestTacticRegistryRecord:
    """record() tracks detected patterns and returns them."""

    def test_record_for_loop_returns_iteration(self):
        registry = TacticRegistry()
        patterns = registry.record("for i in range(10): pass", passed=True)
        assert "iteration" in patterns

    def test_record_increments_attempts(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        assert registry.pattern_attempts["iteration"] == 1

    def test_record_passed_true_increments_successes(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        assert registry.pattern_successes["iteration"] == 1

    def test_record_passed_false_does_not_increment_successes(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=False)
        assert registry.pattern_successes.get("iteration", 0) == 0

    def test_record_multiple_calls_accumulate(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        registry.record("for i in range(10): pass", passed=False)
        assert registry.pattern_attempts["iteration"] == 2
        assert registry.pattern_successes.get("iteration", 0) == 1

    def test_record_returns_set_of_patterns(self):
        registry = TacticRegistry()
        result = registry.record("for i in range(5): pass", passed=True)
        assert isinstance(result, set)
        assert len(result) > 0

    def test_record_empty_code_returns_empty_set(self):
        registry = TacticRegistry()
        result = registry.record("", passed=True)
        assert result == set()

    def test_record_syntax_error_code_returns_empty_set(self):
        registry = TacticRegistry()
        result = registry.record("def !!!", passed=True)
        assert result == set()


class TestTacticRegistryMastery:
    """mastery() returns the success rate for a pattern."""

    def test_unseen_pattern_returns_zero(self):
        registry = TacticRegistry()
        assert registry.mastery("__never_seen__") == 0.0

    def test_all_pass_returns_one(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        registry.record(code, passed=True)
        registry.record(code, passed=True)
        assert registry.mastery("iteration") == pytest.approx(1.0)

    def test_all_fail_returns_zero(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        registry.record(code, passed=False)
        registry.record(code, passed=False)
        assert registry.mastery("iteration") == pytest.approx(0.0)

    def test_mixed_returns_correct_rate(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        registry.record(code, passed=True)   # 1 success
        registry.record(code, passed=False)  # 1 failure
        registry.record(code, passed=True)   # 2 successes
        # 2 out of 3 attempts
        assert registry.mastery("iteration") == pytest.approx(2 / 3)

    def test_mastery_range_zero_to_one(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        for _ in range(5):
            registry.record(code, passed=True)
        for _ in range(3):
            registry.record(code, passed=False)
        rate = registry.mastery("iteration")
        assert 0.0 <= rate <= 1.0


class TestTacticRegistryAllMasteries:
    """all_masteries() returns a dict of all observed pattern rates."""

    def test_returns_dict(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        assert isinstance(registry.all_masteries(), dict)

    def test_contains_recorded_patterns(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        masteries = registry.all_masteries()
        assert "iteration" in masteries

    def test_values_match_mastery_method(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        registry.record(code, passed=True)
        registry.record(code, passed=False)
        masteries = registry.all_masteries()
        for pattern, rate in masteries.items():
            assert rate == pytest.approx(registry.mastery(pattern))

    def test_empty_registry_returns_empty_dict(self):
        registry = TacticRegistry()
        assert registry.all_masteries() == {}

    def test_all_values_in_zero_one(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        registry.record("try:\n    x = 1\nexcept:\n    pass", passed=False)
        for rate in registry.all_masteries().values():
            assert 0.0 <= rate <= 1.0


class TestTacticRegistryFrontierPatterns:
    """frontier_patterns() returns patterns with mastery between low and high."""

    def test_mastered_pattern_not_in_frontier(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        for _ in range(10):
            registry.record(code, passed=True)
        # iteration mastery = 1.0, above default high=0.8
        assert "iteration" not in registry.frontier_patterns()

    def test_unmastered_pattern_not_in_frontier(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        for _ in range(10):
            registry.record(code, passed=False)
        # iteration mastery = 0.0, below default low=0.2
        assert "iteration" not in registry.frontier_patterns()

    def test_partial_mastery_in_frontier(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        # 5 passes, 5 fails → mastery = 0.5, between 0.2 and 0.8
        for _ in range(5):
            registry.record(code, passed=True)
        for _ in range(5):
            registry.record(code, passed=False)
        assert "iteration" in registry.frontier_patterns()

    def test_frontier_respects_custom_thresholds(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        # mastery = 0.9 — above 0.8 (default high) but between 0.85 and 0.95
        for _ in range(9):
            registry.record(code, passed=True)
        for _ in range(1):
            registry.record(code, passed=False)
        assert "iteration" not in registry.frontier_patterns(low=0.2, high=0.8)
        assert "iteration" in registry.frontier_patterns(low=0.85, high=0.95)

    def test_returns_list(self):
        registry = TacticRegistry()
        assert isinstance(registry.frontier_patterns(), list)

    def test_empty_registry_returns_empty_list(self):
        registry = TacticRegistry()
        assert registry.frontier_patterns() == []


class TestTacticRegistryInfrastructureScore:
    """infrastructure_score() is high for mixed mastery, low for uniform."""

    def test_empty_required_returns_zero(self):
        registry = TacticRegistry()
        assert registry.infrastructure_score(()) == pytest.approx(0.0)

    def test_all_unseen_uniform_low_returns_low_score(self):
        registry = TacticRegistry()
        # All patterns have mastery 0.0 — uniform, std=0
        score = registry.infrastructure_score(("recursion", "iteration", "error_handling"))
        assert score == pytest.approx(0.0)

    def test_mixed_mastery_returns_high_score(self):
        registry = TacticRegistry()
        # Set recursion to high mastery, iteration to low mastery
        for _ in range(9):
            registry.record(
                "def f(n):\n    if n <= 0: return 0\n    return f(n-1)",
                passed=True,
            )
        registry.record(
            "def f(n):\n    if n <= 0: return 0\n    return f(n-1)",
            passed=False,
        )
        # Now add iteration with low mastery
        for _ in range(9):
            registry.record("for i in range(5): pass", passed=False)
        registry.record("for i in range(5): pass", passed=True)

        # recursion ~0.9, iteration ~0.1 → std is high, mean is 0.5 → high score
        score = registry.infrastructure_score(("recursion", "iteration"))
        assert score > 0.0

    def test_all_mastered_returns_low_score(self):
        registry = TacticRegistry()
        code_for = "for i in range(5): pass"
        code_rec = "def f(n):\n    if n <= 0: return 0\n    return f(n-1)"
        for _ in range(10):
            registry.record(code_for, passed=True)
            registry.record(code_rec, passed=True)
        # Both at 1.0 — uniform, std=0
        score = registry.infrastructure_score(("recursion", "iteration"))
        assert score == pytest.approx(0.0)

    def test_single_pattern_std_is_zero(self):
        registry = TacticRegistry()
        for _ in range(5):
            registry.record("for i in range(5): pass", passed=True)
        for _ in range(5):
            registry.record("for i in range(5): pass", passed=False)
        # Single element: std=0 → score=0
        score = registry.infrastructure_score(("iteration",))
        assert score == pytest.approx(0.0)


class TestTacticRegistrySummary:
    """summary() returns a structured dict of all pattern stats."""

    def test_returns_dict(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        assert isinstance(registry.summary(), dict)

    def test_summary_has_all_recorded_patterns(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        summary = registry.summary()
        assert "iteration" in summary

    def test_summary_entry_has_required_keys(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        entry = registry.summary()["iteration"]
        assert "attempts" in entry
        assert "successes" in entry
        assert "mastery" in entry

    def test_summary_counts_correct(self):
        registry = TacticRegistry()
        code = "for i in range(5): pass"
        registry.record(code, passed=True)
        registry.record(code, passed=True)
        registry.record(code, passed=False)
        entry = registry.summary()["iteration"]
        assert entry["attempts"] == 3
        assert entry["successes"] == 2
        assert entry["mastery"] == pytest.approx(round(2 / 3, 3))

    def test_summary_is_sorted_alphabetically(self):
        registry = TacticRegistry()
        registry.record("for i in range(5): pass", passed=True)
        registry.record("try:\n    x=1\nexcept:\n    pass", passed=True)
        keys = list(registry.summary().keys())
        assert keys == sorted(keys)

    def test_empty_registry_summary_is_empty(self):
        registry = TacticRegistry()
        assert registry.summary() == {}
