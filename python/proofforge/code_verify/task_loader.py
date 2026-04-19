"""Code task definitions and loading for the fractal curriculum.

Tasks are organized in 4 difficulty levels following the fractal curriculum
principle: each level's solutions use infrastructure from the previous level.

Level 0: Single-expression (fibonacci(10), reverse a string)
Level 1: Single-function (binary search, parse CSV line)
Level 2: Multi-function (class with 2-3 methods)
Level 3: Module-level (small library with interacting components)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TaskLevel(IntEnum):
    """Fractal curriculum difficulty levels."""
    EXPRESSION = 0
    FUNCTION = 1
    MULTI_FUNCTION = 2
    MODULE = 3


@dataclass(frozen=True)
class CodeTask:
    """A code generation task with its test suite.

    The task_id uniquely identifies this task for the TestSuiteOracle.
    The prompt is what the model sees. The test_code is the verification
    oracle (hidden from the model).
    """

    task_id: str
    """Unique identifier (e.g., 'l0_fibonacci', 'l2_linked_list')."""

    level: TaskLevel
    """Curriculum difficulty level."""

    prompt: str
    """The task description shown to the model. Includes function signature,
    docstring with examples, and expected behavior."""

    test_code: str
    """Test suite code that calls _run_test(name, fn) for each test case.
    This is the reward oracle — hidden from the model."""

    canonical_solution: Optional[str] = None
    """Reference solution for validation (not shown to model)."""

    required_infrastructure: tuple[str, ...] = ()
    """Infrastructure patterns needed (e.g., ('error_handling', 'iteration')).
    Used by infrastructure-aware sampling (App 2)."""

    tags: tuple[str, ...] = ()
    """Categorization tags (e.g., ('string', 'recursion', 'sorting'))."""


# -----------------------------------------------------------------------
# Built-in task library: starter tasks for each curriculum level
# -----------------------------------------------------------------------

BUILTIN_TASKS: list[CodeTask] = [
    # --- Level 0: Single-expression tasks ---
    CodeTask(
        task_id="l0_fibonacci",
        level=TaskLevel.EXPRESSION,
        prompt='def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number (0-indexed). fibonacci(0)=0, fibonacci(1)=1, fibonacci(10)=55."""',
        test_code='\n'.join([
            '_run_test("fib_0", lambda: fibonacci(0) == 0)',
            '_run_test("fib_1", lambda: fibonacci(1) == 1)',
            '_run_test("fib_10", lambda: fibonacci(10) == 55)',
            '_run_test("fib_20", lambda: fibonacci(20) == 6765)',
        ]),
        canonical_solution="def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        required_infrastructure=("iteration",),
        tags=("math", "recursion"),
    ),
    CodeTask(
        task_id="l0_reverse_string",
        level=TaskLevel.EXPRESSION,
        prompt='def reverse_string(s: str) -> str:\n    """Reverse a string. reverse_string("hello") == "olleh"."""',
        test_code='\n'.join([
            '_run_test("empty", lambda: reverse_string("") == "")',
            '_run_test("single", lambda: reverse_string("a") == "a")',
            '_run_test("hello", lambda: reverse_string("hello") == "olleh")',
            '_run_test("palindrome", lambda: reverse_string("racecar") == "racecar")',
        ]),
        canonical_solution='def reverse_string(s: str) -> str:\n    return s[::-1]',
        required_infrastructure=("slicing",),
        tags=("string",),
    ),
    CodeTask(
        task_id="l0_is_prime",
        level=TaskLevel.EXPRESSION,
        prompt='def is_prime(n: int) -> bool:\n    """Return True if n is a prime number. is_prime(2)==True, is_prime(4)==False."""',
        test_code='\n'.join([
            '_run_test("zero", lambda: is_prime(0) == False)',
            '_run_test("one", lambda: is_prime(1) == False)',
            '_run_test("two", lambda: is_prime(2) == True)',
            '_run_test("four", lambda: is_prime(4) == False)',
            '_run_test("17", lambda: is_prime(17) == True)',
            '_run_test("100", lambda: is_prime(100) == False)',
        ]),
        required_infrastructure=("iteration", "modular_arithmetic"),
        tags=("math",),
    ),

    # --- Level 1: Single-function tasks ---
    CodeTask(
        task_id="l1_binary_search",
        level=TaskLevel.FUNCTION,
        prompt='def binary_search(arr: list[int], target: int) -> int:\n    """Return index of target in sorted arr, or -1 if not found."""',
        test_code='\n'.join([
            '_run_test("found_mid", lambda: binary_search([1,3,5,7,9], 5) == 2)',
            '_run_test("found_first", lambda: binary_search([1,3,5,7,9], 1) == 0)',
            '_run_test("found_last", lambda: binary_search([1,3,5,7,9], 9) == 4)',
            '_run_test("not_found", lambda: binary_search([1,3,5,7,9], 4) == -1)',
            '_run_test("empty", lambda: binary_search([], 1) == -1)',
            '_run_test("single_hit", lambda: binary_search([5], 5) == 0)',
            '_run_test("single_miss", lambda: binary_search([5], 3) == -1)',
        ]),
        required_infrastructure=("iteration", "comparison", "index_arithmetic"),
        tags=("search", "algorithm"),
    ),
    CodeTask(
        task_id="l1_flatten_list",
        level=TaskLevel.FUNCTION,
        prompt='def flatten(lst: list) -> list:\n    """Flatten a nested list of arbitrary depth. flatten([1,[2,[3,4],5]]) == [1,2,3,4,5]."""',
        test_code='\n'.join([
            '_run_test("simple", lambda: flatten([1, 2, 3]) == [1, 2, 3])',
            '_run_test("nested", lambda: flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5])',
            '_run_test("empty", lambda: flatten([]) == [])',
            '_run_test("deep", lambda: flatten([[[1]], [[2]], [[3]]]) == [1, 2, 3])',
            '_run_test("mixed", lambda: flatten([1, [2], [[3]], [[[4]]]]) == [1, 2, 3, 4])',
        ]),
        required_infrastructure=("recursion", "type_checking", "list_ops"),
        tags=("recursion", "list"),
    ),
    CodeTask(
        task_id="l1_parse_csv_line",
        level=TaskLevel.FUNCTION,
        prompt='def parse_csv_line(line: str) -> list[str]:\n    """Parse a CSV line handling quoted fields. parse_csv_line(\'a,"b,c",d\') == ["a","b,c","d"]."""',
        test_code='\n'.join([
            '_run_test("simple", lambda: parse_csv_line("a,b,c") == ["a", "b", "c"])',
            '_run_test("quoted", lambda: parse_csv_line(\'a,"b,c",d\') == ["a", "b,c", "d"])',
            '_run_test("empty_field", lambda: parse_csv_line("a,,c") == ["a", "", "c"])',
            '_run_test("single", lambda: parse_csv_line("hello") == ["hello"])',
            '_run_test("empty", lambda: parse_csv_line("") == [""])',
        ]),
        required_infrastructure=("string_parsing", "state_machine", "error_handling"),
        tags=("parsing", "string"),
    ),

    # --- Level 2: Multi-function / class tasks ---
    CodeTask(
        task_id="l2_stack",
        level=TaskLevel.MULTI_FUNCTION,
        prompt='''class Stack:
    """A stack with push, pop, peek, is_empty, and size methods.
    Raises IndexError on pop/peek of empty stack."""
''',
        test_code='\n'.join([
            '_run_test("empty", lambda: Stack().is_empty() == True)',
            'def _t_push_pop():',
            '    s = Stack(); s.push(1); s.push(2); return s.pop() == 2',
            '_run_test("push_pop", _t_push_pop)',
            'def _t_peek():',
            '    s = Stack(); s.push(42); return s.peek() == 42 and s.size() == 1',
            '_run_test("peek", _t_peek)',
            'def _t_pop_empty():',
            '    _assert_raises(IndexError, lambda: Stack().pop()); return True',
            '_run_test("pop_empty", _t_pop_empty)',
            'def _t_size():',
            '    s = Stack(); s.push(1); s.push(2); s.push(3); return s.size() == 3',
            '_run_test("size", _t_size)',
        ]),
        required_infrastructure=("class_definition", "error_handling", "list_ops"),
        tags=("data_structure", "class"),
    ),
    CodeTask(
        task_id="l2_lru_cache",
        level=TaskLevel.MULTI_FUNCTION,
        prompt='''class LRUCache:
    """Least Recently Used cache with get(key) and put(key, value).
    get returns -1 if key not found. Evicts least recently used on capacity overflow."""
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...
''',
        test_code='\n'.join([
            'def _t_basic():',
            '    c = LRUCache(2); c.put(1, 1); c.put(2, 2); return c.get(1) == 1',
            '_run_test("basic", _t_basic)',
            'def _t_evict():',
            '    c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.put(3, 3); return c.get(1) == -1',
            '_run_test("evict", _t_evict)',
            'def _t_recency():',
            '    c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3)',
            '    return c.get(2) == -1 and c.get(1) == 1',
            '_run_test("update_recency", _t_recency)',
            'def _t_overwrite():',
            '    c = LRUCache(1); c.put(1, 10); c.put(1, 20); return c.get(1) == 20',
            '_run_test("overwrite", _t_overwrite)',
        ]),
        required_infrastructure=("class_definition", "dict_ops", "linked_list_or_ordered_dict"),
        tags=("data_structure", "class", "algorithm"),
    ),

    # --- Level 3: Module-level tasks ---
    CodeTask(
        task_id="l3_calculator",
        level=TaskLevel.MODULE,
        prompt='''"""Simple expression calculator module.

Implement:
  tokenize(expr: str) -> list[Token]   — break "3 + 4 * 2" into tokens
  parse(tokens: list[Token]) -> AST    — build expression tree (respecting precedence)
  evaluate(ast: AST) -> float          — evaluate the expression tree

Support: +, -, *, /, parentheses, integers, floats.
Raise ValueError on invalid expressions.
"""
''',
        test_code='\n'.join([
            '_run_test("add", lambda: evaluate(parse(tokenize("3 + 4"))) == 7.0)',
            '_run_test("precedence", lambda: evaluate(parse(tokenize("3 + 4 * 2"))) == 11.0)',
            '_run_test("parens", lambda: evaluate(parse(tokenize("(3 + 4) * 2"))) == 14.0)',
            '_run_test("nested", lambda: evaluate(parse(tokenize("((2 + 3) * (4 - 1))"))) == 15.0)',
            '_run_test("division", lambda: abs(evaluate(parse(tokenize("10 / 3"))) - 3.333333) < 0.001)',
            '_run_test("float", lambda: evaluate(parse(tokenize("1.5 + 2.5"))) == 4.0)',
            'def _t_invalid():',
            '    _assert_raises(ValueError, lambda: evaluate(parse(tokenize("3 +")))); return True',
            '_run_test("invalid", _t_invalid)',
        ]),
        required_infrastructure=("tokenizer", "parser", "tree_traversal", "error_handling", "class_definition"),
        tags=("parsing", "interpreter", "module"),
    ),
]


def _assert_raises(exc_type, fn):
    """Helper for test code: assert that fn() raises exc_type."""
    try:
        fn()
        raise AssertionError(f"Expected {exc_type.__name__} but no exception raised")
    except exc_type:
        pass


@dataclass
class TaskLoader:
    """Load and manage code tasks for the fractal curriculum.

    Provides builtin tasks and supports loading custom tasks from JSON.
    """

    tasks: dict[str, CodeTask] = field(default_factory=dict)

    def __post_init__(self):
        for task in BUILTIN_TASKS:
            self.tasks[task.task_id] = task

    def load_from_json(self, path: str) -> int:
        """Load additional tasks from a JSON file.

        Expected format: list of objects with task_id, level, prompt,
        test_code, and optional canonical_solution, required_infrastructure, tags.

        Returns number of tasks loaded.
        """
        with open(path) as f:
            raw = json.load(f)

        count = 0
        for entry in raw:
            task = CodeTask(
                task_id=entry["task_id"],
                level=TaskLevel(entry["level"]),
                prompt=entry["prompt"],
                test_code=entry["test_code"],
                canonical_solution=entry.get("canonical_solution"),
                required_infrastructure=tuple(entry.get("required_infrastructure", [])),
                tags=tuple(entry.get("tags", [])),
            )
            self.tasks[task.task_id] = task
            count += 1

        return count

    def by_level(self, level: TaskLevel) -> list[CodeTask]:
        """Get all tasks at a given difficulty level."""
        return [t for t in self.tasks.values() if t.level == level]

    def by_infrastructure(self, *patterns: str) -> list[CodeTask]:
        """Get tasks requiring specific infrastructure patterns."""
        pattern_set = set(patterns)
        return [
            t for t in self.tasks.values()
            if pattern_set & set(t.required_infrastructure)
        ]

    def by_tag(self, *tags: str) -> list[CodeTask]:
        """Get tasks matching any of the given tags."""
        tag_set = set(tags)
        return [t for t in self.tasks.values() if tag_set & set(t.tags)]

    def frontier_tasks(self, mastery: dict[str, float], threshold: float = 0.5) -> list[CodeTask]:
        """Get tasks at the frontier: partially mastered.

        A task is at the frontier when some of its required infrastructure
        is mastered (> threshold) and some isn't. These are the tasks where
        GRPO gradient is most useful — the model has SOME of what it needs.

        This implements infrastructure-aware sampling (App 2).
        """
        frontier = []
        for task in self.tasks.values():
            if not task.required_infrastructure:
                continue
            masteries = [mastery.get(p, 0.0) for p in task.required_infrastructure]
            has_mastered = any(m > threshold for m in masteries)
            has_unmastered = any(m <= threshold for m in masteries)
            if has_mastered and has_unmastered:
                frontier.append(task)
        return frontier

    @property
    def all_tasks(self) -> list[CodeTask]:
        return list(self.tasks.values())

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def level_summary(self) -> dict[str, int]:
        return {level.name: len(self.by_level(level)) for level in TaskLevel}
