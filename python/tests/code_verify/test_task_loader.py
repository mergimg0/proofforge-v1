"""Tests for proofforge.code_verify.task_loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proofforge.code_verify.task_loader import (
    BUILTIN_TASKS,
    CodeTask,
    TaskLevel,
    TaskLoader,
)


# ---------------------------------------------------------------------------
# TaskLevel ordering
# ---------------------------------------------------------------------------


class TestTaskLevelOrdering:
    """TaskLevel values follow the intended curriculum ordering."""

    def test_expression_less_than_function(self):
        assert TaskLevel.EXPRESSION < TaskLevel.FUNCTION

    def test_function_less_than_multi_function(self):
        assert TaskLevel.FUNCTION < TaskLevel.MULTI_FUNCTION

    def test_multi_function_less_than_module(self):
        assert TaskLevel.MULTI_FUNCTION < TaskLevel.MODULE

    def test_full_ordering(self):
        levels = [TaskLevel.EXPRESSION, TaskLevel.FUNCTION, TaskLevel.MULTI_FUNCTION, TaskLevel.MODULE]
        assert levels == sorted(levels)

    def test_expression_is_zero(self):
        assert int(TaskLevel.EXPRESSION) == 0

    def test_module_is_three(self):
        assert int(TaskLevel.MODULE) == 3

    def test_int_enum_comparison_with_int(self):
        assert TaskLevel.EXPRESSION == 0
        assert TaskLevel.FUNCTION == 1
        assert TaskLevel.MULTI_FUNCTION == 2
        assert TaskLevel.MODULE == 3


# ---------------------------------------------------------------------------
# CodeTask creation and frozen fields
# ---------------------------------------------------------------------------


class TestCodeTaskCreation:
    """CodeTask is a frozen dataclass with expected fields."""

    def test_minimal_creation(self):
        task = CodeTask(
            task_id="test_001",
            level=TaskLevel.EXPRESSION,
            prompt="def foo(): ...",
            test_code="_run_test('x', lambda: assert foo() is None)",
        )
        assert task.task_id == "test_001"
        assert task.level == TaskLevel.EXPRESSION
        assert task.prompt == "def foo(): ..."
        assert task.test_code == "_run_test('x', lambda: assert foo() is None)"

    def test_defaults_for_optional_fields(self):
        task = CodeTask(
            task_id="test_002",
            level=TaskLevel.FUNCTION,
            prompt="def bar(): ...",
            test_code="",
        )
        assert task.canonical_solution is None
        assert task.required_infrastructure == ()
        assert task.tags == ()

    def test_full_creation_with_all_fields(self):
        task = CodeTask(
            task_id="test_003",
            level=TaskLevel.MODULE,
            prompt="full prompt",
            test_code="full test code",
            canonical_solution="def foo(): return 42",
            required_infrastructure=("recursion", "error_handling"),
            tags=("math", "algorithm"),
        )
        assert task.canonical_solution == "def foo(): return 42"
        assert task.required_infrastructure == ("recursion", "error_handling")
        assert task.tags == ("math", "algorithm")

    def test_frozen_cannot_set_task_id(self):
        task = CodeTask(
            task_id="frozen_test",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
        )
        with pytest.raises((AttributeError, TypeError)):
            task.task_id = "changed"  # type: ignore[misc]

    def test_frozen_cannot_set_level(self):
        task = CodeTask(
            task_id="frozen_test",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
        )
        with pytest.raises((AttributeError, TypeError)):
            task.level = TaskLevel.MODULE  # type: ignore[misc]

    def test_frozen_cannot_set_prompt(self):
        task = CodeTask(
            task_id="frozen_test",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
        )
        with pytest.raises((AttributeError, TypeError)):
            task.prompt = "new prompt"  # type: ignore[misc]

    def test_frozen_cannot_set_tags(self):
        task = CodeTask(
            task_id="frozen_test",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
            tags=("a",),
        )
        with pytest.raises((AttributeError, TypeError)):
            task.tags = ("b",)  # type: ignore[misc]

    def test_equality_by_value(self):
        t1 = CodeTask(task_id="eq", level=TaskLevel.FUNCTION, prompt="p", test_code="t")
        t2 = CodeTask(task_id="eq", level=TaskLevel.FUNCTION, prompt="p", test_code="t")
        assert t1 == t2

    def test_inequality_on_task_id(self):
        t1 = CodeTask(task_id="a", level=TaskLevel.FUNCTION, prompt="p", test_code="t")
        t2 = CodeTask(task_id="b", level=TaskLevel.FUNCTION, prompt="p", test_code="t")
        assert t1 != t2

    def test_infrastructure_is_tuple_not_list(self):
        task = CodeTask(
            task_id="t",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
            required_infrastructure=("iteration",),
        )
        assert isinstance(task.required_infrastructure, tuple)

    def test_tags_is_tuple_not_list(self):
        task = CodeTask(
            task_id="t",
            level=TaskLevel.EXPRESSION,
            prompt="p",
            test_code="t",
            tags=("math",),
        )
        assert isinstance(task.tags, tuple)


# ---------------------------------------------------------------------------
# TaskLoader — post_init loads BUILTIN_TASKS
# ---------------------------------------------------------------------------


class TestTaskLoaderInit:
    """TaskLoader loads the builtin task library on construction."""

    def test_post_init_loads_builtin_tasks(self):
        loader = TaskLoader()
        assert loader.task_count == 9

    def test_all_builtin_task_ids_present(self):
        loader = TaskLoader()
        builtin_ids = {t.task_id for t in BUILTIN_TASKS}
        loader_ids = {t.task_id for t in loader.all_tasks}
        assert builtin_ids == loader_ids

    def test_tasks_property_is_dict(self):
        loader = TaskLoader()
        assert isinstance(loader.tasks, dict)

    def test_all_tasks_property_returns_list(self):
        loader = TaskLoader()
        assert isinstance(loader.all_tasks, list)
        assert len(loader.all_tasks) == 9

    def test_task_count_property(self):
        loader = TaskLoader()
        assert loader.task_count == len(loader.all_tasks)


# ---------------------------------------------------------------------------
# TaskLoader.by_level
# ---------------------------------------------------------------------------


class TestTaskLoaderByLevel:
    """by_level() returns tasks at the specified difficulty level."""

    def test_expression_returns_3_tasks(self):
        loader = TaskLoader()
        result = loader.by_level(TaskLevel.EXPRESSION)
        assert len(result) == 3

    def test_function_returns_3_tasks(self):
        loader = TaskLoader()
        result = loader.by_level(TaskLevel.FUNCTION)
        assert len(result) == 3

    def test_multi_function_returns_2_tasks(self):
        loader = TaskLoader()
        result = loader.by_level(TaskLevel.MULTI_FUNCTION)
        assert len(result) == 2

    def test_module_returns_1_task(self):
        loader = TaskLoader()
        result = loader.by_level(TaskLevel.MODULE)
        assert len(result) == 1

    def test_level_counts_sum_to_total(self):
        loader = TaskLoader()
        total = sum(len(loader.by_level(lvl)) for lvl in TaskLevel)
        assert total == loader.task_count

    def test_all_returned_tasks_have_correct_level(self):
        loader = TaskLoader()
        for lvl in TaskLevel:
            for task in loader.by_level(lvl):
                assert task.level == lvl

    def test_returns_list(self):
        loader = TaskLoader()
        assert isinstance(loader.by_level(TaskLevel.EXPRESSION), list)


# ---------------------------------------------------------------------------
# TaskLoader.by_tag
# ---------------------------------------------------------------------------


class TestTaskLoaderByTag:
    """by_tag() filters tasks by tag membership."""

    def test_math_tag_returns_tasks(self):
        loader = TaskLoader()
        result = loader.by_tag("math")
        assert len(result) > 0

    def test_math_tagged_tasks_contain_fibonacci_and_is_prime(self):
        loader = TaskLoader()
        task_ids = {t.task_id for t in loader.by_tag("math")}
        assert "l0_fibonacci" in task_ids
        assert "l0_is_prime" in task_ids

    def test_nonexistent_tag_returns_empty(self):
        loader = TaskLoader()
        result = loader.by_tag("__does_not_exist__")
        assert result == []

    def test_returned_tasks_all_have_the_tag(self):
        loader = TaskLoader()
        for task in loader.by_tag("math"):
            assert "math" in task.tags

    def test_string_tag_returns_string_tasks(self):
        loader = TaskLoader()
        result = loader.by_tag("string")
        assert len(result) > 0
        for task in result:
            assert "string" in task.tags

    def test_by_tag_returns_list(self):
        loader = TaskLoader()
        assert isinstance(loader.by_tag("math"), list)


# ---------------------------------------------------------------------------
# TaskLoader.by_infrastructure
# ---------------------------------------------------------------------------


class TestTaskLoaderByInfrastructure:
    """by_infrastructure() filters tasks by required infrastructure patterns."""

    def test_recursion_returns_tasks(self):
        loader = TaskLoader()
        result = loader.by_infrastructure("recursion")
        assert len(result) > 0

    def test_recursion_tasks_include_flatten(self):
        loader = TaskLoader()
        ids = {t.task_id for t in loader.by_infrastructure("recursion")}
        assert "l1_flatten_list" in ids

    def test_all_returned_tasks_require_the_pattern(self):
        loader = TaskLoader()
        for task in loader.by_infrastructure("recursion"):
            assert "recursion" in task.required_infrastructure

    def test_nonexistent_pattern_returns_empty(self):
        loader = TaskLoader()
        result = loader.by_infrastructure("__nonexistent_pattern__")
        assert result == []

    def test_error_handling_returns_tasks(self):
        loader = TaskLoader()
        result = loader.by_infrastructure("error_handling")
        assert len(result) > 0

    def test_multiple_patterns_returns_union(self):
        loader = TaskLoader()
        result_single = loader.by_infrastructure("recursion")
        result_multi = loader.by_infrastructure("recursion", "error_handling")
        # Multi-pattern search returns at least as many tasks as single
        assert len(result_multi) >= len(result_single)

    def test_returns_list(self):
        loader = TaskLoader()
        assert isinstance(loader.by_infrastructure("recursion"), list)


# ---------------------------------------------------------------------------
# TaskLoader.frontier_tasks
# ---------------------------------------------------------------------------


class TestTaskLoaderFrontierTasks:
    """frontier_tasks() returns tasks with mixed infrastructure mastery."""

    def test_empty_mastery_returns_no_frontier(self):
        loader = TaskLoader()
        # With all masteries at 0.0, nothing is mastered, so nothing is at frontier
        result = loader.frontier_tasks({})
        assert result == []

    def test_full_mastery_returns_no_frontier(self):
        loader = TaskLoader()
        # With everything mastered above threshold, nothing has unmastered patterns
        all_patterns = {p for t in loader.all_tasks for p in t.required_infrastructure}
        mastery = {p: 1.0 for p in all_patterns}
        result = loader.frontier_tasks(mastery)
        assert result == []

    def test_mixed_mastery_returns_frontier_tasks(self):
        loader = TaskLoader()
        # l0_fibonacci requires ("iteration",)
        # l1_flatten_list requires ("recursion", "type_checking", "list_ops")
        # Set recursion mastered, type_checking and list_ops unmastered
        mastery = {
            "recursion": 0.9,
            "type_checking": 0.1,
            "list_ops": 0.0,
        }
        result = loader.frontier_tasks(mastery)
        # l1_flatten_list has mastered recursion (>0.5) and unmastered others
        ids = {t.task_id for t in result}
        assert "l1_flatten_list" in ids

    def test_tasks_without_infrastructure_never_in_frontier(self):
        loader = TaskLoader()
        # Add a task with no required infrastructure
        no_infra_task = CodeTask(
            task_id="no_infra",
            level=TaskLevel.EXPRESSION,
            prompt="def f(): ...",
            test_code="",
            required_infrastructure=(),
        )
        loader.tasks["no_infra"] = no_infra_task
        mastery = {"iteration": 0.8, "recursion": 0.1}
        result = loader.frontier_tasks(mastery)
        ids = {t.task_id for t in result}
        assert "no_infra" not in ids

    def test_frontier_respects_threshold_parameter(self):
        loader = TaskLoader()
        # l0_fibonacci requires ("iteration",)
        # With iteration at 0.4, it's above threshold=0.3 (mastered) but below 0.5 (not)
        # So threshold determines whether a pattern counts as "mastered"
        # We need at least one pattern above threshold and one below
        mastery = {
            "recursion": 0.9,
            "type_checking": 0.0,
            "list_ops": 0.0,
        }
        # With threshold=0.5: recursion (0.9) > 0.5 → mastered; others ≤ 0.5 → unmastered
        result_default = loader.frontier_tasks(mastery, threshold=0.5)
        # With threshold=0.95: recursion (0.9) ≤ 0.95 → not mastered → no frontier
        result_high = loader.frontier_tasks(mastery, threshold=0.95)
        assert len(result_default) >= len(result_high)

    def test_returns_list(self):
        loader = TaskLoader()
        assert isinstance(loader.frontier_tasks({}), list)


# ---------------------------------------------------------------------------
# TaskLoader.load_from_json
# ---------------------------------------------------------------------------


class TestTaskLoaderLoadFromJson:
    """load_from_json() adds tasks from a JSON file."""

    def _make_json_tasks(self, tasks: list[dict]) -> str:
        return json.dumps(tasks)

    def test_load_single_task_increases_count(self):
        loader = TaskLoader()
        initial_count = loader.task_count
        data = [
            {
                "task_id": "json_task_001",
                "level": 0,
                "prompt": "def add(a, b): ...",
                "test_code": "_run_test('add', lambda: assert add(1,2)==3)",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        count = loader.load_from_json(path)
        assert count == 1
        assert loader.task_count == initial_count + 1

    def test_loaded_task_accessible_by_id(self):
        loader = TaskLoader()
        data = [
            {
                "task_id": "json_unique_task",
                "level": 1,
                "prompt": "def foo(): ...",
                "test_code": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        loader.load_from_json(path)
        assert "json_unique_task" in loader.tasks

    def test_loaded_task_has_correct_level(self):
        loader = TaskLoader()
        data = [
            {
                "task_id": "json_level_test",
                "level": 3,
                "prompt": "module prompt",
                "test_code": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        loader.load_from_json(path)
        task = loader.tasks["json_level_test"]
        assert task.level == TaskLevel.MODULE

    def test_loaded_task_optional_fields(self):
        loader = TaskLoader()
        data = [
            {
                "task_id": "json_full",
                "level": 0,
                "prompt": "p",
                "test_code": "t",
                "canonical_solution": "def f(): return 1",
                "required_infrastructure": ["recursion", "iteration"],
                "tags": ["math", "algo"],
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        loader.load_from_json(path)
        task = loader.tasks["json_full"]
        assert task.canonical_solution == "def f(): return 1"
        assert task.required_infrastructure == ("recursion", "iteration")
        assert task.tags == ("math", "algo")

    def test_load_multiple_tasks_returns_correct_count(self):
        loader = TaskLoader()
        data = [
            {"task_id": f"json_batch_{i}", "level": 0, "prompt": "p", "test_code": "t"}
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        count = loader.load_from_json(path)
        assert count == 5

    def test_existing_task_id_overwritten(self):
        loader = TaskLoader()
        # Overwrite l0_fibonacci with a different prompt
        data = [
            {
                "task_id": "l0_fibonacci",
                "level": 0,
                "prompt": "OVERWRITTEN PROMPT",
                "test_code": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        loader.load_from_json(path)
        assert loader.tasks["l0_fibonacci"].prompt == "OVERWRITTEN PROMPT"
        # Total count stays the same (overwrite, not add)
        assert loader.task_count == 9


# ---------------------------------------------------------------------------
# TaskLoader.level_summary
# ---------------------------------------------------------------------------


class TestTaskLoaderLevelSummary:
    """level_summary() returns a dict mapping level names to task counts."""

    def test_returns_dict(self):
        loader = TaskLoader()
        summary = loader.level_summary()
        assert isinstance(summary, dict)

    def test_all_level_names_present(self):
        loader = TaskLoader()
        summary = loader.level_summary()
        expected_keys = {"EXPRESSION", "FUNCTION", "MULTI_FUNCTION", "MODULE"}
        assert expected_keys == set(summary.keys())

    def test_expression_count_is_3(self):
        loader = TaskLoader()
        assert loader.level_summary()["EXPRESSION"] == 3

    def test_function_count_is_3(self):
        loader = TaskLoader()
        assert loader.level_summary()["FUNCTION"] == 3

    def test_multi_function_count_is_2(self):
        loader = TaskLoader()
        assert loader.level_summary()["MULTI_FUNCTION"] == 2

    def test_module_count_is_1(self):
        loader = TaskLoader()
        assert loader.level_summary()["MODULE"] == 1

    def test_counts_sum_to_total(self):
        loader = TaskLoader()
        summary = loader.level_summary()
        assert sum(summary.values()) == loader.task_count
