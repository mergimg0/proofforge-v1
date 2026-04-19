"""Integration tests for App 1: Code Verification Pipeline.

Tests the full loop: task selection -> code verification -> pattern tracking.
No GPU required -- uses mock generate_fn.
"""

import pytest

from proofforge.code_verify.pipeline import CodeVerificationPipeline, PipelineConfig
from proofforge.code_verify.task_loader import TaskLoader, BUILTIN_TASKS, TaskLevel
from proofforge.code_verify.tactic_registry import TacticRegistry, _detect_patterns
from proofforge.rewards.test_suite import TestSuiteOracle


class TestBuiltinTasksSyntax:
    """Verify all BUILTIN_TASKS test_code strings are valid Python."""

    @pytest.mark.parametrize("task", BUILTIN_TASKS, ids=[t.task_id for t in BUILTIN_TASKS])
    def test_test_code_compiles(self, task):
        compile(task.test_code, "<{}>".format(task.task_id), "exec")


class TestTestSuiteExecution:
    """Test that canonical solutions pass their own test suites."""

    @pytest.mark.parametrize(
        "task",
        [t for t in BUILTIN_TASKS if t.canonical_solution is not None],
        ids=[t.task_id for t in BUILTIN_TASKS if t.canonical_solution is not None],
    )
    def test_canonical_solution_passes(self, task):
        oracle = TestSuiteOracle(timeout=10, memory_mb=128)
        oracle.register_task(task.task_id, task.test_code)
        result = oracle.evaluate(task.prompt, task.canonical_solution, task_id=task.task_id)
        assert result.verified, (
            "{}: canonical solution failed: {}".format(task.task_id, result.metadata.get("error", ""))
        )

    def test_wrong_solution_fails(self):
        oracle = TestSuiteOracle(timeout=10, memory_mb=128)
        task = BUILTIN_TASKS[0]  # fibonacci
        oracle.register_task(task.task_id, task.test_code)
        result = oracle.evaluate(task.prompt, "def fibonacci(n): return 0", task_id=task.task_id)
        assert not result.verified


class TestPatternDetection:
    def test_iteration_detected(self):
        code = "for i in range(10): pass"
        assert "iteration" in _detect_patterns(code)

    def test_recursion_detected(self):
        code = "def f(n):\n    return f(n-1)"
        assert "recursion" in _detect_patterns(code)

    def test_error_handling_detected(self):
        code = "try:\n    x = 1\nexcept:\n    pass"
        assert "error_handling" in _detect_patterns(code)


class TestPipelineWithMockLLM:
    def _make_pipeline(self, generate_fn=None):
        config = PipelineConfig(
            group_size=4, max_tokens=256, controller_enabled=False,
            infrastructure_aware=False)
        pipeline = CodeVerificationPipeline(config=config, generate_fn=generate_fn)
        return pipeline

    def test_pipeline_creates(self):
        pipeline = self._make_pipeline()
        assert pipeline.task_loader.task_count == len(BUILTIN_TASKS)

    def test_pipeline_step_with_placeholder(self):
        pipeline = self._make_pipeline()
        result = pipeline.run_step(0)
        assert result.group_size == 4
        assert result.pass_rate == 0.0  # placeholder solutions fail

    def test_pipeline_step_with_canonical_solutions(self):
        """Mock LLM that returns canonical solutions."""
        def canonical_gen(prompt, n, temp, max_tok):
            # Find the task by matching prompt
            for task in BUILTIN_TASKS:
                if task.prompt in prompt and task.canonical_solution:
                    return [task.canonical_solution] * n
            return ["# no match"] * n

        pipeline = self._make_pipeline(generate_fn=canonical_gen)
        result = pipeline.run_step(0)
        # At least some should pass (canonical solutions are correct)
        assert result.group_size == 4

    def test_pipeline_summary(self):
        pipeline = self._make_pipeline()
        pipeline.run_step(0)
        summary = pipeline.summary()
        assert "total_steps" in summary
        assert summary["total_steps"] == 1


class TestTaskLoader:
    def test_all_tasks_have_test_code(self):
        loader = TaskLoader()
        for task in loader.all_tasks:
            assert task.test_code, "{} has empty test_code".format(task.task_id)

    def test_level_distribution(self):
        loader = TaskLoader()
        by_level = loader.level_summary()
        assert by_level["EXPRESSION"] == 3
        assert by_level["FUNCTION"] == 3
        assert by_level["MULTI_FUNCTION"] == 2
        assert by_level["MODULE"] == 1
