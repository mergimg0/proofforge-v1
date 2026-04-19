"""Tests for proofforge.rewards.test_suite."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from proofforge.rewards.test_suite import (
    TestSuiteOracle,
    _check_imports_safe,
    _execute_with_tests,
)

# macOS: resource.setrlimit(RLIMIT_AS) cannot lower an already-infinite hard
# limit, so the sandbox runner crashes before any test code runs.
# Tests that spawn a real subprocess are skipped on macOS.
_SKIP_EXEC = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="resource.setrlimit(RLIMIT_AS) unsettable on macOS (RLIM_INFINITY hard limit)",
)

# Canned dicts returned by the mocked runner
_PASS_EXEC = {"passed": True, "error": "", "duration_ms": 5, "tests_run": 1, "tests_passed": 1}
_FAIL_EXEC = {"passed": False, "error": "AssertionError: always fails", "duration_ms": 5, "tests_run": 1, "tests_passed": 0}
_TIMEOUT_EXEC = {"passed": False, "error": "timeout after 2s", "duration_ms": 2000, "tests_run": 0, "tests_passed": 0}
_RUNNER = "proofforge.rewards.test_suite._execute_with_tests"


# ---------------------------------------------------------------------------
# _check_imports_safe()
# ---------------------------------------------------------------------------


class TestCheckImportsSafe:
    def test_safe_code_with_math(self):
        code = "import math\nresult = math.sqrt(4)"
        safe, reason = _check_imports_safe(code)
        assert safe is True
        assert reason == ""

    def test_safe_code_with_collections(self):
        code = "from collections import defaultdict\nd = defaultdict(int)"
        safe, reason = _check_imports_safe(code)
        assert safe is True
        assert reason == ""

    def test_safe_code_no_imports(self):
        code = "def add(a, b):\n    return a + b"
        safe, reason = _check_imports_safe(code)
        assert safe is True
        assert reason == ""

    def test_safe_code_with_itertools(self):
        code = "import itertools\nlist(itertools.chain([1], [2]))"
        safe, reason = _check_imports_safe(code)
        assert safe is True
        assert reason == ""

    def test_forbidden_import_os(self):
        code = "import os\nos.listdir('.')"
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert "Forbidden import: os" in reason

    def test_forbidden_import_sys(self):
        code = "import sys\nprint(sys.argv)"
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert "sys" in reason

    def test_forbidden_from_import_subprocess(self):
        code = "from subprocess import run\nrun(['ls'])"
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert "subprocess" in reason

    def test_forbidden_from_import_os_path(self):
        code = "from os.path import join"
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert "os" in reason

    def test_forbidden_import_socket(self):
        code = "import socket"
        safe, reason = _check_imports_safe(code)
        assert safe is False

    def test_forbidden_import_threading(self):
        code = "import threading"
        safe, reason = _check_imports_safe(code)
        assert safe is False

    def test_syntax_error_code(self):
        code = "def foo(:\n    pass"
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert "SyntaxError" in reason

    def test_syntax_error_reason_contains_detail(self):
        code = "import ("
        safe, reason = _check_imports_safe(code)
        assert safe is False
        assert reason.startswith("SyntaxError")

    def test_forbidden_import_shutil(self):
        code = "import shutil"
        safe, reason = _check_imports_safe(code)
        assert safe is False

    def test_safe_import_typing(self):
        code = "from typing import List, Optional"
        safe, reason = _check_imports_safe(code)
        assert safe is True

    def test_safe_import_dataclasses(self):
        code = "from dataclasses import dataclass, field"
        safe, reason = _check_imports_safe(code)
        assert safe is True


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SIMPLE_TEST_SUITE = """\
def _check():
    assert 1 + 1 == 2
_run_test("test_arithmetic", _check)
"""

_ALWAYS_PASS_SOLUTION = "x = 1"

_ALWAYS_FAIL_SUITE = """\
def _check():
    assert False, "always fails"
_run_test("test_fail", _check)
"""

_TIMEOUT_SOLUTION = """\
def loop():
    while True:
        pass
loop()
"""


# ---------------------------------------------------------------------------
# TestSuiteOracle
# ---------------------------------------------------------------------------


class TestTestSuiteOracleRegistration:
    def test_register_task_stores_suite(self):
        oracle = TestSuiteOracle()
        oracle.register_task("task1", _SIMPLE_TEST_SUITE)
        assert "task1" in oracle.test_suites

    def test_evaluate_registered_task_passes(self):
        oracle = TestSuiteOracle()
        oracle.register_task("task1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_PASS_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="task1")
        assert result.verified is True
        assert result.reward == pytest.approx(1.0)

    def test_evaluate_with_no_test_suite_returns_error(self):
        oracle = TestSuiteOracle()
        result = oracle.evaluate("stmt", "x = 1", task_id="nonexistent_task")
        assert result.verified is False
        assert result.reward == pytest.approx(0.0)
        assert "no test suite" in result.metadata.get("error", "")

    def test_evaluate_unregistered_uses_hash_fallback(self):
        oracle = TestSuiteOracle()
        # No task_id kwarg → uses hash(statement) → no test suite registered
        result = oracle.evaluate("some statement", "x = 1")
        assert result.verified is False


class TestTestSuiteOraclePassingCode:
    def test_passing_code_verified(self):
        oracle = TestSuiteOracle()
        suite = """\
def _check():
    assert add(2, 3) == 5
_run_test("test_add", _check)
"""
        oracle.register_task("add_task", suite)
        with patch(_RUNNER, return_value=_PASS_EXEC):
            result = oracle.evaluate("stmt", "def add(a, b): return a + b", task_id="add_task")
        assert result.verified is True
        assert result.reward == pytest.approx(1.0)

    def test_metadata_contains_oracle_key(self):
        oracle = TestSuiteOracle()
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_PASS_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="t1")
        assert result.metadata.get("oracle") == "test_suite"

    def test_metadata_contains_tests_run(self):
        oracle = TestSuiteOracle()
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_PASS_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="t1")
        assert result.metadata.get("tests_run", 0) >= 1


class TestTestSuiteOracleFailingCode:
    def test_failing_code_not_verified(self):
        oracle = TestSuiteOracle()
        oracle.register_task("t1", _ALWAYS_FAIL_SUITE)
        with patch(_RUNNER, return_value=_FAIL_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="t1")
        assert result.verified is False
        assert result.reward == pytest.approx(0.0)

    def test_failing_code_error_in_metadata(self):
        oracle = TestSuiteOracle()
        oracle.register_task("t1", _ALWAYS_FAIL_SUITE)
        with patch(_RUNNER, return_value=_FAIL_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="t1")
        assert "error" in result.metadata

    def test_assertion_error_gives_zero_reward(self):
        oracle = TestSuiteOracle()
        suite = """\
def _check():
    assert 1 == 2
_run_test("test_assert", _check)
"""
        oracle.register_task("t1", suite)
        with patch(_RUNNER, return_value=_FAIL_EXEC):
            result = oracle.evaluate("stmt", "pass", task_id="t1")
        assert result.reward == pytest.approx(0.0)


class TestTestSuiteOracleForbiddenImport:
    def test_forbidden_import_rejected(self):
        oracle = TestSuiteOracle(require_safe_imports=True)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        result = oracle.evaluate("stmt", "import os\nx=1", task_id="t1")
        assert result.verified is False
        assert result.reward == pytest.approx(0.0)
        assert result.metadata.get("rejection") == "import_check"

    def test_forbidden_import_error_in_metadata(self):
        oracle = TestSuiteOracle(require_safe_imports=True)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        result = oracle.evaluate("stmt", "import os", task_id="t1")
        assert "Forbidden import" in result.metadata.get("error", "")

    def test_safe_import_check_disabled(self):
        oracle = TestSuiteOracle(require_safe_imports=False)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_PASS_EXEC):
            result = oracle.evaluate("stmt", _ALWAYS_PASS_SOLUTION, task_id="t1")
        assert result.metadata.get("rejection") != "import_check"


class TestTestSuiteOracleTimeout:
    def test_infinite_loop_times_out(self):
        oracle = TestSuiteOracle(timeout=2)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_TIMEOUT_EXEC):
            result = oracle.evaluate("stmt", _TIMEOUT_SOLUTION, task_id="t1")
        assert result.verified is False
        assert result.reward == pytest.approx(0.0)
        assert "timeout" in result.metadata.get("error", "").lower()

    def test_timeout_metadata_present(self):
        oracle = TestSuiteOracle(timeout=2)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        with patch(_RUNNER, return_value=_TIMEOUT_EXEC):
            result = oracle.evaluate("stmt", _TIMEOUT_SOLUTION, task_id="t1")
        assert "duration_ms" in result.metadata

    @_SKIP_EXEC
    def test_real_infinite_loop_times_out(self):
        """Real subprocess timeout — only runs on non-macOS platforms."""
        oracle = TestSuiteOracle(timeout=2)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        result = oracle.evaluate("stmt", _TIMEOUT_SOLUTION, task_id="t1")
        assert result.verified is False
        assert "timeout" in result.metadata.get("error", "").lower()


class TestTestSuiteOracleBatchEvaluate:
    def test_batch_missing_task_id_gives_error(self):
        oracle = TestSuiteOracle()
        pairs = [("stmt", "x=1")]
        results = oracle.batch_evaluate(pairs, task_ids=["no_such_task"])
        assert len(results) == 1
        assert results[0].verified is False
        assert "no test suite" in results[0].metadata.get("error", "")

    def test_batch_empty_input(self):
        oracle = TestSuiteOracle()
        results = oracle.batch_evaluate([], task_ids=[])
        assert results == []

    def test_batch_import_rejection_without_execution(self):
        # Import check fires before any subprocess — no mock needed.
        oracle = TestSuiteOracle(require_safe_imports=True)
        oracle.register_task("t1", _SIMPLE_TEST_SUITE)
        pairs = [("stmt", "import os\nx=1")]
        results = oracle.batch_evaluate(pairs, task_ids=["t1"])
        assert results[0].verified is False
        assert results[0].metadata.get("rejection") == "import_check"

    def test_batch_returns_correct_length(self):
        # Only the pre-execution paths are exercised here (no subprocess needed):
        # two tasks with missing IDs → two error results.
        oracle = TestSuiteOracle()
        pairs = [("s1", "x=1"), ("s2", "x=2"), ("s3", "x=3")]
        results = oracle.batch_evaluate(pairs, task_ids=["missing1", "missing2", "missing3"])
        assert len(results) == 3

    @_SKIP_EXEC
    def test_batch_all_pass_real(self):
        """Real subprocess execution — skipped on macOS."""
        oracle = TestSuiteOracle()
        suite = """\
def _check():
    pass
_run_test("t", _check)
"""
        oracle.register_task("t1", suite)
        oracle.register_task("t2", suite)
        pairs = [("s1", _ALWAYS_PASS_SOLUTION), ("s2", _ALWAYS_PASS_SOLUTION)]
        results = oracle.batch_evaluate(pairs, task_ids=["t1", "t2"])
        assert all(r.verified for r in results)

    @_SKIP_EXEC
    def test_batch_mixed_results_real(self):
        """Real subprocess execution — skipped on macOS."""
        oracle = TestSuiteOracle()
        pass_suite = """\
def _check():
    assert add(1, 1) == 2
_run_test("t", _check)
"""
        oracle.register_task("pass_task", pass_suite)
        oracle.register_task("fail_task", _ALWAYS_FAIL_SUITE)
        pairs = [
            ("stmt1", "def add(a, b): return a + b"),
            ("stmt2", _ALWAYS_PASS_SOLUTION),
        ]
        results = oracle.batch_evaluate(pairs, task_ids=["pass_task", "fail_task"])
        assert len(results) == 2
        assert results[0].verified is True
        assert results[1].verified is False
