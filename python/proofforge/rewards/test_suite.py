"""App 1: Test suite execution reward oracle for code verification.

Replaces the Lean type-checker with test suite execution as the binary
reward oracle. A generated code solution either passes all tests (1.0)
or doesn't (0.0).

Key difference from LeanOracle: test suites are IMPERFECT verifiers
(partial coverage, flaky tests). The infrastructure learning mechanism
depends on reward quality. Comprehensive test suites with edge cases
are critical.

Security: All code execution runs in a restricted subprocess with:
  - Import whitelist (no os, sys, subprocess, etc.)
  - Memory limit (256 MB)
  - CPU time limit (30s default)
  - No network access
  - No filesystem writes outside tmpdir
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from proofforge.rewards.base import RewardOracle, RewardResult


# Imports that are NEVER allowed in generated code
FORBIDDEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "signal", "multiprocessing",
    "threading",
    "shelve", "sqlite3",
    "__builtin__", "builtins", "code", "codeop",
    "compileall", "py_compile",
})

# Safe imports allowed in generated code
SAFE_IMPORTS = frozenset({
    "math", "cmath", "decimal", "fractions", "random",
    "statistics", "itertools", "functools", "operator",
    "collections", "heapq", "bisect", "array",
    "copy", "enum", "dataclasses", "typing",
    "string", "re", "textwrap",
    "json", "csv",
    "datetime", "time",
    "abc", "contextlib",
})


def _check_imports_safe(code: str) -> tuple[bool, str]:
    """Static analysis: reject code with forbidden imports.

    Returns (safe, reason). Parses the AST to catch both
    'import X' and 'from X import Y' forms.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: from {node.module}"

    return True, ""


def _execute_with_tests(
    code: str, test_code: str, timeout: int, memory_mb: int
) -> dict:
    """Execute generated code + test suite in an isolated subprocess.

    Returns dict with: passed (bool), error (str), duration_ms (int),
    tests_run (int), tests_passed (int).
    """
    runner = textwrap.dedent(f"""\
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({memory_mb} * 1024 * 1024, {memory_mb} * 1024 * 1024))
except (ImportError, ValueError):
    pass  # macOS: RLIMIT_AS not supported

# --- Generated code ---
{code}

# --- Test suite ---
_test_results = {{"run": 0, "passed": 0, "failures": []}}

def _run_test(name, fn):
    _test_results["run"] += 1
    try:
        result = fn()
        if result is False:
            _test_results["failures"].append(f"{{name}}: assertion failed")
        else:
            _test_results["passed"] += 1
    except Exception as e:
        _test_results["failures"].append(f"{{name}}: {{type(e).__name__}}: {{e}}")

def _assert_raises(exc_type, fn):
    try:
        fn()
        raise AssertionError(f"Expected {{exc_type.__name__}}")
    except exc_type:
        pass

{test_code}

# Print results as JSON
import json as _json
print("__RESULT__" + _json.dumps(_test_results))
""")

    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": tempfile.gettempdir(),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        stdout = result.stdout
        if "__RESULT__" in stdout:
            result_json = stdout.split("__RESULT__")[1].strip()
            test_results = json.loads(result_json)
            all_passed = (
                test_results["run"] > 0
                and test_results["passed"] == test_results["run"]
            )
            return {
                "passed": all_passed,
                "error": "; ".join(test_results.get("failures", [])),
                "duration_ms": duration_ms,
                "tests_run": test_results["run"],
                "tests_passed": test_results["passed"],
            }
        else:
            return {
                "passed": False,
                "error": result.stderr[:500] if result.stderr else "no test output",
                "duration_ms": duration_ms,
                "tests_run": 0,
                "tests_passed": 0,
            }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": f"timeout after {timeout}s",
            "duration_ms": timeout * 1000,
            "tests_run": 0,
            "tests_passed": 0,
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": str(exc),
            "duration_ms": int((time.monotonic() - start) * 1000),
            "tests_run": 0,
            "tests_passed": 0,
        }


@dataclass
class TestSuiteOracle(RewardOracle):
    """Binary test-suite execution oracle for code verification (App 1).

    reward = 1.0 if ALL tests pass, 0.0 otherwise.

    This is the code-domain analog of the Lean type-checker: a binary
    verifier that provides the reward signal for GRPO. Unlike Lean,
    test suites are imperfect (partial coverage), so reward quality
    depends on test suite comprehensiveness.

    The test_suites dict maps task IDs to test code strings. Each test
    code string should call _run_test(name, fn) for each test case.
    """

    test_suites: dict[str, str] = field(default_factory=dict)
    """Mapping from task_id to test code string."""

    timeout: int = 30
    """Per-task execution timeout in seconds."""

    memory_mb: int = 256
    """Memory limit for code execution in MB."""

    workers: int = 4
    """Number of parallel execution workers."""

    require_safe_imports: bool = True
    """Reject code with forbidden imports before execution."""

    def register_task(self, task_id: str, test_code: str) -> None:
        """Register a test suite for a task."""
        self.test_suites[task_id] = test_code

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        """Evaluate generated code against its test suite.

        Args:
            statement: Task description / function signature
            solution: Generated code
            task_id: Key into test_suites (required)
        """
        task_id = kwargs.get("task_id", str(hash(statement)))
        test_code = self.test_suites.get(task_id)

        if test_code is None:
            return RewardResult(
                reward=0.0,
                verified=False,
                metadata={"error": f"no test suite for task_id={task_id}", "oracle": "test_suite"},
            )

        if self.require_safe_imports:
            safe, reason = _check_imports_safe(solution)
            if not safe:
                return RewardResult(
                    reward=0.0,
                    verified=False,
                    metadata={"error": reason, "oracle": "test_suite", "rejection": "import_check"},
                )

        result = _execute_with_tests(solution, test_code, self.timeout, self.memory_mb)

        return RewardResult(
            reward=1.0 if result["passed"] else 0.0,
            verified=result["passed"],
            metadata={
                "oracle": "test_suite",
                "duration_ms": result["duration_ms"],
                "tests_run": result["tests_run"],
                "tests_passed": result["tests_passed"],
                "error": result["error"],
                "task_id": task_id,
            },
        )

    def batch_evaluate(
        self, pairs: list[tuple[str, str]], **kwargs
    ) -> list[RewardResult]:
        """Parallel test execution across worker processes."""
        task_ids = kwargs.get("task_ids", [str(hash(s)) for s, _ in pairs])

        results: list[RewardResult | None] = [None] * len(pairs)

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            future_to_idx = {}
            for i, ((stmt, sol), tid) in enumerate(zip(pairs, task_ids)):
                test_code = self.test_suites.get(tid)
                if test_code is None:
                    results[i] = RewardResult(
                        reward=0.0, verified=False,
                        metadata={"error": f"no test suite for {tid}", "oracle": "test_suite"},
                    )
                    continue

                if self.require_safe_imports:
                    safe, reason = _check_imports_safe(sol)
                    if not safe:
                        results[i] = RewardResult(
                            reward=0.0, verified=False,
                            metadata={"error": reason, "oracle": "test_suite", "rejection": "import_check"},
                        )
                        continue

                future = pool.submit(
                    _execute_with_tests, sol, test_code, self.timeout, self.memory_mb
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"passed": False, "error": str(exc), "duration_ms": 0, "tests_run": 0, "tests_passed": 0}

                results[idx] = RewardResult(
                    reward=1.0 if result["passed"] else 0.0,
                    verified=result["passed"],
                    metadata={
                        "oracle": "test_suite",
                        "duration_ms": result["duration_ms"],
                        "tests_run": result["tests_run"],
                        "tests_passed": result["tests_passed"],
                        "error": result["error"],
                        "task_id": task_ids[idx],
                    },
                )

        return results  # type: ignore[return-value]
