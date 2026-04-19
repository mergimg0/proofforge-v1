"""Lean 4 type-checker reward oracle — the CONCRETE SOS binary verifier.

This is the foundation reward: a proof either type-checks (1.0) or doesn't (0.0).
No approximation error. No learned value function. The Lean type system IS the
evaluator E in the SOS formalization.

Parallel verification via ProcessPoolExecutor for throughput on multi-core CPUs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from proofforge.rewards.base import RewardOracle, RewardResult


def _lean_check_single(statement: str, proof_text: str, timeout: int) -> dict:
    """Worker function for parallel Lean verification. Must be top-level for pickling.

    Sends the FULL proof text to Lean — multi-tactic proofs using <;> or
    multi-line tactic blocks are verified as-is. The Lean type-checker
    handles semicolons natively as tactic combinators.
    """
    cleaned = proof_text.replace("\u010a", "\n").replace("\u0120", " ").strip()

    if not cleaned:
        return {"verified": False, "error": "empty proof", "duration_ms": 0}

    # Indent each line of the proof under the statement's `by` block
    proof_lines = cleaned.split("\n")
    indented_proof = "\n".join(f"  {line}" for line in proof_lines)
    source = f"{statement}\n{indented_proof}\n"
    start = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(source)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ["lean", tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        if result.returncode == 0:
            return {"verified": True, "error": "", "duration_ms": duration_ms}
        else:
            return {
                "verified": False,
                "error": result.stderr[:500],
                "duration_ms": duration_ms,
            }
    except subprocess.TimeoutExpired:
        return {"verified": False, "error": "timeout", "duration_ms": timeout * 1000}
    except FileNotFoundError:
        return {"verified": False, "error": "lean not found on PATH", "duration_ms": 0}
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


@dataclass
class LeanOracle(RewardOracle):
    """Binary Lean 4 type-checker oracle.

    reward = 1.0 if Lean accepts, 0.0 otherwise.
    This is the exact reward that makes GRPO a concrete SOS.
    """

    timeout: int = 30
    """Per-proof verification timeout in seconds."""

    workers: int = 8
    """Number of parallel Lean verification workers."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        timeout = kwargs.get("timeout", self.timeout)
        result = _lean_check_single(statement, solution, timeout)

        return RewardResult(
            reward=1.0 if result["verified"] else 0.0,
            verified=result["verified"],
            metadata={
                "check_duration_ms": result["duration_ms"],
                "error": result["error"],
                "oracle": "lean4",
            },
        )

    def batch_evaluate(
        self, pairs: list[tuple[str, str]], **kwargs
    ) -> list[RewardResult]:
        """Parallel Lean verification across CPU cores."""
        timeout = kwargs.get("timeout", self.timeout)
        workers = kwargs.get("workers", self.workers)

        results: list[RewardResult | None] = [None] * len(pairs)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {}
            for i, (stmt, sol) in enumerate(pairs):
                future = pool.submit(_lean_check_single, stmt, sol, timeout)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"verified": False, "error": str(exc), "duration_ms": 0}

                results[idx] = RewardResult(
                    reward=1.0 if result["verified"] else 0.0,
                    verified=result["verified"],
                    metadata={
                        "check_duration_ms": result["duration_ms"],
                        "error": result["error"],
                        "oracle": "lean4",
                    },
                )

        return results  # type: ignore[return-value]
