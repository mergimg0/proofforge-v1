"""Shared fixtures for ProofForge test suite."""

import pytest
from proofforge.rewards.base import RewardOracle, RewardResult


class MockBinaryOracle(RewardOracle):
    """Mock oracle: accepts proofs containing 'rfl', 'trivial', or 'simp'."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        verified = kwargs.get("_verified", None)
        if verified is None:
            text = solution.lower()
            verified = any(t in text for t in ("rfl", "trivial", "simp"))
        return RewardResult(
            reward=1.0 if verified else 0.0,
            verified=verified,
            metadata={"oracle": "mock", "check_duration_ms": 1},
        )


@pytest.fixture
def mock_oracle():
    return MockBinaryOracle()
