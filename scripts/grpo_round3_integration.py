"""ProofForge Round 3 Integration: App 6 (Efficiency Reward) + App 8 (Controller).

Drop-in integration module for grpo_round3.py. Import this module and call
the integration functions to wire efficiency reward and adaptive controller
into the existing training loop.

Usage in grpo_round3.py:
    from grpo_round3_integration import (
        create_efficiency_oracle,
        create_controller,
        compute_efficiency_rewards,
        controller_step,
        compute_retention_rate,
        count_distinct_tactics,
    )

    # At init:
    eff_oracle = create_efficiency_oracle()
    controller = create_controller(output_dir)

    # In the training loop, after lean_verify_batch:
    raw_rewards = [1.0 if v else 0.0 for v in verification_results]
    eff_rewards = compute_efficiency_rewards(
        eff_oracle, thm, completions, verification_results, raw_rewards, step, pass_rate
    )
    # Use eff_rewards instead of raw_rewards for unlikeliness computation

    # After metrics:
    interventions = controller_step(
        controller, step, mean_reward, pass_rate,
        loss=loss_val, proof_length=mean_proof_len,
        retention_rate=retention, tactic_diversity=n_distinct_tactics
    )
"""

from __future__ import annotations

import re
from typing import Optional

# These imports work when proofforge is installed (pip install -e python/)
from proofforge.rewards.efficiency import (
    EfficiencyReward,
    EfficiencyConfig,
)
from proofforge.rewards.base import RewardResult
from proofforge.controller.controller import AdaptiveController, ControllerConfig
from proofforge.controller.interventions import Intervention, InterventionType


# ---------------------------------------------------------------------------
# Lean 4 tactic extraction (for diversity tracking)
# ---------------------------------------------------------------------------

LEAN_FIRST_TACTIC_RE = re.compile(
    r"^\s*(simp|rfl|exact|apply|intro|intros|cases|induction|rw|rewrite|have|let"
    r"|show|calc|omega|linarith|norm_num|ring|field_simp|push_neg|contrapose"
    r"|by_contra|trivial|decide|native_decide|aesop|tauto|constructor|ext"
    r"|funext|congr|refine|use|exists)\b",
    re.MULTILINE,
)


def extract_first_tactic(proof_text: str) -> str:
    """Extract the first tactic name from a proof."""
    cleaned = proof_text.replace("\u010a", "\n").replace("\u0120", " ").strip()
    m = LEAN_FIRST_TACTIC_RE.search(cleaned)
    return m.group(1) if m else "unknown"


def count_distinct_tactics(completions: list, verification_results: list) -> int:
    """Count distinct first-tactics used in successful proofs.

    Returns the number of unique tactics. Low count (e.g., 1-2) signals
    simp monoculture.
    """
    tactics = set()
    for comp, verified in zip(completions, verification_results):
        if verified:
            tactic = extract_first_tactic(comp.get("text", ""))
            tactics.add(tactic)
    return len(tactics) if tactics else 0


# ---------------------------------------------------------------------------
# Retention rate computation
# ---------------------------------------------------------------------------

class RetentionTracker:
    """Track which theorems were solved at previous checkpoint.

    Retention = fraction of previously-solved theorems still solved.
    The 9.1% → 95.2% retention jump is the consolidation signal.
    """

    def __init__(self):
        self._prev_solved: set[str] = set()
        self._curr_solved: set[str] = set()

    def record(self, theorem_id: str, solved: bool) -> None:
        if solved:
            self._curr_solved.add(theorem_id)

    def compute_and_rotate(self) -> float:
        """Compute retention rate and rotate: current becomes previous."""
        if not self._prev_solved:
            rate = 0.0
        else:
            retained = self._prev_solved & self._curr_solved
            rate = len(retained) / len(self._prev_solved)

        self._prev_solved = self._curr_solved.copy()
        self._curr_solved = set()
        return rate


# ---------------------------------------------------------------------------
# Efficiency Oracle (App 6)
# ---------------------------------------------------------------------------

class _MockBinaryOracle:
    """Wraps the lean_verify_batch results as a RewardOracle."""

    def evaluate(self, statement: str, solution: str, **kwargs) -> RewardResult:
        verified = kwargs.get("_verified", False)
        return RewardResult(
            reward=1.0 if verified else 0.0,
            verified=verified,
            metadata={"oracle": "lean4_round3"},
        )

    def batch_evaluate(self, pairs, **kwargs):
        return [self.evaluate(s, p, **kwargs) for s, p in pairs]


def create_efficiency_oracle(
    alpha_mastered: float = 0.5,
    alpha_moderate: float = 0.2,
    phase_in_step: int = 50,
    phase_in_pass_rate: float = 0.20,
    max_proof_tokens: int = 256,
) -> EfficiencyReward:
    """Create an efficiency reward oracle for Round 3.

    Wraps Lean verification with adaptive per-theorem efficiency weighting.
    During bootstrap (before phase_in_step), behaves as pure binary.
    """
    return EfficiencyReward(
        base_oracle=_MockBinaryOracle(),
        config=EfficiencyConfig(
            alpha_mastered=alpha_mastered,
            alpha_moderate=alpha_moderate,
            alpha_fragile=0.0,
            max_proof_tokens=max_proof_tokens,
            phase_in_step=phase_in_step,
            phase_in_pass_rate=phase_in_pass_rate,
            length_floor=0.3,
        ),
    )


def compute_efficiency_rewards(
    oracle: EfficiencyReward,
    theorem: dict,
    completions: list,
    verification_results: list,
    raw_rewards: list,
    step: int,
    pass_rate: float,
    temperature: float = 0.7,
) -> list[float]:
    """Compute efficiency-weighted rewards for a GRPO group.

    Replaces raw binary rewards with efficiency-weighted rewards
    for mastered theorems. Before phase-in, returns raw rewards unchanged.

    Args:
        oracle: The EfficiencyReward oracle
        theorem: Theorem dict with 'id' and 'statement'
        completions: List of completion dicts with 'text' and 'token_ids'
        verification_results: List of bool from lean_verify_batch
        raw_rewards: List of float (0.0 or 1.0)
        step: Current training step
        pass_rate: Current pass rate
        temperature: Sampling temperature used

    Returns:
        List of float rewards (efficiency-weighted for mastered theorems)
    """
    oracle.update_training_state(step, pass_rate)

    thm_id = theorem.get("id", str(hash(theorem["statement"])))

    efficiency_rewards = []
    for comp, verified, raw_r in zip(completions, verification_results, raw_rewards):
        proof_text = comp.get("text", "")
        proof_tokens = len(comp.get("token_ids", [])) or max(1, len(proof_text) // 4)

        result = oracle.evaluate(
            theorem["statement"],
            proof_text,
            theorem_id=thm_id,
            temperature=temperature,
            proof_tokens=proof_tokens,
            _verified=verified,
        )
        efficiency_rewards.append(result.reward)

    return efficiency_rewards


# ---------------------------------------------------------------------------
# Adaptive Controller (App 8)
# ---------------------------------------------------------------------------

def create_controller(
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> AdaptiveController:
    """Create an adaptive training controller for Round 3."""
    return AdaptiveController(
        config=ControllerConfig(
            log_dir=output_dir,
            verbose=verbose,
            auto_apply_checkpoints=True,
            efficiency_reward_enabled=True,
            expanding_ring_enabled=True,
        )
    )


def controller_step(
    controller: AdaptiveController,
    step: int,
    reward: float,
    pass_rate: Optional[float] = None,
    loss: Optional[float] = None,
    proof_length: Optional[float] = None,
    retention_rate: Optional[float] = None,
    tactic_diversity: Optional[float] = None,
) -> list[Intervention]:
    """Feed one training step to the controller.

    Returns list of recommended interventions (possibly empty).
    Critical interventions should be acted on by the training loop.
    """
    return controller.update(
        step=step,
        reward=reward,
        pass_rate=pass_rate,
        loss=loss,
        proof_length=proof_length,
        retention_rate=retention_rate,
        tactic_diversity=tactic_diversity,
    )


def should_switch_to_efficiency(interventions: list[Intervention]) -> bool:
    """Check if the controller recommends switching to efficiency reward."""
    return any(i.type == InterventionType.SWITCH_EFFICIENCY for i in interventions)


def should_expand_dataset(interventions: list[Intervention]) -> bool:
    """Check if the controller recommends expanding the theorem set."""
    return any(i.type == InterventionType.EXPAND_DATASET for i in interventions)


def tactic_diversity_warning(interventions: list[Intervention]) -> Optional[str]:
    """Check if there's a tactic monoculture warning."""
    for i in interventions:
        if i.type == InterventionType.DIVERSIFY_TACTICS:
            return i.reason
    return None
