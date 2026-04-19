//! Reward oracle trait: the interface between ProofForge and the Lean 4 type checker.
//!
//! The reward oracle is BINARY: a proof either type-checks (reward = 1.0)
//! or it doesn't (reward = 0.0). This is what makes GRPO on proof generation
//! a CONCRETE SOS rather than an axiomatised one — there is no learned value
//! function introducing approximation error.

use crate::types::{ProofAttempt, ProofResult};

/// The reward oracle trait. Implementations provide the binary
/// proof-checking capability.
///
/// In production, this wraps the Lean 4 type checker (pf-lean crate).
/// In testing, this can be a mock that checks simple properties.
pub trait RewardOracle: Send + Sync {
    /// Check a proof attempt against the Lean 4 type checker.
    ///
    /// Returns Verified (reward 1.0) or Failed (reward 0.0).
    /// This is the binary reward oracle from the ProofForge paper.
    fn check(&self, attempt: &ProofAttempt) -> ProofResult;
}

/// A mock oracle for testing: accepts proofs containing "trivial" or "rfl".
pub struct MockOracle;

impl RewardOracle for MockOracle {
    fn check(&self, attempt: &ProofAttempt) -> ProofResult {
        let proof_lower = attempt.proof.to_lowercase();
        if proof_lower.contains("trivial")
            || proof_lower.contains("rfl")
            || proof_lower.contains("simp")
        {
            ProofResult::Verified {
                check_duration_ms: 1,
            }
        } else {
            ProofResult::Failed {
                error: "mock: proof does not contain a recognized tactic".to_string(),
                check_duration_ms: 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    #[test]
    fn test_mock_oracle() {
        let oracle = MockOracle;

        let good = ProofAttempt {
            statement: "theorem t : True := by".to_string(),
            proof: "trivial".to_string(),
            agent_id: AgentId::Tactic,
            policy_version: 1,
        };
        assert!(oracle.check(&good).is_verified());

        let bad = ProofAttempt {
            statement: "theorem t : 1 + 1 = 3 := by".to_string(),
            proof: "sorry_not_sorry".to_string(),
            agent_id: AgentId::Algebra,
            policy_version: 1,
        };
        assert!(!oracle.check(&bad).is_verified());
    }
}
