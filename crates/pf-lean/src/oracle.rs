//! Binary reward oracle: bridges the Lean 4 type checker to the GRPO reward signal.
//!
//! This is the critical component that makes ProofForge work:
//! - Lean type-checks → Verified → reward = 1.0
//! - Lean rejects → Failed → reward = 0.0
//!
//! Binary. Deterministic. No hallucination surface.

use pf_core::reward::RewardOracle;
use pf_core::types::{ProofAttempt, ProofResult};

use crate::checker::{LeanChecker, LeanCheckerConfig};

/// The Lean 4 reward oracle: wraps LeanChecker to implement RewardOracle.
pub struct LeanRewardOracle {
    checker: LeanChecker,
    /// Default imports for proof checking (e.g., Mathlib modules)
    default_imports: Vec<String>,
}

impl LeanRewardOracle {
    pub fn new(config: LeanCheckerConfig) -> Self {
        Self {
            checker: LeanChecker::new(config),
            default_imports: vec!["Mathlib".to_string()],
        }
    }

    pub fn with_imports(mut self, imports: Vec<String>) -> Self {
        self.default_imports = imports;
        self
    }
}

impl RewardOracle for LeanRewardOracle {
    fn check(&self, attempt: &ProofAttempt) -> ProofResult {
        let imports: Vec<&str> = self.default_imports.iter().map(|s| s.as_str()).collect();

        match self
            .checker
            .check_proof(&attempt.statement, &attempt.proof, &imports)
        {
            Ok(result) => {
                if result.success {
                    ProofResult::Verified {
                        check_duration_ms: result.duration.as_millis() as u64,
                    }
                } else {
                    ProofResult::Failed {
                        error: result.errors.join("\n"),
                        check_duration_ms: result.duration.as_millis() as u64,
                    }
                }
            }
            Err(e) => ProofResult::Failed {
                error: format!("Checker error: {e}"),
                check_duration_ms: 0,
            },
        }
    }
}
