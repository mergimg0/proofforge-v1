//! # pf-execution: Deterministic execution layer
//!
//! Translates Lean 4 verified theorems to deterministic actions.
//! This is NOT an intelligence. It's pure translation mapping.
//! No LLM. No hallucination surface.
//!
//! From Ollie: "The execution bot is not a bot. It's not an intelligence.
//! It's just execution."

use pf_core::types::ProofResult;
use serde::{Deserialize, Serialize};

/// A verified theorem reference: proof that a strategy satisfies
/// all backtester invariants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedTheorem {
    /// The Lean 4 theorem statement
    pub statement: String,
    /// Hash of the verified proof
    pub proof_hash: String,
    /// Which invariants were proven
    pub invariants: Vec<String>,
    /// The proof result (must be Verified)
    pub verification: ProofResult,
}

/// A verified action: a trade action with its mathematical proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifiedAction {
    /// Enter a position, backed by a verified theorem
    Enter {
        symbol: String,
        size: f64,
        theorem: VerifiedTheorem,
    },
    /// Exit a position, backed by a verified theorem
    Exit {
        symbol: String,
        theorem: VerifiedTheorem,
    },
    /// Hold current position, backed by a verified theorem
    Hold {
        theorem: VerifiedTheorem,
    },
}

impl VerifiedAction {
    /// Every action must have a verified theorem backing it.
    pub fn theorem(&self) -> &VerifiedTheorem {
        match self {
            VerifiedAction::Enter { theorem, .. } => theorem,
            VerifiedAction::Exit { theorem, .. } => theorem,
            VerifiedAction::Hold { theorem } => theorem,
        }
    }

    /// Check that the backing theorem is actually verified.
    pub fn is_valid(&self) -> bool {
        self.theorem().verification.is_verified()
    }
}

/// The execution translator: maps verified theorems to actions.
///
/// Pure deterministic translation. The theorem's proof guarantees
/// the action satisfies all backtester invariants. This function
/// just maps the proven strategy to an executable action.
pub struct ExecutionTranslator;

impl ExecutionTranslator {
    pub fn new() -> Self {
        Self
    }

    /// Translate a verified theorem to a deterministic action.
    ///
    /// This is pure mapping -- no inference, no approximation.
    /// The theorem already proves the strategy is sound;
    /// we just extract the action parameters.
    pub fn translate(&self, theorem: VerifiedTheorem) -> Option<VerifiedAction> {
        if !theorem.verification.is_verified() {
            return None; // Never execute an unverified strategy
        }

        // Pattern match on the theorem structure to extract action
        // In production: parse the Lean 4 theorem to determine
        // signal direction, position sizing, timing
        Some(VerifiedAction::Hold { theorem })
    }
}

impl Default for ExecutionTranslator {
    fn default() -> Self {
        Self::new()
    }
}
