//! # pf-agents: Multi-agent proof search ensemble
//!
//! Specialist agents for different proof strategies:
//! - Induction: structural/strong induction
//! - Algebra: ring/field rewriting, simp lemmas
//! - Tactic: tactic combination search (omega, linarith, norm_num)
//! - Analogy: find similar proofs in Mathlib and adapt
//!
//! Diversity in the ensemble feeds the variance-acceleration mechanism
//! from Section 9 of the SOS paper: high Var[E(omega)] provides extra
//! descent c * Var[E] beyond the mean improvement.

pub mod ensemble;
pub mod llm;
pub mod adaptive;

use pf_core::types::{AgentId, ProofAttempt};

/// Trait for a proof search specialist agent.
pub trait ProofAgent: Send + Sync {
    /// The agent's identifier.
    fn id(&self) -> AgentId;

    /// Generate proof attempts for a theorem statement.
    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt>;

    /// Downcast to AdaptiveAgent for memory updates. Default: None (stateless agents).
    fn as_adaptive_mut(&mut self) -> Option<&mut crate::adaptive::AdaptiveAgent> { None }
}

/// Induction specialist: generates proofs using structural and strong induction.
pub struct InductionAgent;

impl ProofAgent for InductionAgent {
    fn id(&self) -> AgentId {
        AgentId::Induction
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        // Placeholder: in production, this calls the LLM with induction-focused prompts
        (0..count)
            .map(|i| ProofAttempt {
                statement: statement.to_string(),
                proof: format!("-- induction attempt {i}\ninduction n with\n| zero => simp\n| succ n ih => simp [ih]"),
                agent_id: self.id(),
                policy_version,
            })
            .collect()
    }
}

/// Algebra specialist: generates proofs using ring/field rewriting.
pub struct AlgebraAgent;

impl ProofAgent for AlgebraAgent {
    fn id(&self) -> AgentId {
        AgentId::Algebra
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        (0..count)
            .map(|i| ProofAttempt {
                statement: statement.to_string(),
                proof: format!("-- algebra attempt {i}\nring"),
                agent_id: self.id(),
                policy_version,
            })
            .collect()
    }
}

/// Tactic specialist: generates proofs using tactic combination search.
pub struct TacticAgent;

impl ProofAgent for TacticAgent {
    fn id(&self) -> AgentId {
        AgentId::Tactic
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        let tactics = ["simp", "omega", "linarith", "norm_num", "trivial", "rfl"];
        (0..count)
            .map(|i| ProofAttempt {
                statement: statement.to_string(),
                proof: format!("-- tactic attempt {i}\n{}", tactics[i % tactics.len()]),
                agent_id: self.id(),
                policy_version,
            })
            .collect()
    }
}

/// Analogy specialist: finds similar proofs in Mathlib and adapts them.
pub struct AnalogyAgent;

impl ProofAgent for AnalogyAgent {
    fn id(&self) -> AgentId {
        AgentId::Analogy
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        // Placeholder: in production, this searches Mathlib for similar theorems
        (0..count)
            .map(|i| ProofAttempt {
                statement: statement.to_string(),
                proof: format!("-- analogy attempt {i}\nexact?"),
                agent_id: self.id(),
                policy_version,
            })
            .collect()
    }
}
