//! Multi-agent ensemble: aggregates diverse proof strategies.
//!
//! Diversity is not a cost -- it's acceleration fuel (Section 9, SOS paper).
//! High evaluator variance (diverse specialists) corresponds to high
//! spectral discordance, and the variance-controlled rate provides
//! additional descent c * Var[E] beyond the mean improvement.

use pf_core::types::ProofAttempt;

use crate::ProofAgent;

/// The proof ensemble: multiple specialist agents contributing diverse proofs.
pub struct ProofEnsemble {
    agents: Vec<Box<dyn ProofAgent>>,
}

impl ProofEnsemble {
    pub fn new(agents: Vec<Box<dyn ProofAgent>>) -> Self {
        Self { agents }
    }

    /// Generate proof attempts from all agents.
    ///
    /// Distributes the budget across agents. Diversity in the ensemble
    /// feeds variance acceleration: different agents produce structurally
    /// different proofs, increasing Var[E(omega)] and thus the extra
    /// descent term c * Var[E].
    pub fn generate(
        &self,
        statement: &str,
        total_count: usize,
        policy_version: u64,
    ) -> Vec<ProofAttempt> {
        if self.agents.is_empty() {
            return vec![];
        }

        let per_agent = total_count / self.agents.len();
        let remainder = total_count % self.agents.len();

        let mut attempts = Vec::with_capacity(total_count);
        for (i, agent) in self.agents.iter().enumerate() {
            let count = per_agent + if i < remainder { 1 } else { 0 };
            attempts.extend(agent.generate(statement, count, policy_version));
        }
        attempts
    }

    /// Number of specialist agents in the ensemble.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

/// Create the default ensemble with all four specialist agents.
pub fn default_ensemble() -> ProofEnsemble {
    ProofEnsemble::new(vec![
        Box::new(crate::InductionAgent),
        Box::new(crate::AlgebraAgent),
        Box::new(crate::TacticAgent),
        Box::new(crate::AnalogyAgent),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_distributes_budget() {
        let ensemble = default_ensemble();
        let attempts = ensemble.generate("theorem t : True := by", 16, 1);

        assert_eq!(attempts.len(), 16);

        // Each agent should have generated 4 attempts
        use pf_core::types::AgentId;
        let induction_count = attempts.iter().filter(|a| a.agent_id == AgentId::Induction).count();
        let algebra_count = attempts.iter().filter(|a| a.agent_id == AgentId::Algebra).count();
        let tactic_count = attempts.iter().filter(|a| a.agent_id == AgentId::Tactic).count();
        let analogy_count = attempts.iter().filter(|a| a.agent_id == AgentId::Analogy).count();

        assert_eq!(induction_count, 4);
        assert_eq!(algebra_count, 4);
        assert_eq!(tactic_count, 4);
        assert_eq!(analogy_count, 4);
    }

    #[test]
    fn test_ensemble_diversity() {
        let ensemble = default_ensemble();
        let attempts = ensemble.generate("theorem t : Nat.add 0 n = n := by", 8, 1);

        // Proofs from different agents should be different (diversity)
        let proofs: Vec<&str> = attempts.iter().map(|a| a.proof.as_str()).collect();
        let unique: std::collections::HashSet<&str> = proofs.iter().copied().collect();

        // With 4 agents generating 2 each, we should have some diversity
        assert!(unique.len() > 1, "Ensemble should produce diverse proofs");
    }
}
