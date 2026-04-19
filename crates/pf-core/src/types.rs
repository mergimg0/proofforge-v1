//! Core types aligned with the SOS formalization.

use serde::{Deserialize, Serialize};

/// A proof attempt: the LLM's output for a given theorem statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttempt {
    /// The theorem statement (Lean 4 syntax)
    pub statement: String,
    /// The proposed proof (Lean 4 tactic block)
    pub proof: String,
    /// Which agent produced this attempt
    pub agent_id: AgentId,
    /// Policy version that generated this attempt
    pub policy_version: u64,
}

/// Result of type-checking a proof attempt.
/// This is the binary reward oracle: Verified = 1.0, Failed = 0.0.
/// No approximation. No hallucination. No learned value function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofResult {
    /// Lean 4 type checker accepted the proof. Reward = 1.0.
    Verified {
        /// Time taken for type checking
        check_duration_ms: u64,
    },
    /// Lean 4 type checker rejected the proof. Reward = 0.0.
    Failed {
        /// The error message from Lean 4
        error: String,
        /// Time taken before failure
        check_duration_ms: u64,
    },
}

impl ProofResult {
    /// Binary reward: 1.0 if verified, 0.0 if failed.
    /// This is the exact reward that makes GRPO a concrete SOS
    /// (no value function approximation error).
    pub fn reward(&self) -> f64 {
        match self {
            ProofResult::Verified { .. } => 1.0,
            ProofResult::Failed { .. } => 0.0,
        }
    }

    pub fn is_verified(&self) -> bool {
        matches!(self, ProofResult::Verified { .. })
    }
}

/// Identifier for a proof search specialist agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentId {
    /// Structural and strong induction strategies
    Induction,
    /// Ring/field rewriting, simp lemmas
    Algebra,
    /// Tactic combination search (omega, linarith, norm_num, etc.)
    Tactic,
    /// Find similar proofs in Mathlib and adapt
    Analogy,
}

/// A reward channel for per-channel GRPO normalization.
/// ProofForge uses three channels (from the ProofForge paper):
/// correctness, elegance, and generality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardChannel {
    pub name: String,
    pub weight: f64,
}

/// Per-channel reward values for a single proof attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRewards {
    /// Binary correctness: did the proof type-check?
    pub correctness: f64,
    /// Elegance: inverse of proof length / tactic count (shorter = better)
    pub elegance: f64,
    /// Generality: does the proof approach generalize to related lemmas?
    pub generality: f64,
}

impl ChannelRewards {
    /// Weighted combination of channel rewards.
    pub fn combined(&self, weights: &ChannelWeights) -> f64 {
        weights.correctness * self.correctness
            + weights.elegance * self.elegance
            + weights.generality * self.generality
    }
}

/// Channel weights for the combined reward.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChannelWeights {
    pub correctness: f64,
    pub elegance: f64,
    pub generality: f64,
}

impl Default for ChannelWeights {
    fn default() -> Self {
        // Correctness dominates: a wrong proof is worthless regardless of elegance
        Self {
            correctness: 0.7,
            elegance: 0.2,
            generality: 0.1,
        }
    }
}

/// SOS verification result for a single GRPO step.
/// Mirrors the three axioms from the Lean formalization.
#[derive(Debug, Clone)]
pub struct SOSVerification {
    /// Axiom (i): E(δ(π)) ≥ E(π)
    pub monotone_improvement: bool,
    /// Axiom (ii): d(δ(π), π) ≤ Δ
    pub bounded_step: bool,
    /// Axiom (iii): C(π) → C(δ(π))
    pub constraint_preservation: bool,
}

impl SOSVerification {
    /// All three axioms satisfied — the SOS convergence theorem applies.
    pub fn all_satisfied(&self) -> bool {
        self.monotone_improvement && self.bounded_step && self.constraint_preservation
    }
}

/// Configuration for the GRPO training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GRPOConfig {
    /// G: number of proof attempts per theorem statement
    pub group_size: usize,
    /// ε-clip: bounds policy ratio, controlling step size (Δ in SOS)
    pub epsilon_clip: f64,
    /// KL penalty coefficient to reference policy
    pub kl_beta: f64,
    /// Per-channel reward weights
    pub channel_weights: ChannelWeights,
    /// Staleness parameter η: max policy version gap
    pub staleness_eta: u64,
    /// M2PO threshold τ: max importance ratio second moment
    pub m2po_tau: f64,
    /// Maximum gradient norm before step-bound warning.
    /// This is an observational proxy, NOT the formal SOS bounded-step axiom
    /// (which requires KL divergence or total variation distance).
    pub max_grad_norm: f64,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            group_size: 16,
            epsilon_clip: 0.2,
            kl_beta: 0.01,
            channel_weights: ChannelWeights::default(),
            staleness_eta: 4,
            m2po_tau: 2.0,
            max_grad_norm: 2.0, // post-clipping norms above 2.0 indicate large updates
        }
    }
}
