//! GRPO (Group Relative Policy Optimisation) algorithm implementation.
//!
//! The core training algorithm for ProofForge. GRPO eliminates the value
//! function entirely: advantages are computed by group normalization of
//! sampled rewards, avoiding approximation error from V(s) estimation.
//!
//! On tasks with verifiable rewards (proof type-checking = binary oracle),
//! GRPO satisfies Monotone Improvement as a hypothesis, making it a
//! CONCRETE SOS — no axioms needed (proven in AReaL.lean).

use crate::types::{ChannelRewards, ChannelWeights, GRPOConfig, ProofResult, SOSVerification};

/// A group of proof attempts for a single theorem statement,
/// with their type-checking results.
#[derive(Debug)]
pub struct ProofGroup {
    /// The theorem statement being proved
    pub statement: String,
    /// Per-attempt channel rewards (correctness, elegance, generality)
    pub rewards: Vec<ChannelRewards>,
    /// Combined rewards after channel weighting
    pub combined_rewards: Vec<f64>,
}

/// GRPO advantages computed via group normalization.
/// This is the key innovation: no learned value function.
#[derive(Debug)]
pub struct GroupAdvantages {
    /// Per-attempt advantage values (group-normalized)
    pub advantages: Vec<f64>,
    /// Group mean reward
    pub group_mean: f64,
    /// Group standard deviation
    pub group_std: f64,
}

/// Compute GRPO advantages for a proof group.
///
/// The advantage of attempt i is:
///   A_i = (r_i - mean(r)) / std(r)
///
/// This group normalization is what eliminates the value function:
/// advantages are relative to the group, not to a learned baseline.
pub fn compute_advantages(group: &ProofGroup) -> GroupAdvantages {
    let n = group.combined_rewards.len();
    if n == 0 {
        return GroupAdvantages {
            advantages: vec![],
            group_mean: 0.0,
            group_std: 1.0,
        };
    }

    let mean: f64 = group.combined_rewards.iter().sum::<f64>() / n as f64;

    let variance: f64 = group
        .combined_rewards
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let std = variance.sqrt().max(1e-8); // avoid division by zero

    let advantages: Vec<f64> = group
        .combined_rewards
        .iter()
        .map(|r| (r - mean) / std)
        .collect();

    GroupAdvantages {
        advantages,
        group_mean: mean,
        group_std: std,
    }
}

/// Compute per-channel rewards from a proof result.
pub fn compute_channel_rewards(result: &ProofResult, proof_length: usize) -> ChannelRewards {
    let correctness = result.reward();

    // Elegance: inverse of proof length, normalized to [0, 1]
    // Shorter proofs are more elegant. Only meaningful if proof is correct.
    let elegance = if result.is_verified() {
        1.0 / (1.0 + proof_length as f64 / 100.0)
    } else {
        0.0
    };

    // Generality: placeholder — requires checking proof against related lemmas
    // In the full system, this would run the proof on variant statements
    let generality = if result.is_verified() { 0.5 } else { 0.0 };

    ChannelRewards {
        correctness,
        elegance,
        generality,
    }
}

/// Compute the combined reward from channel rewards.
pub fn combined_reward(channel: &ChannelRewards, weights: &ChannelWeights) -> f64 {
    channel.combined(weights)
}

/// Check SOS axioms for a training step.
///
/// This mirrors the Lean 4 verification:
/// - monotone_improvement: E(proofStep(π)) ≥ E(π)
/// - bounded_step: d(proofStep(π), π) ≤ D
/// - constraint_preservation: C(π) → C(proofStep(π))
pub fn verify_sos_step(
    evaluator_before: f64,
    evaluator_after: f64,
    step_distance: f64,
    config: &GRPOConfig,
) -> SOSVerification {
    SOSVerification {
        monotone_improvement: evaluator_after >= evaluator_before,
        bounded_step: step_distance <= config.epsilon_clip,
        constraint_preservation: true, // C = ⊤ for unconstrained GRPO
    }
}

/// Staleness filter: reject training data from policy versions
/// more than η steps behind the current version.
///
/// This is constraint lifting C_stale from AReaL.lean:
/// constraintLift(S_GRPO, C_stale)
pub fn staleness_filter(current_version: u64, data_version: u64, eta: u64) -> bool {
    current_version.saturating_sub(data_version) <= eta
}

/// M2PO token mask: reject tokens with high importance-ratio variance.
///
/// This is the second constraint lift C_m2po from AReaL.lean:
/// constraintLift(constraintLift(S_GRPO, C_stale), C_m2po)
pub fn m2po_filter(importance_ratio_m2: f64, tau: f64) -> bool {
    importance_ratio_m2 <= tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_advantages_normalization() {
        let group = ProofGroup {
            statement: "theorem test : True := trivial".to_string(),
            rewards: vec![], // not used directly
            combined_rewards: vec![1.0, 0.0, 1.0, 0.0],
        };
        let adv = compute_advantages(&group);

        // Mean should be 0.5
        assert!((adv.group_mean - 0.5).abs() < 1e-10);

        // Advantages should sum to approximately 0
        let sum: f64 = adv.advantages.iter().sum();
        assert!(sum.abs() < 1e-10);

        // Correct proofs should have positive advantage
        assert!(adv.advantages[0] > 0.0);
        assert!(adv.advantages[2] > 0.0);

        // Failed proofs should have negative advantage
        assert!(adv.advantages[1] < 0.0);
        assert!(adv.advantages[3] < 0.0);
    }

    #[test]
    fn test_binary_reward_oracle() {
        let verified = ProofResult::Verified {
            check_duration_ms: 100,
        };
        let failed = ProofResult::Failed {
            error: "type mismatch".to_string(),
            check_duration_ms: 50,
        };

        assert_eq!(verified.reward(), 1.0);
        assert_eq!(failed.reward(), 0.0);
    }

    #[test]
    fn test_staleness_filter() {
        assert!(staleness_filter(10, 8, 4)); // gap 2 ≤ 4
        assert!(staleness_filter(10, 6, 4)); // gap 4 ≤ 4
        assert!(!staleness_filter(10, 5, 4)); // gap 5 > 4
    }

    #[test]
    fn test_m2po_filter() {
        assert!(m2po_filter(1.5, 2.0)); // 1.5 ≤ 2.0
        assert!(!m2po_filter(2.5, 2.0)); // 2.5 > 2.0
    }

    #[test]
    fn test_sos_verification() {
        let config = GRPOConfig::default();

        // Valid step: evaluator improved, step bounded
        let v = verify_sos_step(0.5, 0.6, 0.1, &config);
        assert!(v.all_satisfied());

        // Invalid: evaluator decreased (violates monotone improvement)
        let v = verify_sos_step(0.6, 0.5, 0.1, &config);
        assert!(!v.monotone_improvement);
        assert!(!v.all_satisfied());
    }
}
