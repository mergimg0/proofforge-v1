//! Strategy-level policy: a learnable distribution over proof strategies.
//!
//! The policy lives on the probability simplex: π = softmax(logits).
//! GRPO updates the logits based on group-normalized advantages,
//! shifting weight toward strategies that produce verified proofs.
//!
//! This is a concrete SOS on a finite-dimensional policy space:
//! - Policy space Π = probability simplex (compact, metrizable)
//! - Evaluator E = expected proof success rate under current distribution
//! - Update δ = GRPO step on logits (softmax re-normalized)
//! - Monotone Improvement: policy gradient theorem on softmax policies
//! - Bounded Step: learning rate + softmax bounds KL divergence
//! - Constraint Preservation: C = ⊤ (unconstrained)

use serde::{Deserialize, Serialize};

/// A strategy-level policy: softmax distribution over proof strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPolicy {
    /// Raw logits (unnormalized log-probabilities)
    pub logits: Vec<f64>,
    /// Strategy names (for display)
    pub strategy_names: Vec<String>,
    /// Learning rate for GRPO updates
    pub lr: f64,
    /// Version counter (for staleness tracking)
    pub version: u64,
}

impl StrategyPolicy {
    pub fn new(strategy_names: Vec<String>, lr: f64) -> Self {
        let n = strategy_names.len();
        Self {
            logits: vec![0.0; n], // Uniform initialization
            strategy_names,
            lr,
            version: 0,
        }
    }

    /// Current probability distribution (softmax of logits).
    pub fn probabilities(&self) -> Vec<f64> {
        softmax(&self.logits)
    }

    /// Sample a strategy index according to the current distribution.
    pub fn sample(&self, rng: &mut impl FnMut() -> f64) -> usize {
        let probs = self.probabilities();
        let u = rng();
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Number of strategies.
    pub fn num_strategies(&self) -> usize {
        self.logits.len()
    }

    /// GRPO update: given per-strategy advantages, update logits.
    ///
    /// The update rule is: logit_i += lr * advantage_i * prob_i
    /// (policy gradient with softmax parameterization)
    ///
    /// Returns the KL divergence between old and new policy (for step bound checking).
    pub fn grpo_update(&mut self, advantages: &[f64]) -> GRPOUpdateResult {
        assert_eq!(advantages.len(), self.logits.len());

        let old_probs = self.probabilities();
        let old_evaluator = self.expected_reward_from_probs(&old_probs, advantages);

        // Policy gradient update on logits
        for (i, adv) in advantages.iter().enumerate() {
            self.logits[i] += self.lr * adv;
        }

        let new_probs = self.probabilities();
        let new_evaluator = self.expected_reward_from_probs(&new_probs, advantages);

        // KL divergence: D_KL(new || old)
        let kl = kl_divergence(&new_probs, &old_probs);

        self.version += 1;

        GRPOUpdateResult {
            old_probs,
            new_probs,
            kl_divergence: kl,
            evaluator_before: old_evaluator,
            evaluator_after: new_evaluator,
        }
    }

    /// Compute expected reward given per-strategy success rates and current probs.
    fn expected_reward_from_probs(&self, probs: &[f64], rates: &[f64]) -> f64 {
        probs.iter().zip(rates.iter()).map(|(p, r)| p * r).sum()
    }

    /// Compute expected reward given per-strategy success rates.
    pub fn expected_reward(&self, strategy_rates: &[f64]) -> f64 {
        let probs = self.probabilities();
        self.expected_reward_from_probs(&probs, strategy_rates)
    }
}

/// Result of a GRPO update step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRPOUpdateResult {
    pub old_probs: Vec<f64>,
    pub new_probs: Vec<f64>,
    pub kl_divergence: f64,
    pub evaluator_before: f64,
    pub evaluator_after: f64,
}

impl GRPOUpdateResult {
    /// Check SOS monotone improvement axiom.
    pub fn is_monotone_improvement(&self) -> bool {
        self.evaluator_after >= self.evaluator_before - 1e-10
    }
}

/// Compute GRPO advantages from per-strategy success rates.
///
/// Group normalization: advantage_i = (rate_i - mean) / std
/// This is the core GRPO insight: no learned value function,
/// just group statistics.
pub fn compute_strategy_advantages(strategy_rates: &[f64]) -> Vec<f64> {
    let n = strategy_rates.len();
    if n == 0 {
        return vec![];
    }

    let mean: f64 = strategy_rates.iter().sum::<f64>() / n as f64;
    let variance: f64 = strategy_rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt().max(1e-8);

    strategy_rates.iter().map(|r| (r - mean) / std).collect()
}

/// Softmax: convert logits to probabilities.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// KL divergence: D_KL(p || q) = Σ p_i * ln(p_i / q_i)
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi > 1e-15 && *qi > 1e-15 {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_init() {
        let policy = StrategyPolicy::new(
            vec!["a".into(), "b".into(), "c".into()],
            0.1,
        );
        let probs = policy.probabilities();
        // Should be approximately uniform
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_grpo_update_shifts_weight() {
        let mut policy = StrategyPolicy::new(
            vec!["good".into(), "bad".into()],
            1.0,
        );

        // Strategy 0 succeeds always (rate=1.0), strategy 1 never (rate=0.0)
        let rates = vec![1.0, 0.0];
        let advantages = compute_strategy_advantages(&rates);

        // Advantage of strategy 0 should be positive
        assert!(advantages[0] > 0.0);
        assert!(advantages[1] < 0.0);

        let result = policy.grpo_update(&advantages);

        // After update, strategy 0 should have higher probability
        assert!(result.new_probs[0] > result.old_probs[0]);
        assert!(result.new_probs[1] < result.old_probs[1]);
    }

    #[test]
    fn test_grpo_monotone_improvement() {
        let mut policy = StrategyPolicy::new(
            vec!["a".into(), "b".into(), "c".into()],
            0.5,
        );

        // Run 20 GRPO steps with consistent rewards
        let rates = vec![0.8, 0.3, 0.1]; // Strategy 0 is best
        let mut evaluators = Vec::new();

        for _ in 0..20 {
            let eval = policy.expected_reward(&rates);
            evaluators.push(eval);

            let advantages = compute_strategy_advantages(&rates);
            policy.grpo_update(&advantages);
        }

        // Evaluator should be monotone non-decreasing
        for window in evaluators.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-10,
                "Monotone violation: {} < {}",
                window[1],
                window[0]
            );
        }

        // Final evaluator should be close to the best strategy rate
        let final_eval = *evaluators.last().unwrap();
        assert!(final_eval > 0.7, "Should converge toward best strategy: {final_eval}");
    }

    #[test]
    fn test_softmax_properties() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1");
        assert!(probs[2] > probs[1] && probs[1] > probs[0], "Should be ordered");
    }

    #[test]
    fn test_kl_divergence_self() {
        let p = vec![0.3, 0.5, 0.2];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "KL(p||p) should be 0");
    }
}
