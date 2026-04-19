//! Evaluator tracking: monitors the proof success rate E(π_n) across
//! training steps and verifies the SOS axioms empirically.
//!
//! The evaluator is the SACRED OBJECT: it must never decrease.
//! This module tracks it and flags violations.

use crate::types::SOSVerification;
use serde::{Deserialize, Serialize};

/// A single training step's metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step: usize,
    /// E(π_n): proof success rate at this step
    pub evaluator: f64,
    /// Number of proofs attempted
    pub attempts: usize,
    /// Number of proofs verified
    pub verified: usize,
    /// SOS axiom verification
    pub sos_check: SOSCheck,
    /// Per-agent success rates (for variance measurement)
    pub agent_rates: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOSCheck {
    pub monotone_improvement: bool,
    pub bounded_step: bool,
    pub constraint_preservation: bool,
}

impl From<SOSVerification> for SOSCheck {
    fn from(v: SOSVerification) -> Self {
        Self {
            monotone_improvement: v.monotone_improvement,
            bounded_step: v.bounded_step,
            constraint_preservation: v.constraint_preservation,
        }
    }
}

/// The evaluator tracker: accumulates metrics across training steps.
pub struct EvaluatorTracker {
    steps: Vec<StepMetrics>,
    /// Upper bound M for the evaluator (proof success rate ≤ 1.0)
    pub upper_bound: f64,
}

impl EvaluatorTracker {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            upper_bound: 1.0, // Proof success rate is bounded by 1.0
        }
    }

    /// Record a training step.
    pub fn record(&mut self, metrics: StepMetrics) {
        self.steps.push(metrics);
    }

    /// Get the current evaluator value E(π_n).
    pub fn current_evaluator(&self) -> f64 {
        self.steps.last().map(|s| s.evaluator).unwrap_or(0.0)
    }

    /// Get the previous evaluator value E(π_{n-1}).
    pub fn previous_evaluator(&self) -> f64 {
        if self.steps.len() >= 2 {
            self.steps[self.steps.len() - 2].evaluator
        } else {
            0.0
        }
    }

    /// Check monotone improvement: E(π_n) ≥ E(π_{n-1}) for all n.
    pub fn is_monotone(&self) -> bool {
        self.steps
            .windows(2)
            .all(|w| w[1].evaluator >= w[0].evaluator - 1e-10)
    }

    /// Count monotonicity violations.
    pub fn monotonicity_violations(&self) -> usize {
        self.steps
            .windows(2)
            .filter(|w| w[1].evaluator < w[0].evaluator - 1e-10)
            .count()
    }

    /// Compute the evaluator gap: M - E(π_n) for each step.
    pub fn gaps(&self) -> Vec<f64> {
        self.steps
            .iter()
            .map(|s| self.upper_bound - s.evaluator)
            .collect()
    }

    /// Compute the evaluator variance across agent success rates
    /// (for variance-acceleration analysis).
    pub fn agent_variance(&self) -> Vec<f64> {
        self.steps
            .iter()
            .map(|s| {
                if s.agent_rates.is_empty() {
                    return 0.0;
                }
                let mean: f64 =
                    s.agent_rates.iter().map(|(_, r)| r).sum::<f64>() / s.agent_rates.len() as f64;
                s.agent_rates
                    .iter()
                    .map(|(_, r)| (r - mean).powi(2))
                    .sum::<f64>()
                    / s.agent_rates.len() as f64
            })
            .collect()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> EvaluatorSummary {
        let n = self.steps.len();
        let gaps = self.gaps();
        let variances = self.agent_variance();

        EvaluatorSummary {
            total_steps: n,
            initial_evaluator: self.steps.first().map(|s| s.evaluator).unwrap_or(0.0),
            final_evaluator: self.steps.last().map(|s| s.evaluator).unwrap_or(0.0),
            upper_bound: self.upper_bound,
            is_monotone: self.is_monotone(),
            monotonicity_violations: self.monotonicity_violations(),
            initial_gap: gaps.first().copied().unwrap_or(1.0),
            final_gap: gaps.last().copied().unwrap_or(1.0),
            mean_agent_variance: if variances.is_empty() {
                0.0
            } else {
                variances.iter().sum::<f64>() / variances.len() as f64
            },
            total_attempts: self.steps.iter().map(|s| s.attempts).sum(),
            total_verified: self.steps.iter().map(|s| s.verified).sum(),
        }
    }

    /// Get all recorded steps.
    pub fn steps(&self) -> &[StepMetrics] {
        &self.steps
    }
}

impl Default for EvaluatorTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorSummary {
    pub total_steps: usize,
    pub initial_evaluator: f64,
    pub final_evaluator: f64,
    pub upper_bound: f64,
    pub is_monotone: bool,
    pub monotonicity_violations: usize,
    pub initial_gap: f64,
    pub final_gap: f64,
    pub mean_agent_variance: f64,
    pub total_attempts: usize,
    pub total_verified: usize,
}

impl std::fmt::Display for EvaluatorSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== ProofForge Evaluator Summary ===")?;
        writeln!(f, "Steps: {}", self.total_steps)?;
        writeln!(
            f,
            "Evaluator: {:.4} -> {:.4} (bound: {:.1})",
            self.initial_evaluator, self.final_evaluator, self.upper_bound
        )?;
        writeln!(
            f,
            "Gap: {:.4} -> {:.4}",
            self.initial_gap, self.final_gap
        )?;
        writeln!(
            f,
            "Monotone: {} (violations: {})",
            if self.is_monotone { "YES" } else { "NO" },
            self.monotonicity_violations
        )?;
        writeln!(
            f,
            "Proofs: {}/{} verified ({:.1}%)",
            self.total_verified,
            self.total_attempts,
            if self.total_attempts > 0 {
                100.0 * self.total_verified as f64 / self.total_attempts as f64
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "Agent variance (mean): {:.6}",
            self.mean_agent_variance
        )?;
        Ok(())
    }
}
