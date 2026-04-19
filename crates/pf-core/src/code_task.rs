//! App 1: Code verification task types.
//!
//! Defines task types for the self-bootstrapping code verification
//! pipeline. These are the code-domain analogs of theorem statements
//! and proof attempts.

use serde::{Deserialize, Serialize};

/// Fractal curriculum difficulty levels for code tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskLevel {
    /// Single-expression tasks (fibonacci, reverse string)
    Expression = 0,
    /// Single-function tasks (binary search, CSV parsing)
    Function = 1,
    /// Multi-function / class tasks (Stack, LRU cache)
    MultiFunction = 2,
    /// Module-level tasks (calculator with tokenizer + parser + evaluator)
    Module = 3,
}

/// A code generation task with its verification oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTask {
    /// Unique task identifier
    pub task_id: String,
    /// Curriculum difficulty level
    pub level: TaskLevel,
    /// Task prompt shown to the model
    pub prompt: String,
    /// Required infrastructure patterns (for infrastructure-aware sampling)
    pub required_infrastructure: Vec<String>,
    /// Categorization tags
    pub tags: Vec<String>,
}

/// A code solution attempt (analog of ProofAttempt for code).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAttempt {
    /// The task this attempt solves
    pub task_id: String,
    /// Generated Python code
    pub code: String,
    /// Policy version that generated this attempt
    pub policy_version: u64,
    /// Sampling temperature used
    pub temperature: f64,
}

/// Result of test suite execution (analog of ProofResult for code).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestResult {
    /// All tests passed. Reward = 1.0.
    Passed {
        tests_run: usize,
        duration_ms: u64,
    },
    /// Some or all tests failed. Reward = 0.0.
    Failed {
        tests_run: usize,
        tests_passed: usize,
        error: String,
        duration_ms: u64,
    },
    /// Code was rejected before execution (forbidden imports, etc.)
    Rejected {
        reason: String,
    },
}

impl TestResult {
    /// Binary reward: 1.0 if all tests passed, 0.0 otherwise.
    pub fn reward(&self) -> f64 {
        match self {
            TestResult::Passed { .. } => 1.0,
            TestResult::Failed { .. } => 0.0,
            TestResult::Rejected { .. } => 0.0,
        }
    }

    pub fn is_passed(&self) -> bool {
        matches!(self, TestResult::Passed { .. })
    }
}

/// Infrastructure pattern mastery for a coding pattern.
/// Analog of tactic mastery for Lean proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMastery {
    pub pattern_name: String,
    pub attempts: usize,
    pub successes: usize,
}

impl PatternMastery {
    pub fn mastery_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 0.0;
        }
        self.successes as f64 / self.attempts as f64
    }
}

/// Expanding ring curriculum state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumState {
    /// Current highest active tier
    pub active_tier: u32,
    /// Total available tiers
    pub total_tiers: u32,
    /// Items per tier
    pub items_per_tier: Vec<usize>,
    /// Pass rates per tier (rolling average)
    pub tier_pass_rates: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_levels() {
        assert!((TaskLevel::Expression as u8) < (TaskLevel::Function as u8));
        assert!((TaskLevel::Function as u8) < (TaskLevel::MultiFunction as u8));
        assert!((TaskLevel::MultiFunction as u8) < (TaskLevel::Module as u8));
    }

    #[test]
    fn test_test_result_reward() {
        let passed = TestResult::Passed {
            tests_run: 5,
            duration_ms: 100,
        };
        assert_eq!(passed.reward(), 1.0);

        let failed = TestResult::Failed {
            tests_run: 5,
            tests_passed: 3,
            error: "assertion".to_string(),
            duration_ms: 50,
        };
        assert_eq!(failed.reward(), 0.0);

        let rejected = TestResult::Rejected {
            reason: "forbidden import".to_string(),
        };
        assert_eq!(rejected.reward(), 0.0);
    }

    #[test]
    fn test_pattern_mastery() {
        let p = PatternMastery {
            pattern_name: "recursion".to_string(),
            attempts: 10,
            successes: 7,
        };
        assert!((p.mastery_rate() - 0.7).abs() < 1e-10);
    }
}
