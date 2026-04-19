//! Async scheduler: distributes proof generation and checking across worker pools.

use pf_core::types::{GRPOConfig, ProofAttempt, ProofResult};
use pf_core::grpo;
use pf_core::reward::RewardOracle;

use std::sync::Arc;

/// A proof task: generate and check a proof for a theorem statement.
#[derive(Debug)]
pub struct ProofTask {
    pub statement: String,
    pub policy_version: u64,
}

/// Result of processing a proof task through the pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    pub task: ProofTask,
    pub attempts: Vec<ProofAttempt>,
    pub results: Vec<ProofResult>,
    /// Whether this batch passed staleness filtering
    pub staleness_ok: bool,
}

/// The async pipeline scheduler.
///
/// Orchestrates:
/// 1. GPU pool: generates proof attempts (LLM inference)
/// 2. CPU pool: type-checks proofs (Lean 4 compiler)
/// 3. Staleness filter: rejects stale batches (constraint lift 1)
/// 4. M2PO filter: masks high-variance tokens (constraint lift 2)
/// 5. GRPO update on filtered results
pub struct AsyncPipeline {
    config: GRPOConfig,
    oracle: Arc<dyn RewardOracle>,
    current_policy_version: u64,
}

impl AsyncPipeline {
    pub fn new(config: GRPOConfig, oracle: Arc<dyn RewardOracle>) -> Self {
        Self {
            config,
            oracle,
            current_policy_version: 0,
        }
    }

    /// Process a batch of proof tasks through the pipeline.
    ///
    /// This is one iteration of the doubly constraint-lifted SOS:
    /// 1. Generate proof attempts (would be GPU in production)
    /// 2. Type-check each attempt via the reward oracle
    /// 3. Apply staleness filter (C_stale)
    /// 4. Apply M2PO filter (C_m2po)
    /// 5. Compute GRPO advantages on filtered results
    pub fn process_batch(
        &mut self,
        tasks: Vec<ProofTask>,
        attempts_per_task: Vec<Vec<ProofAttempt>>,
    ) -> Vec<PipelineResult> {
        let mut results = Vec::new();

        for (task, attempts) in tasks.into_iter().zip(attempts_per_task.into_iter()) {
            // Step 1: Type-check each attempt
            let check_results: Vec<ProofResult> =
                attempts.iter().map(|a| self.oracle.check(a)).collect();

            // Step 2: Staleness filter (constraint lift 1)
            let staleness_ok = attempts.iter().all(|a| {
                grpo::staleness_filter(
                    self.current_policy_version,
                    a.policy_version,
                    self.config.staleness_eta,
                )
            });

            results.push(PipelineResult {
                task,
                attempts,
                results: check_results,
                staleness_ok,
            });
        }

        results
    }

    /// Advance the policy version after a GRPO update step.
    pub fn advance_policy_version(&mut self) {
        self.current_policy_version += 1;
    }

    pub fn current_version(&self) -> u64 {
        self.current_policy_version
    }
}
