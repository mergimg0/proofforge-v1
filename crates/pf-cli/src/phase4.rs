//! Phase 4: Neural GRPO training with Python bridge and SOS governance.
//!
//! Extracted from main.rs to reduce monolith re-read cost.
//! Contains run_phase4(), build_few_shot_payload(), evaluate_policy_detailed(),
//! and the AgentExt trait.

use pf_agents::adaptive::AdaptiveAgent;
use pf_agents::ProofAgent;
use pf_core::bridge::TrainingBridge;
use pf_core::dataset::TheoremDataset;
use pf_core::evaluator::{EvaluatorTracker, SOSCheck, StepMetrics};
use pf_core::grpo;
use pf_core::memory::{ProofExample, ProofMemory};
use pf_core::policy::StrategyPolicy;
use pf_core::types::{GRPOConfig, SOSVerification};
use pf_lean::checker::LeanChecker;

/// Trait extension to allow downcasting agents for memory updates.
pub trait AgentExt {
    fn as_any_mut(&mut self) -> Option<&mut AdaptiveAgent> { None }
}

impl AgentExt for Box<dyn ProofAgent> {
    fn as_any_mut(&mut self) -> Option<&mut AdaptiveAgent> {
        (**self).as_adaptive_mut()
    }
}

/// Build a JSON-serializable few-shot payload from ProofMemory, one entry
/// per difficulty tier (top-3 examples each). Used to push context to Python.
pub fn build_few_shot_payload(memory: &ProofMemory) -> Vec<serde_json::Value> {
    let mut entries = Vec::new();
    for difficulty in 1u8..=4 {
        for ex in memory.few_shot_examples(difficulty, &[], 3) {
            entries.push(serde_json::json!({
                "statement": ex.statement,
                "proof": ex.proof,
                "difficulty": ex.difficulty,
            }));
        }
    }
    entries
}

/// Evaluate per-theorem and return (pass_rate, per_statement_pass_vec).
/// Used by Phase 4's SOS violation handler to identify the regressed difficulty tier.
pub fn evaluate_policy_detailed(
    bridge: &TrainingBridge,
    lean_checker: &LeanChecker,
    eval_statements: &[String],
) -> (f64, Vec<bool>) {
    let eval_response = match bridge.evaluate(eval_statements, 4) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Evaluation failed: {e}");
            return (0.0, vec![false; eval_statements.len()]);
        }
    };

    let mut verified = 0;
    let mut total = 0;
    let mut per_stmt = Vec::with_capacity(eval_statements.len());
    for (proofs, stmt) in eval_response.proofs.iter().zip(eval_statements.iter()) {
        let mut stmt_passed = false;
        for proof in proofs {
            total += 1;
            let source = format!("{}\n  {}\n", stmt, proof);
            if lean_checker.check_source(&source)
                .map(|r| r.success).unwrap_or(false)
            {
                verified += 1;
                stmt_passed = true;
            }
        }
        per_stmt.push(stmt_passed);
    }

    let rate = if total > 0 { verified as f64 / total as f64 } else { 0.0 };
    (rate, per_stmt)
}

/// Phase 4: Neural GRPO training with Python bridge and SOS governance.
///
/// `proof_memory` is a Rust-side experience buffer. Verified proofs are added
/// here and periodically pushed to the Python server via /configure so that
/// `_generate_proofs()` can use them as few-shot examples even when the reward
/// signal hasn't yet flowed through the /reward endpoint.
pub fn run_phase4(
    bridge: &TrainingBridge,
    lean_checker: &LeanChecker,
    dataset: &TheoremDataset,
    config: &GRPOConfig,
    num_steps: usize,
    proof_memory: &mut ProofMemory,
) {
    use rand::seq::SliceRandom;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ProofForge — Phase 4: Neural GRPO via Training Bridge    ║");
    println!("║   Real Backprop + Lean Verification + SOS Governance       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut tracker = EvaluatorTracker::new();
    let mut rng = rand::thread_rng();
    let all_theorems = dataset.all_theorems();
    let batch_size = std::cmp::min(4, all_theorems.len());
    let eval_interval = 5;
    let max_regression = 0.05;

    tracing::info!(
        "Steps: {num_steps} | Batch: {batch_size} | Group: {} | Eval every {eval_interval} steps",
        config.group_size
    );

    let eval_count = 12.min(all_theorems.len());
    let eval_per_tier = ((eval_count / 4).max(1)) as usize;
    let mut stratified_eval: Vec<(String, u8)> = (1u8..=4)
        .flat_map(|d| {
            dataset.theorems_by_difficulty(d)
                .into_iter()
                .take(eval_per_tier)
                .map(move |t| (t.statement.clone(), d))
        })
        .take(eval_count)
        .collect();
    if stratified_eval.len() < eval_count {
        let already: std::collections::HashSet<String> =
            stratified_eval.iter().map(|(s, _)| s.clone()).collect();
        for t in &all_theorems {
            if stratified_eval.len() >= eval_count { break; }
            if !already.contains(&t.statement) {
                stratified_eval.push((t.statement.clone(), t.difficulty));
            }
        }
    }
    let eval_statements: Vec<String> = stratified_eval.iter().map(|(s, _)| s.clone()).collect();
    let eval_difficulties: Vec<u8> = stratified_eval.iter().map(|(_, d)| *d).collect();

    tracing::info!(
        "Theorems: {} total, eval set: {} (stratified: {}×{} tiers)",
        all_theorems.len(), eval_statements.len(), eval_per_tier, 4
    );

    let (mut last_evaluator, mut prev_eval_pass) =
        evaluate_policy_detailed(bridge, lean_checker, &eval_statements);
    let mut max_evaluator = last_evaluator;
    tracing::info!("Baseline evaluator: {last_evaluator:.4}");

    let mut formal_monotone_checks: usize = 0;
    let mut formal_monotone_violations: usize = 0;
    let mut formal_bounded_violations: usize = 0;
    let mut tolerated_violations: usize = 0;
    let mut current_lr: f64 = 5e-6;
    let min_lr: f64 = 1e-7;

    println!("┌─────┬──────────┬──────────┬────────┬──────────┬────┐");
    println!("│ Step│ BatchRwd │ Loss     │ GradN  │ Eval     │ M? │");
    println!("├─────┼──────────┼──────────┼────────┼──────────┼────┤");

    for step in 0..num_steps {
        let batch: Vec<_> = all_theorems.choose_multiple(&mut rng, batch_size).collect();
        let statements: Vec<String> = batch.iter().map(|t| t.statement.clone()).collect();

        let gen_response = match bridge.generate(&statements, config.group_size) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Step {step}: generate failed: {e}");
                tracker.record(StepMetrics {
                    step, evaluator: last_evaluator, attempts: 0, verified: 0,
                    sos_check: SOSCheck { monotone_improvement: true,
                        bounded_step: true, constraint_preservation: true },
                    agent_rates: vec![],
                });
                println!("│ {:>3} │  FAILED  │          │        │          │    │", step);
                continue;
            }
        };

        let rewards: Vec<Vec<f64>> = gen_response.proofs.iter()
            .zip(statements.iter())
            .map(|(proofs, stmt)| {
                proofs.iter().map(|proof| {
                    let source = format!("{}\n  {}\n", stmt, proof);
                    if lean_checker.check_source(&source)
                        .map(|r| r.success).unwrap_or(false) { 1.0 } else { 0.0 }
                }).collect()
            }).collect();

        let mem_version_before = proof_memory.version;
        for (thm, (proofs, proof_rewards)) in batch.iter()
            .zip(gen_response.proofs.iter().zip(rewards.iter()))
        {
            for (proof, &r) in proofs.iter().zip(proof_rewards.iter()) {
                if r > 0.5 && !proof.trim().is_empty() {
                    proof_memory.add(&thm.id, ProofExample {
                        statement: thm.statement.clone(),
                        proof: proof.clone(),
                        difficulty: thm.difficulty,
                        strategy: "neural".to_string(),
                        found_at_step: step,
                        proof_length: proof.len(),
                        tags: vec![],
                    });
                }
            }
        }

        if proof_memory.version != mem_version_before && step % eval_interval == 0 {
            let few_shot_payload = build_few_shot_payload(proof_memory);
            if !few_shot_payload.is_empty() {
                match bridge.configure(&serde_json::json!({"few_shot_context": few_shot_payload})) {
                    Ok(resp) => tracing::debug!("Step {step}: pushed {} few-shot entries: {:?}",
                        few_shot_payload.len(), resp.changes),
                    Err(e) => tracing::warn!("Step {step}: few-shot push failed: {e}"),
                }
            }
        }

        let reward_response = match bridge.reward(&statements, &gen_response.proofs, &rewards) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Step {step}: reward failed: {e}");
                tracker.record(StepMetrics {
                    step, evaluator: last_evaluator, attempts: 0, verified: 0,
                    sos_check: SOSCheck { monotone_improvement: true,
                        bounded_step: true, constraint_preservation: true },
                    agent_rates: vec![],
                });
                println!("│ {:>3} │  FAILED  │          │        │          │    │", step);
                continue;
            }
        };

        let step_response = match bridge.step(
            &statements, &gen_response.proofs,
            &rewards, &reward_response.advantages,
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Step {step}: training step failed: {e}");
                tracker.record(StepMetrics {
                    step, evaluator: last_evaluator, attempts: 0, verified: 0,
                    sos_check: SOSCheck { monotone_improvement: true,
                        bounded_step: true, constraint_preservation: true },
                    agent_rates: vec![],
                });
                println!("│ {:>3} │  FAILED  │          │        │          │    │", step);
                continue;
            }
        };

        let bounded_step = step_response.grad_norm <= config.max_grad_norm;
        if !bounded_step {
            tracing::warn!("Step {step}: GRAD NORM {:.4} exceeds max_grad_norm {:.4}",
                step_response.grad_norm, config.max_grad_norm);
        }

        let batch_attempts: usize = rewards.iter().map(|r| r.len()).sum();
        let batch_verified: usize = rewards.iter().flat_map(|r| r.iter())
            .filter(|&&r| r > 0.5).count();

        let mut eval_str = String::from("—");
        let mut mono_sym = "—";
        let mut evaluator_for_record = last_evaluator;
        let mut eval_sos: Option<SOSVerification> = None;

        if (step + 1) % eval_interval == 0 || step == num_steps - 1 {
            let (evaluator, current_eval_pass) =
                evaluate_policy_detailed(bridge, lean_checker, &eval_statements);
            max_evaluator = max_evaluator.max(evaluator);
            evaluator_for_record = evaluator;

            let sos = grpo::verify_sos_step(
                last_evaluator, evaluator,
                step_response.mean_kl, config,
            );

            let regression_from_peak = max_evaluator - evaluator;
            let tolerated = regression_from_peak <= max_regression;

            if !tolerated {
                tracing::warn!(
                    "Step {step}: REGRESSION BEYOND TOLERANCE — peak {max_evaluator:.4}, \
                     now {evaluator:.4} (drop {regression_from_peak:.4} > max {max_regression:.4})"
                );
                mono_sym = "✗";

                let new_lr = (current_lr * 0.5).max(min_lr);
                if new_lr < current_lr {
                    match bridge.configure(&serde_json::json!({"learning_rate": new_lr})) {
                        Ok(resp) => {
                            tracing::warn!(
                                "Step {step}: SOS violation → LR reduced {current_lr:.2e} → {new_lr:.2e}: {:?}",
                                resp.changes
                            );
                            current_lr = new_lr;
                        }
                        Err(e) => {
                            tracing::error!("Step {step}: LR reduction failed (bridge error): {e}");
                        }
                    }
                }

                let regressed_difficulty = eval_difficulties.iter()
                    .zip(prev_eval_pass.iter().zip(current_eval_pass.iter()))
                    .filter(|(_, (was_pass, now_pass))| **was_pass && !**now_pass)
                    .map(|(&diff, _)| diff)
                    .max()
                    .unwrap_or(1);
                let recovery_examples = proof_memory.few_shot_examples(regressed_difficulty, &[], 3);
                if !recovery_examples.is_empty() {
                    let payload: Vec<serde_json::Value> = recovery_examples.iter().map(|ex| {
                        serde_json::json!({
                            "statement": ex.statement,
                            "proof": ex.proof,
                            "difficulty": ex.difficulty,
                        })
                    }).collect();
                    match bridge.configure(&serde_json::json!({"few_shot_context": payload})) {
                        Ok(resp) => tracing::info!(
                            "Step {step}: SOS recovery — pushed {} tier-{} examples: {:?}",
                            payload.len(), regressed_difficulty, resp.changes
                        ),
                        Err(e) => tracing::warn!("Step {step}: recovery few-shot push failed: {e}"),
                    }
                }
            } else if !sos.monotone_improvement {
                tracing::info!(
                    "Step {step}: Formal monotone violated ({last_evaluator:.4} → {evaluator:.4}), \
                     tolerated (peak distance {regression_from_peak:.4} ≤ {max_regression:.4})"
                );
                mono_sym = "~";
            } else {
                mono_sym = "✓";
            }

            formal_monotone_checks += 1;
            if !sos.monotone_improvement { formal_monotone_violations += 1; }
            if !sos.bounded_step { formal_bounded_violations += 1; }
            if !tolerated { tolerated_violations += 1; }

            eval_sos = Some(sos);

            eval_str = format!("{evaluator:.6}");
            last_evaluator = evaluator;
            prev_eval_pass = current_eval_pass;
        }

        tracker.record(StepMetrics {
            step,
            evaluator: evaluator_for_record,
            attempts: batch_attempts,
            verified: batch_verified,
            sos_check: if let Some(sos) = &eval_sos {
                SOSCheck {
                    monotone_improvement: sos.monotone_improvement,
                    bounded_step: sos.bounded_step,
                    constraint_preservation: true,
                }
            } else {
                SOSCheck {
                    monotone_improvement: true,
                    bounded_step,
                    constraint_preservation: true,
                }
            },
            agent_rates: vec![],
        });

        println!(
            "│ {:>3} │ {:.6} │ {:.6} │ {:.4} │ {:>8} │ {:>2} │",
            step, step_response.batch_reward, step_response.loss,
            step_response.grad_norm, eval_str, mono_sym,
        );
    }

    println!("└─────┴──────────┴──────────┴────────┴──────────┴────┘");
    println!();

    let summary = tracker.summary();
    println!("{summary}");

    println!("=== SOS Axiom Verification (Phase 4) ===");
    println!("  Eval checkpoints: {} (every {} steps)", formal_monotone_checks, eval_interval);
    println!();
    println!("  Monotone Improvement (formal, strict E(π_n) ≥ E(π_{{n-1}})):");
    println!("    {}/{} eval transitions passed",
        formal_monotone_checks - formal_monotone_violations, formal_monotone_checks);
    println!("  Monotone Improvement (operational, {:.0}% tolerance from peak):",
        max_regression * 100.0);
    println!("    {}/{} eval transitions passed",
        formal_monotone_checks - tolerated_violations, formal_monotone_checks);
    println!();
    println!("  Bounded Step (formal, mean_kl ≤ ε_clip = {:.2}):", config.epsilon_clip);
    println!("    {}/{} eval transitions passed",
        formal_monotone_checks - formal_bounded_violations, formal_monotone_checks);
    println!("  Bounded Step (operational, grad_norm ≤ {:.1}):", config.max_grad_norm);
    println!("    observational only — see per-step GRAD NORM warnings above");
    println!();
    println!("  Constraint Preservation: trivially satisfied (C = ⊤)");
    if formal_monotone_violations == 0 && formal_bounded_violations == 0 {
        println!("\n  ✓ ALL FORMAL SOS AXIOMS SATISFIED at eval checkpoints.");
    } else if tolerated_violations == 0 {
        println!("\n  ~ Formal violations exist but within operational tolerance.");
        println!("    Disruption phase detected — model may be transitioning basins.");
    }
}
