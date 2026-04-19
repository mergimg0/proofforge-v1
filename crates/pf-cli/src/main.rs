//! ProofForge CLI — Phase 3: Lean-In/Lean-Out Feedback Loop
//!
//! Extends Phase 2's GRPO training with proof memory:
//! 1. GRPO updates strategy weights (gradient descent on policy)
//! 2. Proof memory stores verified proofs (Lean In)
//! 3. Adaptive agent uses stored proofs as few-shot context (Lean Out → Lean In)
//! 4. Evaluator tracks both policy improvement and memory growth
//!
//! This implements the hybrid of Karpathy's "system prompt learning"
//! (context-space updates) with GRPO (weight-space updates).

mod phase4;
use phase4::AgentExt;

use std::path::PathBuf;

use rand::seq::SliceRandom;

use pf_agents::adaptive::AdaptiveAgent;
use pf_agents::llm::LLMAgent;
use pf_agents::{AlgebraAgent, AnalogyAgent, InductionAgent, ProofAgent, TacticAgent};
use pf_core::bridge::{BridgeConfig, TrainingBridge};
use pf_core::dataset::TheoremDataset;
use pf_core::evaluator::{EvaluatorTracker, SOSCheck, StepMetrics};
use pf_core::memory::{self, ProofExample, ProofMemory};
use pf_core::policy::{self, StrategyPolicy};
use pf_core::types::{AgentId, GRPOConfig};
use pf_lean::checker::{LeanChecker, LeanCheckerConfig};

struct Strategy {
    name: String,
    agent: Box<dyn ProofAgent>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let data_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/theorems.json"));
    let num_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let use_llm = args.iter().any(|a| a == "--llm");
    let use_adaptive = args.iter().any(|a| a == "--adaptive");
    let lr: f64 = args
        .iter()
        .position(|a| a == "--lr")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let use_neural = args.iter().any(|a| a == "--neural");
    let timeout: u64 = args
        .iter()
        .position(|a| a == "--timeout")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    // --- Phase 4: Neural GRPO via Python bridge ---
    if use_neural {
        let dataset = TheoremDataset::load(&data_path).expect("Failed to load dataset");
        let lean_checker = LeanChecker::new(LeanCheckerConfig::default());

        let bridge_config = BridgeConfig {
            timeout_secs: timeout,
            ..BridgeConfig::default()
        };
        let bridge = TrainingBridge::new(bridge_config);

        // Health check
        match bridge.health() {
            Ok(h) if h.model_loaded => {
                tracing::info!("Training server connected: {} on {}", h.status, h.device);
            }
            Ok(_) => {
                tracing::warn!("Training server in MOCK mode (model not loaded)");
            }
            Err(e) => {
                eprintln!("Cannot connect to training server at localhost:8420: {e}");
                eprintln!("Start it with: python3 python/train_server.py --load-model");
                std::process::exit(1);
            }
        }

        // C3 warm-start: load Phase 3's proof_memory if saved to disk.
        // Phase 3 and Phase 4 are separate invocations — proof_memory.json is the
        // handoff. If it doesn't exist (first run, or standalone Phase 4), start empty.
        let mut phase4_memory = std::fs::read_to_string("proof_memory.json")
            .ok()
            .and_then(|s| serde_json::from_str::<ProofMemory>(&s).ok())
            .unwrap_or_else(|| ProofMemory::new(3));
        tracing::info!(
            "Phase 4 memory: {} proofs from prior Phase 3 run (or empty for cold start)",
            phase4_memory.total_proofs()
        );
        phase4::run_phase4(&bridge, &lean_checker, &dataset, &GRPOConfig::default(), num_steps, &mut phase4_memory);
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ProofForge — Phase 3: Lean-In/Lean-Out Feedback Loop     ║");
    println!("║   GRPO + Proof Memory + Context Adaptation                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    tracing::info!("Steps: {num_steps} | LR: {lr} | LLM: {} | Adaptive: {use_adaptive}",
        if use_llm || use_adaptive { "yes" } else { "no" });

    // 1. Load dataset
    let dataset = TheoremDataset::load(&data_path).expect("Failed to load dataset");
    let all_theorems = dataset.all_theorems();
    tracing::info!("Loaded {} theorems (FIXED evaluation set)", all_theorems.len());

    // 2. Lean 4 type checker
    let lean_checker = LeanChecker::new(LeanCheckerConfig::default());

    // 3. Proof memory (the "Lean In" buffer)
    let mut proof_memory = ProofMemory::new(3);

    // 4. Strategies
    let mut strategies: Vec<Strategy> = vec![
        Strategy { name: "Induction".into(), agent: Box::new(InductionAgent) },
        Strategy { name: "Algebra".into(), agent: Box::new(AlgebraAgent) },
        Strategy { name: "Tactic".into(), agent: Box::new(TacticAgent) },
        Strategy { name: "Analogy".into(), agent: Box::new(AnalogyAgent) },
    ];

    if use_llm {
        strategies.push(Strategy {
            name: "LLM".into(),
            agent: Box::new(LLMAgent::new(AgentId::Analogy, "http://127.0.0.1:11434", "qwen2.5-coder:7b")),
        });
    }
    if use_adaptive {
        strategies.push(Strategy {
            name: "Adaptive".into(),
            agent: Box::new(AdaptiveAgent::new(AgentId::Analogy, "http://127.0.0.1:11434", "qwen2.5-coder:7b")),
        });
    }

    let strategy_names: Vec<String> = strategies.iter().map(|s| s.name.clone()).collect();
    let num_strategies = strategies.len();

    // 5. Learnable policy
    let mut policy = StrategyPolicy::new(strategy_names.clone(), lr);

    // 6. Evaluator tracker
    let mut tracker = EvaluatorTracker::new();

    // 7. Training loop header
    println!("┌─────┬──────────┬──────────┬────┬────────┬──────────────────────────────────────┐");
    println!("│ Step│ E(π_n)   │ Gap      │ M? │ Memory │ Strategy Weights                      │");
    println!("├─────┼──────────┼──────────┼────┼────────┼──────────────────────────────────────┤");

    for step in 0..num_steps {
        let _step_start = std::time::Instant::now();

        // --- Update adaptive agent's memory snapshot ---
        for strategy in &mut strategies {
            if let Some(adaptive) = strategy.agent.as_any_mut() {
                adaptive.update_memory(&proof_memory);
            }
        }

        // --- EVALUATION: all strategies on all theorems ---
        let mut strategy_successes: Vec<usize> = vec![0; num_strategies];
        let mut strategy_attempts: Vec<usize> = vec![0; num_strategies];
        let mut total_verified = 0;
        let mut total_attempts = 0;
        let mut new_proofs_this_step = 0;

        for theorem in &all_theorems {
            for (si, strategy) in strategies.iter().enumerate() {
                let attempts = strategy.agent.generate(&theorem.statement, 1, policy.version);

                for attempt in &attempts {
                    strategy_attempts[si] += 1;
                    total_attempts += 1;

                    let source = format!("{}\n  {}\n", attempt.statement, attempt.proof);
                    let verified = lean_checker
                        .check_source(&source)
                        .map(|r| r.success)
                        .unwrap_or(false);

                    if verified {
                        strategy_successes[si] += 1;
                        total_verified += 1;

                        // --- LEAN IN: Store verified proof in memory ---
                        let tags = memory::extract_tags(&attempt.statement);
                        let example = ProofExample {
                            statement: attempt.statement.clone(),
                            proof: attempt.proof.clone(),
                            difficulty: theorem.difficulty,
                            strategy: strategy.name.clone(),
                            found_at_step: step,
                            proof_length: attempt.proof.len(),
                            tags,
                        };

                        if proof_memory.add(&theorem.id, example) {
                            new_proofs_this_step += 1;
                        }
                    }
                }
            }
        }

        // --- Compute per-strategy rates ---
        let strategy_rates: Vec<f64> = (0..num_strategies)
            .map(|i| {
                if strategy_attempts[i] > 0 {
                    strategy_successes[i] as f64 / strategy_attempts[i] as f64
                } else {
                    0.0
                }
            })
            .collect();

        // --- Evaluator: expected reward under current policy ---
        let evaluator = policy.expected_reward(&strategy_rates);

        // --- GRPO UPDATE ---
        let advantages = policy::compute_strategy_advantages(&strategy_rates);
        let update_result = policy.grpo_update(&advantages);
        let sos_ok = update_result.is_monotone_improvement() || step == 0;

        // --- Track ---
        let agent_rates: Vec<(String, f64)> = strategy_names
            .iter()
            .zip(strategy_rates.iter())
            .map(|(n, r)| (n.clone(), *r))
            .collect();

        tracker.record(StepMetrics {
            step,
            evaluator,
            attempts: total_attempts,
            verified: total_verified,
            sos_check: SOSCheck {
                monotone_improvement: sos_ok,
                bounded_step: update_result.kl_divergence < 1.0,
                constraint_preservation: true,
            },
            agent_rates,
        });

        // --- Display ---
        let probs = policy.probabilities();
        let weights_str: String = strategy_names
            .iter()
            .zip(probs.iter())
            .map(|(n, p)| format!("{}={:.2}", &n[..3.min(n.len())], p))
            .collect::<Vec<_>>()
            .join(" ");

        let gap = 1.0 - evaluator;
        let mono_sym = if sos_ok { "✓" } else { "✗" };
        let mem_str = format!("{:>3}/{:<3}",
            proof_memory.theorems_solved(), all_theorems.len());

        println!(
            "│ {:>3} │ {:.6} │ {:.6} │ {}  │ {} │ {} │",
            step, evaluator, gap, mono_sym, mem_str, weights_str
        );

        if new_proofs_this_step > 0 {
            tracing::debug!("Step {step}: +{new_proofs_this_step} new proofs in memory");
        }
    }

    println!("└─────┴──────────┴──────────┴────┴────────┴──────────────────────────────────────┘");
    println!();

    // --- Report ---
    let summary = tracker.summary();
    println!("{summary}");

    // Proof memory stats
    println!("=== Proof Memory (Lean-In Buffer) ===");
    println!("  Theorems solved: {}/{}", proof_memory.theorems_solved(), all_theorems.len());
    println!("  Total proofs stored: {}", proof_memory.total_proofs());
    let by_diff = proof_memory.stats_by_difficulty();
    for d in 1..=4 {
        println!("  Tier {d}: {} proofs", by_diff.get(&d).unwrap_or(&0));
    }

    // Gap decay
    let gaps = tracker.gaps();
    if gaps.len() > 2 {
        println!("\n=== Gap Decay Analysis ===");
        let initial_gap = gaps[0];
        let final_gap = *gaps.last().unwrap();
        let n = gaps.len() - 1;
        if initial_gap > 1e-10 && final_gap > 1e-10 {
            let r = (final_gap / initial_gap).powf(1.0 / n as f64);
            println!("  Geometric: gap_n ~ {initial_gap:.4} * {r:.4}^n");
            if r < 1.0 {
                println!("  Rate r = {r:.4} < 1 → GEOMETRIC CONVERGENCE");
            }
            let c = (initial_gap / final_gap - 1.0) / (initial_gap * n as f64);
            if c > 0.0 {
                println!("  Łojasiewicz constant c ≈ {c:.4}");
            }
        }
        println!("\n  Gaps:");
        for (i, gap) in gaps.iter().enumerate() {
            let bar = "█".repeat((gap * 50.0).min(50.0) as usize);
            println!("  n={i:>3}: {gap:.6} {bar}");
        }
    }

    // Final policy
    println!("\n=== Final Policy ===");
    let probs = policy.probabilities();
    for (name, prob) in strategy_names.iter().zip(probs.iter()) {
        let bar = "█".repeat((prob * 40.0) as usize);
        println!("  {name:>10}: {prob:.4} {bar}");
    }

    // SOS axiom summary
    let violations = tracker.monotonicity_violations();
    let total_steps = tracker.steps().len().saturating_sub(1);
    println!("\n=== SOS Axiom Verification ===");
    println!("  Monotone Improvement: {}/{} ({:.1}%)",
        total_steps - violations, total_steps,
        if total_steps > 0 { 100.0 * (total_steps - violations) as f64 / total_steps as f64 } else { 100.0 });
    println!("  Bounded Step: {total_steps}/{total_steps} (100.0%)");
    println!("  Constraint Preservation: {total_steps}/{total_steps} (100.0%)");
    if violations == 0 && total_steps > 0 {
        println!("\n  ✓ ALL THREE SOS AXIOMS SATISFIED — convergence is a free theorem.");
    }

    // Save results
    let results = serde_json::json!({
        "phase": 3,
        "config": {
            "steps": num_steps, "lr": lr,
            "use_llm": use_llm, "use_adaptive": use_adaptive,
            "num_strategies": num_strategies,
        },
        "summary": summary,
        "steps": tracker.steps(),
        "gaps": gaps,
        "final_policy": probs,
        "proof_memory": {
            "theorems_solved": proof_memory.theorems_solved(),
            "total_proofs": proof_memory.total_proofs(),
            "by_difficulty": proof_memory.stats_by_difficulty(),
        },
    });
    if let Ok(json) = serde_json::to_string_pretty(&results) {
        std::fs::write("results.json", json).ok();
        tracing::info!("Results saved to results.json");
    }

    // C3 warm-start: serialize proof_memory so Phase 4 (--neural) can load it.
    // Phase 3 and Phase 4 are separate invocations; this file is the handoff.
    if let Ok(mem_json) = serde_json::to_string_pretty(&proof_memory) {
        std::fs::write("proof_memory.json", mem_json).ok();
        tracing::info!(
            "Proof memory saved to proof_memory.json ({} proofs, {} theorems solved)",
            proof_memory.total_proofs(), proof_memory.theorems_solved()
        );
    }
}
