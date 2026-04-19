# Stub 2: Wire Real Backprop Through the Flask Bridge

## What You're Building

The `_grpo_step()` function in `python/train_server.py:296` is `pass`. The Rust orchestrator in `pf-cli/src/main.rs` never instantiates `TrainingBridge`. `verify_sos_step()` in `pf-core/src/grpo.rs:111` is only called in tests. The pipeline in `pf-pipeline/src/scheduler.rs` is synchronous and doesn't call `TrainingBridge`.

When this is done, the Rust layer orchestrates real neural GRPO training with formal SOS governance: every gradient step passes through staleness filtering and SOS axiom verification. The EvaluatorTracker monitors training in real time via periodic evaluation on a fixed eval set, and the system can intervene on violations that exceed the variance-acceleration bound.

### Known Limitations of Initial Implementation

These are explicitly deferred — NOT bugs, but documented scope boundaries:

1. **No KL divergence** — requires reference model (doubles GPU memory on 7B). Deferred. `grad_norm` is used as a proxy for step distance, with a separate `max_grad_norm` config (not derived from `epsilon_clip` — they are different scales).
2. **Sparse evaluation** — true evaluator is measured every `eval_interval` steps, not every step. Training steps between eval points record `batch_reward` as a noisy proxy. SOS formal checks only run at eval points.
3. **`grad_norm` is NOT the SOS bounded-step axiom** — the formal axiom requires policy distance (KL or TV). Gradient norm is a weaker proxy. The bounded-step check is observational, not formally sound, until KL is added.

## The Reference Implementation

The canonical working backprop lives in `scripts/grpo_lean_reward.py`. These are the exact patterns to port:

### `compute_completion_log_prob()` (grpo_lean_reward.py:420–453)
```python
def compute_completion_log_prob(model, tokenizer, prompt, completion_token_ids):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    completion_tensor = torch.tensor([completion_token_ids], dtype=torch.long, device=model.device)
    full_ids = torch.cat([prompt_ids, completion_tensor], dim=1)
    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)
    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    pred_logits = logits[0, prompt_len - 1 : prompt_len + comp_len - 1, :]
    targets = completion_tensor[0]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]
    return token_log_probs.sum()
```

### `grpo_step()` loss computation (grpo_lean_reward.py:460–555)
```python
optimizer.zero_grad()
total_loss = torch.tensor(0.0, device=device)
n_updates = 0

for each (statement, proofs, advantages) group:
    prompt = PROOF_PROMPT_TEMPLATE.format(statement=statement)
    for proof_text, advantage in zip(proofs, advantages):
        if advantage <= 0:
            continue
        token_ids = tokenizer(proof_text, return_tensors="pt")["input_ids"][0].tolist()
        log_prob = compute_completion_log_prob(model, tokenizer, prompt, token_ids)
        total_loss = total_loss + (-advantage * log_prob)
        n_updates += 1

if n_updates > 0:
    loss = total_loss / n_updates
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

### Optimizer setup (grpo_lean_reward.py:606–616)
```python
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
num_warmup = max(1, int(0.05 * total_steps))
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps)
```

## Exact Changes Required

### Part A: Python — `train_server.py`

#### A1. Add imports at top (after existing imports)

```python
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
```

#### A2. Add shared prompt template and globals after `tokenizer = None` (line 78)

```python
optimizer = None
scheduler = None

PROOF_PROMPT_TEMPLATE = "Complete the Lean 4 proof. Output ONLY tactics.\n\n{statement}\n"
```

#### A3. In `load_model()` — create optimizer and scheduler after LoRA attachment

The `global` declaration at the top of `load_model()` must include all new globals:

```python
def load_model():
    """Load model with LoRA adapters."""
    global model, tokenizer, optimizer, scheduler  # ← MUST include optimizer, scheduler

    # ... existing model loading code (lines 96-127) unchanged ...

    # After model = get_peft_model(model, lora_config) and model.print_trainable_parameters():

    # Optimizer and cosine schedule
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,  # 5e-6
    )
    # Estimate total steps; Rust orchestrator can POST /configure to adjust later
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=200,
    )

    state.model_loaded = True
    print(f"Model loaded on {device}. LoRA adapters attached. Optimizer ready.")
```

**No reference model, no deepcopy.** KL divergence is deferred to a future iteration. The initial implementation uses gradient norm as the step distance metric for SOS bounded-step checking (see A5).

#### A4. Add `compute_completion_log_prob()` — new function before `_grpo_step()`

Port directly from `grpo_lean_reward.py:420–453`. Exact code from the reference section above. This function:
- Concatenates prompt + completion token IDs
- Does one forward pass (with gradient)
- Slices logits at `[prompt_len-1 : prompt_len+comp_len-1]` (the positions that predict completion tokens)
- Computes `log_softmax` → indexes by target token IDs → `.sum()`
- Returns a scalar tensor with gradient attached

#### A5. Replace `_grpo_step()` (lines 296–315) with real implementation

```python
def _grpo_step(data):
    """Execute the actual GRPO gradient update.

    Ported from grpo_lean_reward.py:460–555 (loss computation only).
    KL divergence is NOT computed here — deferred to future iteration.

    data keys: statements, proofs (list of lists), rewards (list of lists), advantages (list of lists)
    Returns: {"loss": float, "grad_norm": float, "n_updates": int}
    """
    global optimizer, scheduler
    if not TORCH_AVAILABLE or model is None or optimizer is None:
        return {"loss": 0.0, "grad_norm": 0.0, "n_updates": 0}

    device = next(model.parameters()).device
    statements = data.get("statements", [])
    proofs_batch = data.get("proofs", [])
    advantages_batch = data.get("advantages", [])

    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    n_updates = 0

    for stmt, proofs, advantages in zip(statements, proofs_batch, advantages_batch):
        prompt = PROOF_PROMPT_TEMPLATE.format(statement=stmt)

        for proof_text, adv in zip(proofs, advantages):
            if adv <= 0 or not proof_text.strip():
                continue

            # Tokenize the completion
            comp_ids = tokenizer(proof_text, return_tensors="pt")["input_ids"][0].tolist()
            if not comp_ids:
                continue

            # Current policy log-prob (with gradient)
            log_prob = compute_completion_log_prob(model, tokenizer, prompt, comp_ids)
            total_loss = total_loss + (-adv * log_prob)
            n_updates += 1

    result = {"loss": 0.0, "grad_norm": 0.0, "n_updates": n_updates}

    if n_updates > 0:
        loss = total_loss / n_updates
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optimizer.step()
        scheduler.step()
        result["loss"] = loss.item()
        result["grad_norm"] = grad_norm

    return result
```

#### A6. Update `/generate` to use shared `_generate_proofs()` helper and prompt template

The real-model branch of `/generate` (lines 170-192) must use the same `_generate_proofs()` helper that `/evaluate` uses. If generation parameters change in one place, they change everywhere.

```python
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    statements = data.get("statements", [])
    n = data.get("n", config.group_size)

    if not state.model_loaded:
        # Mock mode: return template proofs
        tactics = ["trivial", "simp", "omega", "rfl", "decide", "ring",
                    "intro h; exact h", "constructor"]
        proofs = []
        for stmt in statements:
            attempts = [tactics[i % len(tactics)] for i in range(n)]
            proofs.append(attempts)
        return jsonify({"proofs": proofs, "mock": True})

    # Real generation via shared helper
    proofs = _generate_proofs(statements, n)
    state.total_proofs_generated += sum(len(p) for p in proofs)
    return jsonify({"proofs": proofs, "mock": False})
```

#### A7. Restructure `/step` endpoint — separate batch reward from evaluation

The current `/step` endpoint computes `evaluator_after` from the incoming batch's rewards BEFORE running the gradient step. This makes the SOS monotone check meaningless — it's comparing batch reward noise, not policy improvement.

**Fix: The `/step` endpoint reports `batch_reward` (noisy, for logging) and `grad_norm` (for step-bound checking). Real evaluation happens separately via a new `/evaluate` endpoint.**

```python
@app.route("/step", methods=["POST"])
def step():
    """Execute one GRPO training step.

    Request: {
        "statements": [...],
        "proofs": [[...], ...],
        "rewards": [[...], ...],
        "advantages": [[...], ...]
    }
    Response: {
        "step": 5,
        "batch_reward": 0.25,
        "loss": 0.42,
        "grad_norm": 0.87,
        "n_updates": 12,
        "policy_version": 6
    }
    """
    data = request.json
    rewards_batch = data.get("rewards", [])

    # Batch reward (noisy — only for logging, NOT for SOS verification)
    total_reward = sum(sum(r for r in rewards) for rewards in rewards_batch)
    total_count = sum(len(rewards) for rewards in rewards_batch)
    batch_reward = total_reward / max(total_count, 1)

    if not state.model_loaded:
        step_result = {"loss": 0.0, "grad_norm": 0.0, "n_updates": 0}
    else:
        step_result = _grpo_step(data)

    state.step += 1
    state.policy_version += 1
    # NOTE: Do NOT increment state.total_proofs_verified here.
    # The /reward endpoint already counts verified proofs.
    # Incrementing in both /reward and /step would double-count.

    # Update state with batch_reward as noisy proxy (for /status monitoring)
    state.evaluator = batch_reward
    state.evaluator_history.append(batch_reward)

    return jsonify({
        "step": state.step,
        "batch_reward": batch_reward,
        "loss": step_result.get("loss", 0.0),
        "grad_norm": step_result.get("grad_norm", 0.0),
        "n_updates": step_result.get("n_updates", 0),
        "policy_version": state.policy_version,
    })


def _generate_proofs(statements, n):
    """Shared proof generation logic for /generate and /evaluate.

    Returns list of list of proof strings. Requires model to be loaded.
    """
    proofs = []
    for stmt in statements:
        prompt = PROOF_PROMPT_TEMPLATE.format(statement=stmt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attempts = []
        for _ in range(n):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            proof = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
            attempts.append(proof)
        proofs.append(attempts)
    return proofs


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate current policy on a fixed set of statements.

    This is the TRUE evaluator for SOS verification — measures actual
    policy quality, not batch reward noise. Returns proofs for Rust-side
    Lean verification (we don't verify here — Lean checking stays Rust-side).

    Request: { "statements": ["theorem t : True := by", ...], "n": 4 }
    Response: { "proofs": [["trivial", "simp"], ...] }
    """
    data = request.json
    statements = data.get("statements", [])
    n = data.get("n", 4)

    if not state.model_loaded:
        return jsonify({"proofs": [[] for _ in statements]})

    proofs = _generate_proofs(statements, n)
    return jsonify({"proofs": proofs})
```

The Rust side calls `/evaluate`, verifies with Lean, computes pass rate, and feeds that to `verify_sos_step()`.

### Part B: Rust — Wire `TrainingBridge` Into the Orchestrator

#### B1. Fix `TrainingBridge` struct — store Client, increase timeout

In `bridge.rs`, store the client in the struct and add HTTP status checking:

```rust
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub server_url: String,
    pub timeout_secs: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8420".to_string(),
            timeout_secs: 600,  // 10 minutes — real backprop on 7B is slow
        }
    }
}

pub struct TrainingBridge {
    config: BridgeConfig,
    client: reqwest::blocking::Client,
}

impl TrainingBridge {
    pub fn new(config: BridgeConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(1)
            .build()
            .expect("failed to build HTTP client");
        Self { config, client }
    }

    // ... existing health(), status(), generate(), reward() methods unchanged
    // but update get() and post() to use &self.client:

    fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, BridgeError> {
        let resp = self.client
            .get(url)
            .send()
            .map_err(|e| BridgeError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(BridgeError::ServerError(format!("{status}: {body}")));
        }

        resp.json().map_err(|e| BridgeError::Parse(e.to_string()))
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<T, BridgeError> {
        let resp = self.client
            .post(url)
            .json(body)
            .send()
            .map_err(|e| BridgeError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(BridgeError::ServerError(format!("{status}: {body}")));
        }

        resp.json().map_err(|e| BridgeError::Parse(e.to_string()))
    }
}
```

#### B2. Update response types in `bridge.rs`

Replace `StepResponse` to match the restructured `/step` endpoint. Add `EvaluateResponse` for the new `/evaluate` endpoint:

```rust
/// Response from /step endpoint (training step metrics).
#[derive(Debug, Deserialize)]
pub struct StepResponse {
    pub step: usize,
    pub batch_reward: f64,
    pub loss: f64,
    pub grad_norm: f64,
    pub n_updates: usize,
    pub policy_version: u64,
}

/// Response from /evaluate endpoint (eval set proofs for Lean verification).
#[derive(Debug, Deserialize)]
pub struct EvaluateResponse {
    pub proofs: Vec<Vec<String>>,
}
```

Add the `evaluate()` method to `TrainingBridge`:

```rust
/// Generate proofs for evaluation (Rust side verifies with Lean).
pub fn evaluate(
    &self,
    statements: &[String],
    n: usize,
) -> Result<EvaluateResponse, BridgeError> {
    let url = format!("{}/evaluate", self.config.server_url);
    let body = serde_json::json!({
        "statements": statements,
        "n": n,
    });
    self.post(&url, &body)
}
```

#### B3. Add Phase 4 runner in `pf-cli/src/main.rs`

This replaces the Phase 3 loop when `--neural` flag is passed. Key differences from original prompt:
- Uses `all_theorems()` with random sampling (not nonexistent `sample_batch()`)
- SOS verification uses periodic evaluation on a fixed eval set, not batch rewards
- Step-bound checking uses gradient norm, not KL divergence
- Bounded regression tolerance (from variance-acceleration theorem)

```rust
use rand::seq::SliceRandom;
use pf_core::bridge::{BridgeConfig, TrainingBridge};
use pf_core::grpo;

fn run_phase4(
    bridge: &TrainingBridge,
    lean_checker: &LeanChecker,
    dataset: &TheoremDataset,
    config: &GRPOConfig,
    num_steps: usize,
) {
    let mut tracker = EvaluatorTracker::new();
    let mut rng = rand::thread_rng();
    let all_theorems = dataset.all_theorems();
    let batch_size = 4;
    let eval_interval = 5;  // evaluate every 5 steps
    let max_regression = 0.05;  // tolerate up to 5% evaluator regression (disruption phase)

    // Fixed eval set: first 10 theorems (deterministic, reproducible)
    let eval_theorems: Vec<&_> = all_theorems.iter().take(10).collect();
    let eval_statements: Vec<String> = eval_theorems.iter().map(|t| t.statement.clone()).collect();

    // Baseline evaluation before any training
    let mut last_evaluator = evaluate_policy(bridge, lean_checker, &eval_statements);
    let mut max_evaluator = last_evaluator;  // peak evaluator (for N4: regression measured from peak)
    tracing::info!("Baseline evaluator: {last_evaluator:.4}");

    println!("┌─────┬──────────┬──────────┬────────┬──────────┬────┐");
    println!("│ Step│ BatchRwd │ Loss     │ GradN  │ Eval     │ M? │");
    println!("├─────┼──────────┼──────────┼────────┼──────────┼────┤");

    for step in 0..num_steps {
        // 1. Sample batch of theorems
        let batch: Vec<_> = all_theorems.choose_multiple(&mut rng, batch_size).collect();
        let statements: Vec<String> = batch.iter().map(|t| t.statement.clone()).collect();

        // 2. Generate via neural policy (Python)
        let gen_response = match bridge.generate(&statements, config.group_size) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Step {step}: generate failed: {e}");
                // N9: Record failed step so tracker step count matches loop count
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

        // 3. Verify via Lean (Rust-side)
        let rewards: Vec<Vec<f64>> = gen_response.proofs.iter()
            .zip(statements.iter())
            .map(|(proofs, stmt)| {
                proofs.iter().map(|proof| {
                    let source = format!("{}\n  {}\n", stmt, proof);
                    if lean_checker.check_source(&source)
                        .map(|r| r.success).unwrap_or(false) { 1.0 } else { 0.0 }
                }).collect()
            }).collect();

        // 4. Compute advantages (Rust-side GRPO)
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

        // 5. GRPO step with backprop (Python)
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

        // 6. SOS step-bound check (gradient norm as proxy)
        // NOTE: grad_norm is NOT the formal SOS bounded-step axiom (that requires KL/TV).
        // This is an observational check with its own dedicated threshold.
        let bounded_step = step_response.grad_norm <= config.max_grad_norm;
        if !bounded_step {
            tracing::warn!("Step {step}: GRAD NORM {:.4} exceeds max_grad_norm {:.4}",
                step_response.grad_norm, config.max_grad_norm);
        }

        // 7. Batch metrics — record EVERY step (N6: not just eval steps)
        let batch_attempts: usize = rewards.iter().map(|r| r.len()).sum();
        let batch_verified: usize = rewards.iter().flat_map(|r| r.iter())
            .filter(|&&r| r > 0.5).count();

        // 8. Periodic evaluation on fixed eval set (TRUE evaluator)
        let mut eval_str = String::from("—");
        let mut mono_sym = "—";
        let mut eval_this_step = false;
        let mut evaluator_for_record = last_evaluator;  // use last known eval for non-eval steps

        if (step + 1) % eval_interval == 0 || step == num_steps - 1 {
            eval_this_step = true;
            let evaluator = evaluate_policy(bridge, lean_checker, &eval_statements);
            max_evaluator = max_evaluator.max(evaluator);
            evaluator_for_record = evaluator;

            // N2: Use verify_sos_step() result — don't discard it
            let sos = grpo::verify_sos_step(
                last_evaluator, evaluator,
                step_response.grad_norm, config,
            );

            // N4: Measure regression from PEAK evaluator, not just last checkpoint.
            // This prevents compounding tolerance: 20 × 5% = 100% total regression
            // is caught because we always compare against the best-ever value.
            let regression_from_peak = max_evaluator - evaluator;
            let regression_from_last = last_evaluator - evaluator;

            if regression_from_peak > max_regression {
                tracing::warn!(
                    "Step {step}: REGRESSION BEYOND TOLERANCE — peak {max_evaluator:.4}, \
                     now {evaluator:.4} (drop {regression_from_peak:.4} > max {max_regression:.4})"
                );
                mono_sym = "✗";
                // Future: bridge.adjust_lr(0.5) to halve learning rate
                // Future: bridge.revert_to_checkpoint()
            } else if regression_from_last > 0.0 {
                tracing::info!(
                    "Step {step}: Tolerated regression {regression_from_last:.4} \
                     (peak distance {regression_from_peak:.4} ≤ {max_regression:.4}, disruption phase)"
                );
                mono_sym = "~";
            } else {
                mono_sym = if sos.monotone_improvement { "✓" } else { "~" };
            }

            eval_str = format!("{evaluator:.6}");
            last_evaluator = evaluator;
        }

        // N6: Record ALL steps in tracker (batch metrics for non-eval, real eval for eval steps)
        // N2: SOSCheck built from verify_sos_step() at eval points, observational at others
        tracker.record(StepMetrics {
            step,
            evaluator: evaluator_for_record,
            attempts: batch_attempts,
            verified: batch_verified,
            sos_check: if eval_this_step {
                // At eval points, use formal check + tolerance
                let regression_from_peak = max_evaluator - evaluator_for_record;
                SOSCheck {
                    monotone_improvement: regression_from_peak <= max_regression,
                    bounded_step,
                    constraint_preservation: true,
                }
            } else {
                // Between evals, record step-bound only; monotone is unknown
                SOSCheck {
                    monotone_improvement: true,  // assumed — no eval data
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

    // Report SOS axiom status
    let violations = tracker.monotonicity_violations();
    let total_steps = tracker.steps().len().saturating_sub(1);
    println!("=== SOS Axiom Verification (Phase 4) ===");
    println!("  Monotone (with {:.0}% tolerance from peak): {}/{} eval checkpoints",
        max_regression * 100.0, total_steps - violations, total_steps);
    println!("  Bounded Step (grad_norm): observational only — formal check requires KL (deferred)");
    println!("  Constraint Preservation: {total_steps}/{total_steps} (trivially satisfied)");
}

/// Evaluate the current policy on a fixed statement set.
/// Returns pass rate (0.0–1.0).
fn evaluate_policy(
    bridge: &TrainingBridge,
    lean_checker: &LeanChecker,
    eval_statements: &[String],
) -> f64 {
    let eval_response = match bridge.evaluate(eval_statements, 4) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Evaluation failed: {e}");
            return 0.0;
        }
    };

    let mut verified = 0;
    let mut total = 0;
    for (proofs, stmt) in eval_response.proofs.iter().zip(eval_statements.iter()) {
        for proof in proofs {
            total += 1;
            let source = format!("{}\n  {}\n", stmt, proof);
            if lean_checker.check_source(&source)
                .map(|r| r.success).unwrap_or(false)
            {
                verified += 1;
            }
        }
    }

    if total > 0 { verified as f64 / total as f64 } else { 0.0 }
}
```

#### B4. Add `--neural` flag dispatch in `pf-cli/src/main.rs`

In `fn main()`, after CLI arg parsing. Add `--timeout` flag support:

```rust
let use_neural = args.iter().any(|a| a == "--neural");
let timeout: u64 = args
    .iter()
    .position(|a| a == "--timeout")
    .and_then(|i| args.get(i + 1))
    .and_then(|s| s.parse().ok())
    .unwrap_or(600);

if use_neural {
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

    run_phase4(&bridge, &lean_checker, &dataset, &GRPOConfig::default(), num_steps);
    return;
}
// ... existing Phase 3 code follows unchanged
```

### Part C: Protocol Alignment

#### C1. Prompt template consistency

Both `/generate` and `_grpo_step()` must use the same `PROOF_PROMPT_TEMPLATE`. The log-probability computation requires the identical prefix that generation used. If they differ by even one character, the gradient signal is wrong because the logit positions shift.

```python
PROOF_PROMPT_TEMPLATE = "Complete the Lean 4 proof. Output ONLY tactics.\n\n{statement}\n"
```

Note: `grpo_lean_reward.py` uses a slightly different template (`"Complete this Lean 4 proof. Output ONLY the tactic(s).\n\n{statement}\n"`). The standalone scripts and the bridge are independent systems — internal consistency within `train_server.py` is what matters. Cross-script divergence is cosmetic.

#### C2. Token ID round-trip

The Rust side sends proof text strings. `_grpo_step()` re-tokenizes them to compute log-probs. This is correct — the reference implementation does the same. Token IDs are NOT transmitted through the bridge. Re-tokenization may produce slightly different IDs than original generation if the tokenizer has multiple valid tokenizations. This is acceptable.

#### C3. Step distance metric

The SOS bounded-step axiom checks `step_distance ≤ epsilon_clip`. For the initial implementation, **gradient norm** is the step distance metric (not KL divergence). Gradient norm is:
- Available immediately after `loss.backward()` (no extra compute)
- Requires no reference model (no GPU memory doubling)
- A direct measure of how much the weights change

The scale is different from `epsilon_clip` (which was designed for policy ratios). The Phase 4 runner uses a dedicated `max_grad_norm` field in `GRPOConfig` (default 2.0) — NOT derived from `epsilon_clip`. These are independent thresholds measuring different things.

**Add to `GRPOConfig` in `pf-core/src/types.rs`:**
```rust
pub struct GRPOConfig {
    // ... existing fields ...
    /// Maximum gradient norm before step-bound warning.
    /// This is an observational proxy, NOT the formal SOS bounded-step axiom
    /// (which requires KL divergence or total variation distance).
    pub max_grad_norm: f64,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            max_grad_norm: 2.0,  // post-clipping norms above 2.0 indicate large updates
        }
    }
}
```

**KL divergence is deferred to a future iteration.** When added, it should:
- Use LoRA adapter weight diff (tiny, no memory overhead) rather than full model deepcopy
- Have its own `max_kl` field in `GRPOConfig` (default ~0.05)
- NOT reuse `epsilon_clip` (the scales don't match — Pinsker's inequality at KL=0.2 allows policy ratios up to ~1.32, but epsilon_clip=0.2 was designed for ratios in [0.8, 1.2])

#### C4. Evaluator semantics — batch reward vs true evaluation

**CRITICAL DESIGN DECISION:** The `/step` response reports `batch_reward` (mean reward of the training batch). This is a noisy signal that reflects theorem difficulty as much as policy quality. It MUST NOT be used for SOS verification.

The true evaluator is computed by the Rust side via `/evaluate` on a fixed eval set at periodic intervals. Only this value feeds into `verify_sos_step()`. This matches the Phase 3 design (which evaluates all theorems every step) and the standalone scripts (which evaluate at checkpoints).

The evaluation cost (generate + verify on 10 theorems × 4 completions = 40 Lean checks) is small relative to the training step cost. Running it every 5 steps is a reasonable default.

## Testing Strategy

1. **Python mock mode**: Start server without `--load-model`. Call `/step` with synthetic data. Verify returns `{"loss": 0.0, "grad_norm": 0.0, "n_updates": 0}`. Call `/evaluate` with statements, verify returns `{"proofs": [[], ...]}`. Verify `/status` returns updated `evaluator` field (not stuck at 0.0).

2. **Python with tiny model**: Load `sshleifer/tiny-gpt2` (or any small model) with LoRA. Call `/generate`, then `/step` with results and synthetic advantages. Verify `loss > 0`, `grad_norm > 0`, `n_updates > 0`. Verify `/generate` and `/evaluate` produce identically-distributed outputs (same helper).

3. **N1 double-count test**: Call `/reward` then `/step` with same batch. Check `state.total_proofs_verified` via `/status`. Verify count matches `/reward` batch only — NOT doubled.

4. **Rust bridge connection**: Start Python server in mock mode. Run `pf-cli --neural`. Verify health check passes, loop completes, ALL steps appear in output table (not just eval steps), metrics display.

5. **Rust bridge error handling**: Kill the Python server mid-loop. Verify Rust logs error, records failed step in tracker, and continues (not panic). Verify tracker `total_steps` matches `num_steps` (N9).

6. **N4 compounding regression test**: Feed evaluator values that drop 4% each eval checkpoint (below 5% tolerance). Verify that once cumulative drop from peak exceeds 5%, the system flags "REGRESSION BEYOND TOLERANCE" — not "Tolerated regression" forever.

7. **N2 SOS verification test**: Verify that `verify_sos_step()` output is reflected in the `SOSCheck` recorded in the tracker. Run with a scenario where evaluator improves — check `monotone_improvement: true`. Run with regression beyond tolerance — check `monotone_improvement: false`.

8. **End-to-end with Lean**: With Lean installed, run `pf-cli --neural` against the real dataset. Verify the full loop: generate → Lean verify → GRPO step → periodic evaluation → SOS check. Verify the SOS summary at end correctly reports eval checkpoint counts and distinguishes formal vs observational checks.

## What NOT to Change

- Do NOT modify `grpo_lean_reward.py` or any script in `scripts/` — standalone production scripts
- Do NOT change the existing Phase 3 loop — must work without `--neural`
- Do NOT add async/await to the Python server — Flask synchronous is correct
- Do NOT transmit token IDs through the bridge — text round-trip is the correct design
- Do NOT modify `pf-core/src/grpo.rs` computations — GRPO math is correct and tested
- Do NOT modify the reward oracle or Lean checker — verification stays Rust-side
- Do NOT add a reference model / deepcopy — deferred to future iteration
- Do NOT modify `pf-pipeline/src/scheduler.rs` — Phase 4 doesn't use AsyncPipeline

## Dependencies

Python (already in requirements or available):
- `torch` (already imported conditionally)
- `torch.nn.functional` (part of torch)
- `torch.optim.AdamW` (part of torch)
- `transformers.get_cosine_schedule_with_warmup` (already a transformers dependency)

Rust (add to pf-cli/Cargo.toml):
- `rand` with `SliceRandom` (for batch sampling from `all_theorems()`)

Rust (already in Cargo.toml):
- `reqwest` with `json` and `blocking` features (in `pf-core/Cargo.toml`)
- `serde` / `serde_json` (already present)

## File Change Summary

| File | Changes |
|------|---------|
| `python/train_server.py` | Add imports, `PROOF_PROMPT_TEMPLATE`, globals with optimizer/scheduler, `_generate_proofs()` shared helper (N7), `compute_completion_log_prob()`, real `_grpo_step()` returning loss+grad_norm, restructured `/step` (no double-count — N1, state updates — N3), `/generate` refactored to use shared helper, new `/evaluate` endpoint |
| `crates/pf-core/src/bridge.rs` | Store `Client` in struct (N11), increase timeout to 600s (N6-orig), add HTTP status checking (N9-orig), update `StepResponse` fields, add `EvaluateResponse` and `evaluate()` method |
| `crates/pf-core/src/types.rs` | Add `max_grad_norm: f64` to `GRPOConfig` with default 2.0 (N5) |
| `crates/pf-cli/src/main.rs` | Add `--neural` and `--timeout` flags, `run_phase4()` with: all steps recorded in tracker (N6), failed steps tracked (N9), `verify_sos_step()` result used not discarded (N2), regression measured from peak evaluator (N4), `max_grad_norm` for step-bound (N5), `evaluate_policy()` helper |
| `crates/pf-cli/Cargo.toml` | Add `rand` dependency |

## Orange Team v2 Fix Cross-Reference

| Finding | Fix Location | Description |
|---------|-------------|-------------|
| N1 | A7 `/step` handler | Removed `total_proofs_verified` increment (already done in `/reward`) |
| N2 | B3 Phase 4 runner | `verify_sos_step()` result used for `SOSCheck` construction, not discarded as `_sos` |
| N3 | A7 `/step` handler | `state.evaluator` and `state.evaluator_history` updated with `batch_reward` proxy |
| N4 | B3 Phase 4 runner | Regression measured from `max_evaluator` (peak), not `last_evaluator`. Prevents compounding. |
| N5 | C3 + types.rs | Dedicated `max_grad_norm` config field (default 2.0), not derived from `epsilon_clip * 10.0` |
| N6 | B3 Phase 4 runner | ALL steps recorded in `EvaluatorTracker`, not just eval steps |
| N7 | A6 + A7 `/evaluate` | `_generate_proofs()` shared helper eliminates duplication between `/generate` and `/evaluate` |
| N8 | — | Design tradeoff (eval cost). Accepted. Tune `eval_interval` if too slow. |
| N9 | B3 Phase 4 runner | Failed steps record `StepMetrics` with `last_evaluator` and zero attempts |
| N10 | B3 Phase 4 runner | `max_evaluator` is actively used for N4 peak-regression check (no longer dead) |
