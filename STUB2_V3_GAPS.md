# Stub 2 v3: Remaining Gaps — Verification Honesty & Naming Clarity

## Context

Stub 2 (real backprop through the Flask bridge) has been implemented and reviewed through two Orange Team rounds. The core training loop works: `_grpo_step()` computes real gradients, the Rust orchestrator drives the loop via `TrainingBridge`, periodic evaluation measures true policy quality. This prompt fixes 8 remaining gaps identified in Orange Team v3 — all related to **verification honesty** (the SOS governance layer must not lie about what it checked) and **naming clarity** (competing definitions of "evaluator" confuse monitoring).

## Prerequisites

Read these files before making changes:
- `crates/pf-cli/src/main.rs` — Phase 4 runner (`run_phase4()`)
- `crates/pf-core/src/evaluator.rs` — `SOSCheck`, `StepMetrics`, `EvaluatorTracker`
- `crates/pf-core/src/grpo.rs` — `verify_sos_step()`, `SOSVerification`
- `crates/pf-core/src/types.rs` — `GRPOConfig`
- `python/train_server.py` — `TrainingState`, `/step`, `/evaluate`

## Gap 1: Verification Theater — `sos` result is tautological

### Problem

In `run_phase4()`, `verify_sos_step()` is called at the eval branch. Its result `sos` is only read at the `else` branch (the improvement case), where `regression_from_last <= 0` guarantees `evaluator >= last_evaluator`. Since `verify_sos_step()` checks `evaluator_after >= evaluator_before` with those same values, `sos.monotone_improvement` is **always true** when read. The formal check can never disagree with the branch condition.

Additionally, `sos.bounded_step` (which checks `grad_norm <= epsilon_clip = 0.2`) is **never read anywhere**. The actual bounded-step check uses `config.max_grad_norm = 2.0` instead. Two thresholds, two code paths, one is dead.

The `SOSCheck` recorded in the tracker uses `regression_from_peak <= max_regression`, not `sos.monotone_improvement`.

### Fix

Restructure the eval branch so the formal SOS result influences what's recorded, not just a display character in a dead branch.

In `run_phase4()`, replace the eval branch (the block starting at `if (step + 1) % eval_interval == 0`) with:

```rust
if (step + 1) % eval_interval == 0 || step == num_steps - 1 {
    eval_this_step = true;
    let evaluator = evaluate_policy(bridge, lean_checker, &eval_statements);
    max_evaluator = max_evaluator.max(evaluator);
    evaluator_for_record = evaluator;

    // Formal SOS check (strict axioms)
    let sos = grpo::verify_sos_step(
        last_evaluator, evaluator,
        step_response.grad_norm, config,
    );

    // Operational tolerance check (disruption phase allowance)
    let regression_from_peak = max_evaluator - evaluator;
    let regression_from_last = last_evaluator - evaluator;
    let tolerated = regression_from_peak <= max_regression;

    // Display: use the tighter of formal vs tolerated
    if !tolerated {
        tracing::warn!(
            "Step {step}: REGRESSION BEYOND TOLERANCE — peak {max_evaluator:.4}, \
             now {evaluator:.4} (drop {regression_from_peak:.4} > max {max_regression:.4})"
        );
        mono_sym = "✗";
    } else if !sos.monotone_improvement {
        // Formal axiom violated but within operational tolerance (disruption phase)
        tracing::info!(
            "Step {step}: Formal monotone violated ({last_evaluator:.4} → {evaluator:.4}), \
             tolerated (peak distance {regression_from_peak:.4} ≤ {max_regression:.4})"
        );
        mono_sym = "~";
    } else {
        mono_sym = "✓";
    }

    // Record BOTH formal and operational results
    formal_monotone_checks += 1;
    if !sos.monotone_improvement { formal_monotone_violations += 1; }
    if !sos.bounded_step { formal_bounded_violations += 1; }
    if !tolerated { tolerated_violations += 1; }

    eval_str = format!("{evaluator:.6}");
    last_evaluator = evaluator;
}
```

Add counter variables before the loop:
```rust
let mut formal_monotone_checks: usize = 0;
let mut formal_monotone_violations: usize = 0;
let mut formal_bounded_violations: usize = 0;
let mut tolerated_violations: usize = 0;
```

## Gap 2: `state.is_monotone` never updated

### Problem

`TrainingState.is_monotone` (train_server.py:69) is initialized `True` and never set to `False`. The old `/step` handler had monotone-tracking logic that was removed in the v2 rewrite. `/status` returns `is_monotone: True` regardless of actual behavior.

### Fix

Since the real monotone check is Rust-side (periodic eval), the Python side cannot meaningfully track monotonicity. **Remove the field entirely** rather than implementing a noisy proxy that could mislead.

In `TrainingState` (train_server.py:62-70):
```python
@dataclass
class TrainingState:
    """Tracks training progress. Monotone tracking is Rust-side only."""
    step: int = 0
    batch_reward: float = 0.0       # renamed from evaluator (see Gap 5)
    policy_version: int = 0
    total_proofs_generated: int = 0
    total_proofs_verified: int = 0
    batch_reward_history: list = field(default_factory=list)  # renamed from evaluator_history
    model_loaded: bool = False
    # NOTE: is_monotone removed — monotone tracking requires periodic eval
    # on a fixed set with Lean verification, which only the Rust orchestrator can do.
```

## Gap 3: Non-eval steps assume `monotone_improvement: true`

### Problem

Non-eval steps record `SOSCheck { monotone_improvement: true, ... }` — assumed passing with no eval data. `tracker.monotonicity_violations()` (evaluator.rs:85-90) uses `.windows(2).all(...)` which counts these as passing. The SOS summary reports violations out of `total_steps` (e.g., 20) when only eval transitions (e.g., 4) were actually checked.

### Fix

The SOS summary must report violations out of **eval transitions checked**, not total steps. Replace the summary block at the end of `run_phase4()`:

```rust
// Report SOS axiom status — distinguish formal from operational
println!("=== SOS Axiom Verification (Phase 4) ===");
println!("  Eval checkpoints: {} (every {} steps)", formal_monotone_checks, eval_interval);
println!();
println!("  Monotone Improvement (formal, strict E(π_n) ≥ E(π_{n-1})):");
println!("    {}/{} eval transitions passed",
    formal_monotone_checks - formal_monotone_violations, formal_monotone_checks);
println!("  Monotone Improvement (operational, {:.0}% tolerance from peak):",
    max_regression * 100.0);
println!("    {}/{} eval transitions passed",
    formal_monotone_checks - tolerated_violations, formal_monotone_checks);
println!();
println!("  Bounded Step (formal, grad_norm ≤ ε_clip = {:.2}):", config.epsilon_clip);
println!("    {}/{} eval transitions passed — NOTE: grad_norm is a proxy, not KL/TV",
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
```

## Gap 4: `sos.bounded_step` is dead code

### Problem

`verify_sos_step()` computes `bounded_step: step_distance <= config.epsilon_clip` where `epsilon_clip = 0.2` and `step_distance` is `grad_norm`. This result is never read. The operational check at line ~555 uses `config.max_grad_norm = 2.0`. The formal check is computed and discarded.

### Fix

Already addressed by Gap 1: `sos.bounded_step` is now counted in `formal_bounded_violations` and reported in the summary. The formal check (`grad_norm <= 0.2`) will fire frequently (gradient norms are typically > 0.2 on 7B models). The summary distinguishes this from the operational check (`grad_norm <= 2.0`), making it clear that the formal axiom requires KL, not grad_norm.

Additionally, update the SOSCheck recorded in the tracker to include the formal result:

```rust
sos_check: if eval_this_step {
    SOSCheck {
        // Use formal monotone (strict), not tolerance — tolerance is tracked separately
        monotone_improvement: sos.monotone_improvement,
        bounded_step: sos.bounded_step,  // formal: grad_norm <= epsilon_clip
        constraint_preservation: true,
    }
} else {
    SOSCheck {
        monotone_improvement: true,  // no eval data — not checked
        bounded_step,                // operational: grad_norm <= max_grad_norm
        constraint_preservation: true,
    }
},
```

This means `tracker.is_monotone()` and `tracker.monotonicity_violations()` now report FORMAL axiom status, while the operational tolerance is tracked via the separate `tolerated_violations` counter. The summary reports both.

## Gap 5: Three competing "evaluator" definitions

### Problem

"The evaluator" means:
1. `state.evaluator` in Python: batch_reward (noisy, updated every `/step`)
2. `StepMetrics.evaluator` in Rust: real eval at eval steps, stale at others
3. `verify_sos_step()` input: consecutive eval results

A user seeing `evaluator: 0.35` from `/status` and `final_evaluator: 0.42` from the tracker summary is confused — they are different metrics.

### Fix

Rename the Python-side field to eliminate the ambiguity. In `TrainingState` (already shown in Gap 2 fix):

```python
batch_reward: float = 0.0             # was: evaluator
batch_reward_history: list = field(...)  # was: evaluator_history
```

In the `/step` handler, update references:
```python
state.batch_reward = batch_reward
state.batch_reward_history.append(batch_reward)
```

In `/status` docstring, add:
```python
@app.route("/status", methods=["GET"])
def status():
    """Current training status.

    NOTE: batch_reward is a noisy per-step proxy.
    The true evaluator (SOS E(π_n)) is computed Rust-side via /evaluate
    on a fixed eval set with Lean verification. Do not conflate the two.
    """
    return jsonify(asdict(state))
```

## Gap 6: Rust/Python step counter divergence on errors

### Problem

If the Rust `/step` call fails (e.g., timeout), the Rust loop counter `step` increments but Python's `state.step` doesn't. After failures, the counters diverge permanently.

### Fix

Accept the divergence and document it. They measure different things — Rust counts loop iterations (attempts), Python counts successful gradient updates.

Add a comment in `run_phase4()` at the error branches:
```rust
Err(e) => {
    tracing::error!("Step {step}: training step failed: {e}");
    // Note: Rust 'step' counts loop iterations (including failures).
    // Python state.step counts successful gradient updates only.
    // These diverge on errors — both are valid, they measure different things.
    tracker.record(StepMetrics { /* ... failed step ... */ });
    println!("│ {:>3} │  FAILED  │          │        │          │    │", step);
    continue;
}
```

No code change needed — just the clarifying comment at each error branch.

## Gap 7: `GRPOConfig.max_grad_norm` missing `#[serde(default)]`

### Problem

`GRPOConfig` derives `Deserialize`. The new `max_grad_norm` field has no `#[serde(default)]`. Any JSON config file without `"max_grad_norm"` fails to deserialize.

### Fix

Add `#[serde(default)]` to the struct in `crates/pf-core/src/types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]  // ← ADD THIS — all fields have defaults via Default impl
pub struct GRPOConfig {
    pub group_size: usize,
    pub epsilon_clip: f64,
    pub kl_beta: f64,
    pub channel_weights: ChannelWeights,
    pub staleness_eta: u64,
    pub m2po_tau: f64,
    pub max_grad_norm: f64,
}
```

This makes ALL fields optional during deserialization, falling back to `Default::default()`. Clean and forward-compatible — any future fields added with defaults will also Just Work.

**Also add `#[serde(default)]` to `ChannelWeights`** (same file) since it's nested inside `GRPOConfig` and also has a `Default` impl:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]  // ← ADD THIS
pub struct ChannelWeights {
    pub correctness: f64,
    pub elegance: f64,
    pub generality: f64,
}
```

## Gap 8: `/evaluate` proofs not counted in `total_proofs_generated`

### Problem

`/generate` increments `state.total_proofs_generated`. `/evaluate` calls the same `_generate_proofs()` helper but doesn't count. Over 20 steps with 4 eval rounds × 40 proofs each = 160 uncounted proofs — 25% of total generation work invisible to monitoring.

### Fix

Add a separate counter to `TrainingState` to distinguish training from evaluation generation:

```python
@dataclass
class TrainingState:
    # ... existing fields ...
    total_eval_proofs_generated: int = 0
```

In the `/evaluate` handler, after `proofs = _generate_proofs(statements, n)`:
```python
    proofs = _generate_proofs(statements, n)
    state.total_eval_proofs_generated += sum(len(p) for p in proofs)
    return jsonify({"proofs": proofs})
```

Do NOT count eval proofs in `total_proofs_generated` — that would conflate training cost with evaluation cost. Do NOT move counting into `_generate_proofs()` — that couples the helper to global state.

`/status` now returns both counters, letting monitoring distinguish training from eval cost.

---

## File Change Summary

| File | Changes |
|------|---------|
| `crates/pf-cli/src/main.rs` | Gap 1: restructure eval branch — formal SOS result recorded and counted, separate counters for formal vs operational violations. Gap 3: SOS summary reports violations out of eval transitions. Gap 4: `sos.bounded_step` recorded in tracker SOSCheck. Gap 6: clarifying comment at error branches. |
| `crates/pf-core/src/types.rs` | Gap 7: `#[serde(default)]` on `GRPOConfig` and `ChannelWeights`. |
| `python/train_server.py` | Gap 2: remove `is_monotone` from `TrainingState`. Gap 5: rename `evaluator` → `batch_reward`, `evaluator_history` → `batch_reward_history`. Gap 8: add `total_eval_proofs_generated` counter, increment in `/evaluate`. |

## What NOT to Do

- Do NOT evaluate every step to fix Gap 3 — eval cost is already high. Fix the reporting denominator instead.
- Do NOT duplicate Rust-side monotone logic in Python (Gap 2) — the Python side can't do Lean verification.
- Do NOT remove `verify_sos_step()` — it produces meaningful formal axiom data. Fix is to USE its output, not delete it.
- Do NOT move proof counting into `_generate_proofs()` (Gap 8) — keeps the helper reusable.
- Do NOT make Python compute the true evaluator — Lean checking is Rust-side only.
- Do NOT try to synchronize Rust/Python step counters (Gap 6) — they measure different things by design.

## Testing

1. **Gap 1 verification**: Run Phase 4 for 20 steps. Verify SOS summary shows BOTH formal and operational violation counts. Verify formal monotone violations >= operational tolerated violations (formal is stricter).
2. **Gap 2 `/status` test**: Call `/status`. Verify `is_monotone` field is absent. Verify `batch_reward` field exists (not `evaluator`).
3. **Gap 3 denominator test**: Run 20 steps with eval_interval=5. Verify SOS summary says "X/4 eval transitions" not "X/19 steps."
4. **Gap 4 bounded_step test**: Verify SOS summary reports formal bounded-step violations (grad_norm > 0.2 will be common) separately from operational grad_norm warnings.
5. **Gap 7 serde test**: Deserialize a JSON `GRPOConfig` without `max_grad_norm` field. Verify it succeeds with default 2.0.
6. **Gap 8 counter test**: Run Phase 4 with eval. Call `/status`. Verify `total_eval_proofs_generated > 0` and `total_proofs_generated` does NOT include eval proofs.
