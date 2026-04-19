# Session Post-mortem: G5 Integration + OT Correction Set

**Date**: 2026-03-31
**Scope**: G5 ProofMemory→PythonBridge wiring + 5 OT-flagged corrections (C1-C5)

---

## Architectural Facts — Undocumented Until This Session

### 1. DUAL-MODEL ARCHITECTURE

`AdaptiveAgent` (`crates/pf-agents/src/adaptive.rs`) generates proofs by calling
`{api_url}/api/generate` — an **external Ollama-style endpoint**, NOT `train_server.py`.

Two separate models contribute proofs to the same `ProofMemory`:
- **HuggingFace policy** (via `train_server.py`, actively trained by GRPO, LoRA adapter updated each step)
- **AdaptiveAgent model** (via `api_url`, separate weights, no GRPO training, Ollama-compatible)

Both proof sources go through Lean 4 verification and enter `ProofMemory` via the same
accumulation loop in `pf-cli/src/main.rs`. This is **correct by design** — diverse proof
sources improve few-shot context quality — but must be understood by anyone changing ProofMemory
to avoid the assumption that there is a single model source.

**Implication**: If AdaptiveAgent is disabled, ProofMemory still grows from neural policy proofs.
If train_server.py is in MOCK mode, ProofMemory still grows from AdaptiveAgent proofs.

---

### 2. DUAL-MEMORY ARCHITECTURE

There are two independent proof memory stores:

| | Python `_proof_memory` | Rust `ProofMemory` |
|---|---|---|
| **Location** | `train_server.py` global | `pf-cli` in-process |
| **Type** | `list[dict]` | `HashMap<String, Vec<ProofExample>>` |
| **Cap** | 200 entries (LRU by age) | 3 per theorem (shortest proof wins) |
| **Populated from** | `/reward` when `reward > 0.5` | Lean checker batch verification |
| **Used for** | `_generate_proofs()` few-shot prefix | AdaptiveAgent snapshot + G5 bridge push |
| **Selection** | Shorter proofs preferred, dedup by (stmt, proof) | Difficulty + tag match, score-ranked |

**Divergence risk**: A proof verified on the Rust side (reward = 1.0 from Lean) can have reward
≤ 0.5 from Python's advantage calculation (due to group normalization). Such a proof enters
Rust `ProofMemory` but NOT Python `_proof_memory`.

The G5 bridge `configure()` push (every `eval_interval` steps on version bumps) is **eventually
consistent, not immediately consistent**. Python `_proof_memory` lags behind Rust `ProofMemory`
by up to `eval_interval` training steps.

**Current severity**: Low — `kl_beta=0.01` makes the few-shot context relatively low-weight
in the loss. **Monitoring required as few-shot context becomes more load-bearing** (e.g., when
Tier 4 tactic wall improvements arrive and few-shot is the primary signal source).

---

## Corrections Applied This Session

### C1 — `_grpo_step` distribution mismatch (CRITICAL, Python)
**File**: `python/train_server.py` ~line 396

**Problem**: `_generate_proofs()` conditions on `few_shot + PROOF_PROMPT_TEMPLATE`
but `_grpo_step()` computed log-prob under bare `PROOF_PROMPT_TEMPLATE`. For any proof
requiring few-shot context, the log-prob is artificially low → gradient weakened. As
few-shot context grows more useful (F3 intent), mismatch compounds.

**Fix**: `_grpo_step()` now builds the same prompt as `_generate_proofs()`:
```python
few_shot = _get_few_shot_examples(stmt, k=3)
prompt = few_shot + PROOF_PROMPT_TEMPLATE.format(statement=stmt)
```
Generation distribution and gradient distribution are now aligned.

---

### C2 — `/configure` type guard (MINOR, Python)
**File**: `python/train_server.py`, `/configure` endpoint

**Problem**: `entry.get(...)` throws `AttributeError` if `entry` is not a dict
(bridge serialization bug or None entry). No guard → unhandled 500.

**Fix**: Added `if not isinstance(entry, dict): continue` as first line of loop body.

---

### C3 — Phase 4 memory cold start (HIGH, Rust)
**File**: `crates/pf-cli/src/main.rs`

**Problem**: `phase4_memory = ProofMemory::new(3)` was always empty. Phase 3 proofs were
never transferred to Phase 4 (separate invocations). Phase 4 spent first ~50 steps with
empty few-shot context reaching Python.

**Fix**: Disk persistence handoff:
- Phase 3 now serializes `proof_memory` to `proof_memory.json` at exit
- Phase 4 (`--neural`) loads from `proof_memory.json` if it exists, falls back to empty

Data flow: `Phase 3 training → proof_memory.json → Phase 4 warm start`.
Phase 4 begins with all Tier 1-3 proofs from Phase 3, bridge push at step 0 (or first eval interval).

---

### C4 — `regressed_difficulty` wrong proxy (HIGH, Rust)
**File**: `crates/pf-cli/src/main.rs`, SOS violation handler

**Problem**: `batch.first().difficulty` is the first theorem of the TRAINING batch —
a random sample unrelated to what degraded on eval. T1 batch theorem → T1 recovery push
even when T4 theorems are actually failing.

**Fix**: Per-theorem eval tracking across steps:
- `evaluate_policy_detailed()` returns `(f64, Vec<bool>)` — aggregate + per-theorem pass/fail
- `prev_eval_pass: Vec<bool>` maintained across eval steps
- `eval_difficulties: Vec<u8>` built once from eval theorem dataset info
- `regressed_difficulty = max difficulty among theorems that passed prev eval but fail now`
- Falls back to `1` if no individual theorem regressed (shouldn't happen on SOS violation)

---

### C5 — Remove pf-execution (phantom dependency)
**Files**: `Cargo.toml` (workspace members) and `crates/pf-cli/Cargo.toml`

**Problem**: `ExecutionTranslator::translate()` is a stub returning `Hold` unconditionally.
No call sites exist in `pf-cli/src/`. Type mismatch (`ProofMemory::best_proof()` returns
`Option<&ProofExample>`, `translate()` expects `VerifiedTheorem`) prevents real use without
a full conversion layer. Added compile overhead for zero runtime contribution.

**Fix**: Removed `pf-execution` from both Cargo.toml files. Directory not deleted —
`git status` shows it as untracked (never committed). Remove when ready:
```bash
rm -rf crates/pf-execution/
```

---

## F5 Pre-flight Status (Post-Corrections)

All 5 corrections applied. `cargo build --release`: zero errors, zero warnings.

**F5 is cleared to run** with:
```bash
python3 scripts/grpo_round3.py \
  --stage both \
  --theorems-path data/theorems_level1_2.json \
  --easy-mix-ratio 0.3 \
  --group-size 32 \
  --steps 200 \
  --lr 1e-6
```
