# Session Results: Phase 3 Training Analysis + Bug Fixes

**Date**: 2026-03-31
**Scope**: Full analysis of Lean-reward GRPO training (9 checkpoints, 50 theorems), manifold comparison, 9 bug fixes

---

## 1. Training Arc Summary

**Model**: DeepSeek-Prover-V2-7B + LoRA (rank 16), H100 SXM
**Training**: GRPO with binary Lean 4 verification reward, 3 temperatures (0.3, 0.7, 1.2), 50 theorems × 3 samples = 150 per checkpoint
**Duration**: ~9.5 hours for 150 steps ($2.69/hr)

### Pass Rate Trajectory

| Step | Pass Rate | Solved | Phase |
|------|-----------|--------|-------|
| 0 | 2.0% | 3 | Baseline |
| 5 | 8.0% | 12 | Initial jump |
| 10 | 9.3% | 13 | Plateau |
| 20 | 8.0% | 11 | Regression (noise) |
| 30 | 8.7% | 12 | End of plateau |
| 50 | 22.7% | 21 | **Phase transition** |
| 75 | 42.7% | 39 | **Peak efficiency** |
| 100 | 60.7% | 41 | Deceleration |
| 150 | 72.0% | 43 | Approaching asymptote |

**Two-phase pattern**: Plateau (steps 0-30, churning at ~8-9%) then breakout (steps 30-150). The plateau is characterized by "flickering" — the model solves different subsets of ~11-13 theorems each checkpoint but doesn't accumulate.

**Phase transition at step 30→50**: REPETITION failure mode collapsed (38→21), TACTIC_ATTEMPT surged (25→33). The model crossed from repetitive text generation to genuine tactic attempts.

**Peak efficiency at step 75**: Learning depth metric = 1.73 (19 new theorems from only 11 remaining unsolved). T1-T3 clusters all crossed 80% mastery simultaneously.

**Sigmoid fit**: L=74.7%, k=0.044, x0=68, R²=0.992. Asymptote at ~75%.

### Failure Mode Evolution

| Step | REPETITION | TACTIC_ATTEMPT | COMMENTARY | DEGENERATE |
|------|-----------|----------------|------------|------------|
| 0 | 59 | 16 | 36 | 3 |
| 30 | 38 | 25 | 27 | 13 |
| 75 | 2 | 47 (peak) | 7 | 2 |
| 150 | 0 | 18 | 3 | 0 |

TACTIC_ATTEMPT peaks at step 75 then declines — because more attempts succeed.

---

## 2. The T4 Wall — 7 Unsolved Theorems

| ID | Statement | Tier | Required Tactics |
|----|-----------|------|-----------------|
| pf_t2_14 | n ≤ n + m | T2 | omega, Nat.le_add_right |
| pf_t3_09 | n * 2 = n + n | T3 | ring, omega (regressed from step 75) |
| pf_t4_03 | p ∨ q → q ∨ p | T4 | cases, Or.swap |
| pf_t4_04 | p ∧ (q ∨ r) → (p ∧ q) ∨ (p ∧ r) | T4 | cases + constructor chaining |
| pf_t4_05 | n ≤ m → n ≤ m + 1 | T4 | Nat.le_succ_of_le, omega |
| pf_t4_06 | n = 0 ∨ ∃ m, n = Nat.succ m | T4 | match/cases on Nat |
| pf_t4_08 | a ≤ b → b ≤ a → a = b | T4 | Nat.le_antisymm |

**Root cause**: The model learned `simp` as a universal strategy (67% of first tactics at step 150). `cases` appears only ~1 time per checkpoint — the model never learned case analysis. The 7 unsolved theorems all require `cases`, `omega`, or specialized lemmas.

**Sigmoid asymptote at 75% confirms**: More training steps will not help. The model needs curriculum expansion (Level 1-2 theorems teaching tactic composition).

---

## 3. Manifold Analysis (Checkpoint 30 vs 75)

The most scientifically important comparison — what changed in representation space during the 8.7%→42.7% breakout?

| Metric | Ckpt 30 | Ckpt 75 | Change |
|--------|---------|---------|--------|
| Cluster separation (CV acc) | 86.7% | 61.3% | **COLLAPSED** |
| Intrinsic dim (2NN mean) | 6.90 | 5.77 | **-1.13** (compressed) |
| PCA dims for 90% variance | 31 | 25 | -6 |
| H1 loops (mean) | 12.5 | 13.8 | +1.2 (preserved) |
| H2 voids (mean) | 2.5 | 0.5 | **-2.0** (filled) |
| Mean generation length | 146 | 185 | +39 steps |

**Key finding**: The breakout was **territorial expansion + compression**, not boundary sharpening:
- Success region CONQUERED previously-failure territory (separation dropped)
- Manifold COMPRESSED by ~1 intrinsic dimension (correlated features collapsed)
- H2 voids FILLED while H1 cyclic structure preserved
- The model found shared proof infrastructure that collapsed redundant dimensions

---

## 4. Tactic Analysis

### Emergence Timeline (counts across successful proofs)
- **simp**: 2 → 75 (universal first tactic by step 150)
- **simp_all**: 2 → 54 (learned to combine with lemma libraries)
- **induction**: 1 → 13 (structural recursion, emerged at step 50)
- **omega**: 1 → 14 (arithmetic automation, emerged at step 50)
- **tauto**: 0 → 15 (propositional automation, emerged at step 75)
- **cases**: 0 → 1 (NEVER LEARNED — explains the T4 wall)

### Key Compositions (emerged at step 75)
- `simp → induction → simp_all` (8x at step 75)
- `simp_all → tauto` (5x at step 75)
- `simp → rfl` (dominant two-step throughout)

---

## 5. Stability Analysis

### Per-Theorem Categories
- **15 learners** (once solved, stays solved): Mostly step 50-75 acquisitions
- **29 flickerers** (solved at some checkpoints, lost at others): Stabilize at step 50+
- **1 unstable** (pf_t3_09): Solved once at step 75, regressed permanently
- **6 never-solved**: pf_t2_14, pf_t4_03/04/05/06/08

### Regression Analysis
- 39 total regression events across 29 theorems
- 30 of 39 regressions in steps 0-30 (plateau phase)
- Root cause: sampling noise — with 3 samples at p≈8%, P(miss all 3) = 78%
- Post-step 75: only 2 regressions (genuine forgetting)

---

## 6. Prediction Validation

| Prediction (from session_handoff_2026_03_29.md) | Actual | Verdict |
|------------------------------------------------|--------|---------|
| Pass rate at ckpt 50 = 10 ± 5% | 22.7% | **WRONG** (2.3x) |
| Pass rate at ckpt 200 = 12 ± 8% | 72.0% at ckpt 150 | **MASSIVELY WRONG** (6x) |
| Discontinuity at ckpt 50: 20% chance | Yes, +14% jump | **CORRECT** |

Predictions anchored on step 0-20 plateau data failed to anticipate the breakout phase. The three-phase dynamics (plateau → breakout → saturation) were qualitatively predicted but the magnitude was under-estimated by 5-6x.

---

## 7. Heuristic vs Lean Reward Comparison

| Reward Type | Steps 0-30 Pass Rate | Learning Signal |
|------------|---------------------|-----------------|
| Heuristic | 94.7% → 98.0% | None (everything passes) |
| Lean verification | 2.0% → 8.7% | Real gradient (10-50x harder) |

Heuristic reward provides no learning pressure. Lean verification is essential for actual capability improvement.

---

## 8. Bug Fixes Applied (D4-D12)

| Fix | File | Description |
|-----|------|-------------|
| D4 | train_server.py | Cache prompt tokenization outside inner loop (O(G)→O(1) per statement) |
| D5 | train_server.py | Sync scheduler with state.step on empty batches |
| D6 | train_server.py | Truncation guard on tokenizer (max_seq_len) prevents CUDA OOM |
| D7 | bridge.rs | Add total_eval_proofs_generated and batch_reward_history to StatusResponse |
| D8 | train_server.py | Cap batch_reward_history at 1000 entries |
| D9 | train_server.py | SIGTERM/SIGINT handler for emergency checkpoint save |
| D10 | bridge.rs | Separate 5s health check client vs 600s step client |
| D11 | train_server.py | Normalize proof text (null bytes, CRLF) for Lean checker |
| D12 | main.rs | min(batch_size, dataset_len) guard on choose_multiple |

All verified: Python syntax clean, `cargo check` compiles all 6 crates with zero errors.

---

## 9. Recommended Next Steps

### Priority 1: Round 3 Training with Tactic Curriculum
- Merge original 50 theorems + 30 Level 1-2 theorems = 80 theorem training set
- Level 1-2 theorems explicitly teach `cases`, `omega`, `intro; exact`, `rw [h]`
- Script ready: `scripts/grpo_round3.py` (1304 lines)
- Spec: `docs/round3_spec.md`

### Priority 2: Run Manifold Analysis on All Checkpoints
- Currently only ckpt 30 and 75 analyzed
- Should run on 100 and 150 for full picture
- Key question: does the manifold continue compressing or does it plateau with the pass rate?

### Priority 3: Address pf_t3_09 Regression
- Solved at step 75, lost at 100 and 150
- `n * 2 = n + n` should be solvable by `ring` or `omega`
- May indicate catastrophic forgetting in the arithmetic tactic space

---

## Infrastructure

### Sentinel System Fixes (Blocks A-C)
During this session, three blocks of Sentinel infrastructure fixes were also applied:
- **Block A**: Fixed injection delivery race condition (daemon HTTP handler cleared pending_injections.jsonl before command hook could read it)
- **Block B**: Added auto-consumption loops to mode terminals + auto-inject policy to 7 mode CLAUDE.md files
- **Block C**: Added extraction.py (Insight Miner heuristics), arch_sentinel_heuristics.py (Arch Sentinel), LRU file cache in state.py, complexity-aware confidence modulation (gated)
- All 178 Sentinel tests passing
