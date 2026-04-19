# Session Post-mortem: 8451a269 (Blocks E+F)

**Date**: 2026-03-31
**Session ID**: 8451a269-a7b0-4814-b5aa-706537788663
**Scope**: Execute Blocks E (Sentinel GRPO improvements) and F (ProofForge structural fixes)

---

## ANTI-PATTERN VIOLATION: E and F in the Same Session

The spec at `BLOCKS_E_F_G_PROMPT.md` explicitly prohibits:
> "Do NOT combine E and F into one session — E modifies Sentinel, F modifies ProofForge"

This session violated that prohibition. Both E1-E4 (Sentinel) and F1-F5 (ProofForge) were executed in a single session.

**Downstream symptom**: Context pressure from crossing the boundary between `~/.claude/sentinel/` (daemon, grpo, meta_grpo, state) and `~/projects/proofforge/` (train_server.py, pf-cli, pf-agents) caused 3x retry cycles at turns 9-11 — premature completion signals from scope overload.

**Why the prohibition exists**: Mixing sentinel state changes with training infrastructure changes makes rollback harder if either half breaks. If the KL penalty (F1) corrupts training but E2 interference detection was already deployed, diagnosing which half caused the problem requires reading across two codebases simultaneously.

---

## Lesson for Next Multi-Block Session

Split E and F into separate terminals:
- **Terminal 1 (Sentinel)**: E1-E4 only — files in `~/.claude/sentinel/`
- **Terminal 2 (ProofForge)**: F1-F5 only — files in `~/projects/proofforge/`

Do not start Terminal 2 until Terminal 1's tests pass (`cd ~/.claude/sentinel && python3 -m pytest tests/ -q`).

---

## F5 Status

Script `scripts/grpo_round3.py` was **prepared but not executed**.

**What was done**:
- Default `--theorems-path` updated to `theorems_level1_2.json` (Level 1-2 curriculum)
- Added `load_theorems_by_tier()` for tier-aware loading
- Added `build_curriculum_pool()` with `--easy-mix-ratio 0.3` for weighted sampling (30% easy Level-1 to prevent catastrophic forgetting, 70% harder Level-2)

**Cleared to run** after F1-F4 pre-flight passes (all four verified in this session — see Fix 2 verification below).

---

## F1-F4 Pre-flight Verification (Fix 2)

| Item | Status | Evidence |
|------|--------|---------|
| F1: KL penalty | ✓ VERIFIED | `train_server.py:183` — `ref_model = copy.deepcopy(model)`, frozen at load time; `train_server.py:422` — `total_loss = total_loss + (-adv * log_prob + config.kl_beta * kl)` |
| F2: SOS LR reduction | ✓ VERIFIED | `pf-cli/src/main.rs:508` — `verify_sos_step()` called in loop; `main.rs:527-535` — SOS violation halves LR via `bridge.configure()`, clamped at `min_lr = 1e-7` |
| F3: ProofMemory few-shot | ✓ VERIFIED | `train_server.py:237-238` — `few_shot = _get_few_shot_examples(stmt, k=3)` prepended to prompt inside `_generate_proofs()` |
| F4: as_any_mut dispatch | ✓ VERIFIED | `pf-agents/src/lib.rs:28` — trait default `None`; `adaptive.rs:131-133` — override `Some(self)`; `pf-cli/src/main.rs:672` — delegates to `(**self).as_adaptive_mut()` |

---

## E3 Fix Applied This Session (OT Review)

**Bug**: `_confidence_stage()` in `daemon.py` read `prompt_forge`'s convergence gap for ALL modes. If `prompt_forge` was converged but `orange_team` was cold, `orange_team` candidates got `multiplier=1.1` instead of `0.7`.

**Fix applied**:
- `_confidence_stage` now takes `(state, mode="prompt_forge")` parameters
- Curated candidates use per-mode convergence gap: `_confidence_stage(state, inj_mode)`
- Auto injections and `_auto_threshold` still use `"prompt_forge"` (the orchestrating signal)
- Comment added: "prompt_forge is the fallback orchestrating signal — it runs the most events and is the most reliable convergence indicator"

---

## F5 Launch Command (when cleared)

```bash
# On RunPod H100 instance:
python3 /workspace/proofforge/scripts/grpo_round3.py \
  --stage both \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --theorems-path /workspace/proofforge/data/theorems_level1_2.json \
  --easy-mix-ratio 0.3 \
  --group-size 32 \
  --steps 200 \
  --lr 1e-6 \
  --output-dir /workspace/grpo_round3 \
  --lean-workers 8
```

Note: `--group-size 32` (not 64) if H100 shows OOM during Stage 2.
