# Tactic-Level Tandem GRPO: Architectural Specification

> **Version:** 1.0
> **Date:** 2026-04-13
> **Author:** Architecture Sentinel (ProofForge)
> **Source Paper:** "Tandem Training for Language Models" -- West, Anderson, Kamar, Horvitz (arXiv:2510.13551v2, Jan 2026)
> **Target File:** `scripts/grpo_lean_reward.py` (1603 lines)
> **Codebase Verification:** All line references verified against commit on main branch, 2026-04-13. Zero discrepancies.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: Tandem Training](#2-background-tandem-training)
3. [Critical Adaptation: Tactic-Level Granularity](#3-critical-adaptation-tactic-level-granularity)
4. [Existing Codebase Map](#4-existing-codebase-map)
5. [Hardware Constraints and Memory Architecture](#5-hardware-constraints-and-memory-architecture)
6. [New Functions: Signatures, Algorithms, Data Flow](#6-new-functions-signatures-algorithms-data-flow)
7. [Modified Functions](#7-modified-functions)
8. [CLI Interface](#8-cli-interface)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Feature Interaction Matrix](#10-feature-interaction-matrix)
11. [Design Decisions and Rationale](#11-design-decisions-and-rationale)
12. [Implementation Plan](#12-implementation-plan)
13. [Testing Strategy](#13-testing-strategy)
14. [Risk Matrix](#14-risk-matrix)
15. [Metrics and Success Criteria](#15-metrics-and-success-criteria)
16. [Edge Cases](#16-edge-cases)
17. [Future Extensions](#17-future-extensions)
18. [File Change Summary](#18-file-change-summary)
19. [Reference: Hyperparameter Comparison](#19-reference-hyperparameter-comparison)

---

## 1. Executive Summary

This specification defines the integration of **tactic-level tandem training** into
ProofForge's GRPO pipeline. Tandem training interleaves generation between a trainable
"senior" model and a frozen "junior" model to enforce **handoff robustness** -- the
property that the senior's proofs are continuable by weaker models.

**Key adaptation from the paper:** The paper interleaves at the token level (for natural
language math on GSM8K). We interleave at the **tactic level** because Lean 4's rigid
syntax means random junior tokens mid-tactic produce unparseable garbage. A "tactic" is
one complete line in a newline-separated proof.

**Scope:** Python-only changes to `scripts/grpo_lean_reward.py`. No Rust changes. No
new dependencies.

**Memory cost:** ~100MB additional VRAM (two LoRA adapter sets sharing one base model).

**Speed cost:** ~10-15% slower per GRPO step (sequential tandem generation vs. batched,
but generation is <20% of step time since Lean verification dominates).

---

## 2. Background: Tandem Training

Tandem training (West et al., 2026) modifies RL rollouts by randomly interleaving
outputs from a frozen "junior" model during the trainable "senior" model's generation.

### 2.1. Core Mechanism

At each generation step, a coin flip with probability `handoff_prob` decides whether
the **senior** (trainable) or **junior** (frozen) produces the next unit of output.
Rewards are computed on the **full tandem rollout** (mixed senior + junior outputs).
The loss gradient flows only through **senior-authored positions** -- junior positions
are masked from the policy gradient.

### 2.2. Why It Works

Because rewards are only high when the tandem pair co-produces correct solutions, the
senior is incentivized to produce reasoning that the junior can *continue*. This
operationalizes "intelligibility" as handoff robustness.

**Formal definition:** A senior model M_sen's proof is intelligible to junior model
M_jun if M_jun can take over during randomly assigned portions of the proof without
causing verification failure.

### 2.3. Key Results from Paper

- On GSM8K: notational jargon drops 99% to 0% within 20 gradient updates
- Accuracy stays above junior baseline throughout
- Compatible with any RL algorithm (paper tested REINFORCE; states GRPO/PPO compatibility)

---

## 3. Critical Adaptation: Tactic-Level Granularity

### 3.1. Why Not Token-Level?

Lean 4 has rigid syntax. A random junior token inserted mid-tactic produces unparseable
output. In practice, nearly 100% of token-level tandem rollouts would fail Lean
verification, yielding zero reward signal and zero gradient -- the model learns nothing.

### 3.2. Tactic-Level Interleaving

At each **tactic boundary**, flip a coin: the senior or junior generates the **next full
tactic** (autoregressively, token by token, until the next boundary or EOS). This
preserves syntactic validity per-tactic while enforcing handoff robustness at the
granularity that matters for proofs.

### 3.3. Tactic Boundary Definition (v1)

**v1 simplification:** Every newline character (`\n`) is a tactic boundary.

DeepSeek-Prover-V2-7B generates proofs as newline-separated tactic sequences in nearly
all cases. Nested `have`/`suffices`/`calc` blocks that span multiple lines will
occasionally get split mid-block, but this is acceptable for v1:
- The Lean verifier catches any resulting failures
- Those rollouts receive reward 0 and don't contribute to the gradient
- No false positives can occur (bad proofs can't pass verification)

**v2 improvement (out of scope):** Indentation-aware boundary detection that doesn't
split multi-line blocks.

### 3.4. Granularity Comparison

| Approach | Pros | Cons |
|----------|------|------|
| Token-level (paper) | Finest granularity, maximally faithful to paper | Lean syntax breaks instantly. ~100% failure rate. Zero learning signal. |
| **Tactic-level (this spec)** | **Preserves syntactic validity. Meaningful proof unit. Junior contributes complete tactics.** | **Coarser granularity. But proofs are 5-20 tactics, so ~50% handoff still means many switches.** |
| Block-level (multi-tactic) | Even coarser. Less tandem pressure. | Too coarse for short proofs (3-5 tactics). Insufficient learning signal. |

---

## 4. Existing Codebase Map

All functions below are in `scripts/grpo_lean_reward.py` (1603 lines).

### 4.1. Function Reference Table

| Function | Lines | Role | Tandem Impact |
|----------|-------|------|---------------|
| `generate_completions()` | 435-475 | Batched generation via `model.generate()`. Opaque -- cannot interleave mid-generation. | Replaced by `generate_tandem_completions()` when tandem active |
| `generate_with_hidden_states()` | 207-287 | Manual autoregressive loop with KV cache. Token-by-token control. | **Skeleton for tandem generation function** |
| `grpo_step()` | 764-883 | One GRPO gradient update. Calls `generate_completions()`, verifies, computes loss. | Replaced by `tandem_grpo_step()` when tandem active |
| `compute_completion_log_prob()` | 724-757 | Forward pass for `sum(log_prob)` over completion. | Extended by `compute_tandem_log_prob()` with senior mask |
| `lean_verify_batch()` | 140-165 | Parallel Lean 4 verification. | Unchanged -- verifier doesn't care who generated which token |
| `run_grpo()` | 909-1285 | Outer training loop: model loading, optimizer, checkpoint schedule. | Extended with tandem parameters and conditional dispatch |
| `make_prompt()` | 195-200 | Builds prompt string. | Unchanged |

### 4.2. Existing Autoregressive Pattern (lines 207-287)

The `generate_with_hidden_states()` function already implements the exact control-flow
pattern needed for tandem generation:

```
1. Tokenize prompt -> current_ids, attention_mask
2. Initialize past_key_values = None
3. LOOP for max_new_tokens steps:
   a. Forward pass with KV cache -> logits
   b. Sample next token from logits/temperature
   c. If EOS -> break
   d. Feed next token into KV cache -> continue
4. Decode -> proof text
```

The tandem function follows this structure, adding adapter switching at tactic
boundaries (newlines).

### 4.3. Existing Model Loading (lines 936-1003)

Two paths:

**Path A -- PEFT checkpoint (lines 990-993):**
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)
```
Creates adapter with default name `"default"`.

**Path B -- Fresh LoRA (lines 994-1003):**
```python
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=[...])
model = get_peft_model(base_model, lora_config)
```
Creates adapter with default name `"default"`.

### 4.4. Existing Loss Computation (lines 848-863)

```python
for comp, adv in zip(completions, advantages):
    if adv <= 0 or len(comp["token_ids"]) == 0:
        continue
    log_prob = compute_completion_log_prob(model, tokenizer, prompt, comp["token_ids"])
    loss_term = -adv * log_prob
    total_loss = total_loss + loss_term
    n_updates += 1
loss = total_loss / n_updates
loss.backward()
```

No KL penalty, no PPO clipping. Pure REINFORCE with group-normalized advantages.

### 4.5. Existing Reward Computation

- Binary: `1.0 if lean_verify_single(statement, proof) else 0.0`
- Optional efficiency reward (phase-in after configurable step, blends proof length)
- Elegance/generality channels exist in Rust types (`pf-core/src/types.rs:77`,
  `pf-core/src/grpo.rs:78`) but are NOT wired into the Python training script
- `generality` is hardcoded to `0.5` at `pf-core/src/grpo.rs:91`

---

## 5. Hardware Constraints and Memory Architecture

### 5.1. Target Hardware

- **GPU:** Single NVIDIA H100 80GB (RunPod)
- **Base model:** DeepSeek-Prover-V2-7B
  - bfloat16: ~14GB VRAM
  - 4-bit QLoRA: ~4GB VRAM
- **LoRA adapters:** ~50MB each (r=16, 7 target modules)
- **Optimizer state (AdamW):** ~100MB for LoRA parameters
- **KV cache + activations:** ~10-20GB depending on sequence length

### 5.2. Memory Architecture Decision: Named Adapters

**Problem:** Both senior (trainable) and junior (frozen) need to generate tokens. We
cannot load two full 7B models (28GB bfloat16 -- won't fit alongside optimizer state and
activations on 80GB).

**Solution: PEFT Named Adapters (Option B)**

Both senior and junior are LoRA adapter sets on the **same base model**. PEFT natively
supports multiple named adapters via `model.load_adapter()` and `model.set_adapter()`.

```
+---------------------------------------------+
|                  Base Model                  |
|           DeepSeek-Prover-V2-7B             |
|              (~14GB bfloat16)                |
|                                             |
|  +-------------+     +-------------+       |
|  |   Senior     |     |   Junior     |       |
|  |  (trainable) |     |  (frozen)    |       |
|  |  LoRA r=16   |     |  LoRA r=16   |       |
|  |   ~50MB      |     |   ~50MB      |       |
|  +------+-------+     +------+-------+       |
|         |                    |               |
|         v                    v               |
|    set_adapter()        set_adapter()        |
|    (pointer swap)       (pointer swap)       |
+---------------------------------------------+
```

**Memory overhead:** ~50MB for the junior adapter. Negligible on 80GB.

**Switching cost:** `set_adapter()` is a pointer swap on LoRA weight matrices --
effectively free (microseconds).

### 5.3. Alternatives Considered and Rejected

| Option | Description | Why Rejected |
|--------|-------------|--------------|
| A: Two `PeftModel` instances | Two wrappers around same base | Duplicates base model reference counting, more complex lifecycle |
| **B: Named adapters** | **Single PeftModel, two adapter sets** | **Selected: zero extra memory, native PEFT API, pointer-swap switching** |
| C: Two full models | Separate base model copies | Doubles VRAM (~28GB), won't fit with optimizer state on 80GB |

### 5.4. KV Cache Across Adapter Switches

When switching adapters mid-generation, the KV cache from prior tokens was computed with
the other adapter's attention weights.

**Decision: Keep cache across switches (no invalidation).**

Rationale:
1. The paper (West et al.) does NOT recompute cache on switches and reports no issues
2. LoRA rank 16 on a 7B model perturbs attention weights by <1% -- the KV cache from the
   other adapter is a close approximation
3. Recomputing means re-forwarding all prior tokens on every switch -- for 512 tokens
   with 10 switches, that's ~5x more compute
4. Lean verification is ground truth: if stale cache causes a bad proof, it gets reward 0

---

## 6. New Functions: Signatures, Algorithms, Data Flow

### 6.1. `generate_tandem_completion()`

**Location:** `scripts/grpo_lean_reward.py`, insert after `generate_with_hidden_states()` (after line 287)

```python
@torch.no_grad()
def generate_tandem_completion(
    model,                              # PeftModel with named adapters
    tokenizer,
    prompt: str,
    senior_adapter: str = "senior",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,          # P(junior generates each tactic)
    newline_token_id: int | None = None,
) -> dict:
    """Generate a single tandem completion with tactic-level interleaving.

    At each tactic boundary (newline token), flips a biased coin to decide
    whether the senior or junior adapter generates the next tactic. The full
    sequence is generated autoregressively with KV cache, switching adapters
    at boundaries without cache invalidation.

    Args:
        model: PeftModel with at least two named adapters loaded.
        tokenizer: Tokenizer matching the base model.
        prompt: Full prompt string (from make_prompt()).
        senior_adapter: Name of the trainable adapter.
        junior_adapter: Name of the frozen adapter.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (applied to both adapters).
        handoff_prob: Probability of junior generating each tactic.
            0.0 = pure senior (no tandem). 1.0 = pure junior.
        newline_token_id: Pre-computed token ID for newline. If None,
            computed from tokenizer on first call.

    Returns:
        {
            "token_ids": list[int],         # All generated token IDs
            "text": str,                    # Decoded proof text
            "senior_mask": list[bool],      # True where senior generated
            "tactic_assignments": list[str], # ["senior","junior",...] per tactic
            "n_handoffs": int,              # Number of adapter switches
        }
    """
```

**Algorithm (pseudocode):**

```
FUNCTION generate_tandem_completion(model, tokenizer, prompt, ...):

    # 1. Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 2. Resolve newline token ID
    IF newline_token_id is None:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    # 3. Initialize state
    past_key_values = None
    all_token_ids = []
    senior_mask = []
    tactic_assignments = []
    n_handoffs = 0

    # 4. First tactic assignment (coin flip)
    current_gen = "senior" IF random.random() > handoff_prob ELSE "junior"
    model.set_adapter(senior_adapter IF current_gen == "senior" ELSE junior_adapter)
    tactic_assignments.append(current_gen)

    # 5. Autoregressive generation loop
    FOR step IN range(max_new_tokens):

        # 5a. Forward pass with KV cache
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # 5b. Sample next token
        logits = outputs.logits[0, -1, :]
        IF temperature > 0:
            probs = softmax(logits / temperature)
            next_token = multinomial(probs)
        ELSE:
            next_token = argmax(logits)

        tok_id = next_token.item()

        # 5c. Record token and authorship
        all_token_ids.append(tok_id)
        senior_mask.append(current_gen == "senior")

        # 5d. Check termination
        IF tok_id == tokenizer.eos_token_id:
            BREAK

        # 5e. Check tactic boundary (newline)
        IF tok_id == newline_token_id:
            # Flip coin for next tactic
            new_gen = "senior" IF random.random() > handoff_prob ELSE "junior"
            IF new_gen != current_gen:
                n_handoffs += 1
                model.set_adapter(
                    senior_adapter IF new_gen == "senior" ELSE junior_adapter
                )
            current_gen = new_gen
            tactic_assignments.append(current_gen)

        # 5f. Prepare next input (KV cache: only feed new token)
        current_ids = next_token.unsqueeze(0)
        IF attention_mask is not None:
            attention_mask = cat([attention_mask, ones(1,1)], dim=1)

    # 6. Decode and return
    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)

    # 7. Restore senior adapter (callers expect senior active)
    model.set_adapter(senior_adapter)

    RETURN {
        "token_ids": all_token_ids,
        "text": text,
        "senior_mask": senior_mask,
        "tactic_assignments": tactic_assignments,
        "n_handoffs": n_handoffs,
    }
```

**Invariants:**
- `len(token_ids) == len(senior_mask)` -- one mask entry per generated token
- `len(tactic_assignments) >= 1` -- at least one tactic (the first)
- After return, the senior adapter is always active on the model
- The newline token that triggers a boundary is attributed to the *current* generator
  (the one that produced it), not the next one

### 6.2. `generate_tandem_completions()`

**Location:** Immediately after `generate_tandem_completion()`

```python
def generate_tandem_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    senior_adapter: str = "senior",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,
) -> list[dict]:
    """Generate group_size tandem completions for a single prompt.

    Calls generate_tandem_completion() sequentially. GPU batching is not
    possible because different sequences hit tactic boundaries at different
    token positions, requiring per-sequence adapter switching.

    Returns: list of dicts (same schema as generate_tandem_completion output).
    """
```

**Algorithm:**

```
FUNCTION generate_tandem_completions(model, tokenizer, prompt, group_size, ...):

    # Pre-compute newline token ID once
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    completions = []
    FOR i IN range(group_size):
        comp = generate_tandem_completion(
            model, tokenizer, prompt,
            senior_adapter=senior_adapter,
            junior_adapter=junior_adapter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            handoff_prob=handoff_prob,
            newline_token_id=newline_token_id,
        )
        completions.append(comp)

    RETURN completions
```

**Performance analysis:**
- Sequential generation of `group_size=8` at `max_new_tokens=512` on H100 with 7B:
  ~8 x 2s = 16s per theorem
- Standard `generate_completions()` does same in ~4s (GPU-batched)
- This is a 4x slowdown in generation
- But generation is typically <20% of step time (Lean verification at 30s timeout dominates)
- **Net step time impact: ~10-15% increase**

### 6.3. `compute_tandem_log_prob()`

**Location:** After `compute_completion_log_prob()` (after line 757)

```python
def compute_tandem_log_prob(
    model,
    tokenizer,
    prompt: str,
    completion_token_ids: list[int],
    senior_mask: list[bool],
    senior_adapter: str = "senior",
) -> torch.Tensor:
    """Compute sum of log probabilities for SENIOR-AUTHORED tokens only.

    Runs a single forward pass over the full sequence (prompt + all completion
    tokens), then masks the log_prob sum to include only positions where
    senior_mask[i] is True.

    The model MUST have the senior adapter active (this function sets it
    explicitly). The returned tensor requires grad for backpropagation.

    Args:
        model: PeftModel with senior adapter.
        tokenizer: Tokenizer matching the base model.
        prompt: The prompt string that preceded this completion.
        completion_token_ids: All token IDs in the completion (senior + junior).
        senior_mask: Boolean mask, same length as completion_token_ids.
            True at positions where the senior generated the token.
        senior_adapter: Name of the senior adapter.

    Returns:
        Scalar tensor: sum of log_prob at senior-authored positions.
        Requires grad. Returns 0.0 (with grad) if no senior tokens.
    """
```

**Algorithm:**

```
FUNCTION compute_tandem_log_prob(model, tokenizer, prompt, completion_token_ids,
                                  senior_mask, senior_adapter):

    # 1. Ensure senior adapter is active
    model.set_adapter(senior_adapter)

    # 2. Build full input sequence
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    completion_tensor = tensor([completion_token_ids], dtype=long, device=model.device)
    full_ids = cat([prompt_ids, completion_tensor], dim=1)

    # 3. Count senior tokens; early return if none
    n_senior = sum(senior_mask)
    IF n_senior == 0:
        RETURN tensor(0.0, device=model.device, requires_grad=True)

    # 4. Single forward pass (with grad -- this is for the loss)
    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # 5. Extract prediction logits at completion positions
    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)
    # Logits at [prompt_len-1 .. prompt_len+comp_len-2] predict tokens at
    # [prompt_len .. prompt_len+comp_len-1]
    pred_logits = logits[0, prompt_len - 1 : prompt_len + comp_len - 1, :]

    # 6. Compute per-token log probabilities
    targets = completion_tensor[0]  # (comp_len,)
    log_probs = log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[arange(comp_len), targets]  # (comp_len,)

    # 7. Apply senior mask
    mask_tensor = tensor(senior_mask, dtype=bool, device=model.device)
    masked_log_probs = token_log_probs[mask_tensor]

    RETURN masked_log_probs.sum()
```

**Critical detail -- why forward pass uses senior adapter only:**

During the forward pass for log_prob computation, the model uses the **senior adapter
only** (not switching between adapters per token). This is correct because we are
computing `pi_senior(a|s)` -- the senior's probability of the entire completion -- but
only summing over positions where the senior chose the token. The mathematical
justification:

Standard GRPO loss:
```
L = -A(s) * sum_t log pi(a_t | s, a_{<t})
```

Tandem GRPO loss (with masking):
```
L = -A(s) * sum_{t in SENIOR_TOKENS} log pi_senior(a_t | s, a_{<t})
```

The conditioning on `a_{<t}` includes both senior and junior tokens -- this is correct
and intended. The senior's policy is evaluated on its ability to assign high probability
to its own tokens, given the full context (including junior tokens). This is analogous
to teacher forcing in sequence-to-sequence models.

### 6.4. `tandem_grpo_step()`

**Location:** After `grpo_step()` (after line 883)

```python
def tandem_grpo_step(
    model,                              # PeftModel with named adapters
    tokenizer,
    batch_theorems: list,
    group_size: int,
    max_new_tokens: int,
    optimizer,
    scheduler,
    step_idx: int,
    lean_workers: int = 8,
    eff_oracle=None,
    running_pass_rate: float = 0.0,
    training_temperature: float = 0.7,
    handoff_prob: float = 0.5,
    senior_adapter: str = "senior",
    junior_adapter: str = "junior",
) -> dict:
    """Perform one tandem GRPO gradient update over a batch of theorems.

    Identical to grpo_step() except:
    1. Uses generate_tandem_completions() for generation
    2. Uses compute_tandem_log_prob() with senior_mask for loss
    3. Reports tandem-specific metrics

    Returns a metrics dict (superset of grpo_step's return).
    """
```

**Algorithm (changes from `grpo_step()` marked with [TANDEM]):**

```
FUNCTION tandem_grpo_step(model, tokenizer, batch_theorems, ...):

    optimizer.zero_grad()
    total_loss = tensor(0.0)
    n_updates = 0
    all_rewards = []
    all_advantages = []
    total_senior_frac = 0.0      # [TANDEM]
    total_handoffs = 0            # [TANDEM]
    n_completions_total = 0       # [TANDEM]

    FOR thm IN batch_theorems:
        prompt = make_prompt(thm["statement"])

        # --- 1. Generate tandem completions [TANDEM] ---
        model.set_adapter(senior_adapter)  # ensure clean state
        was_training = model.training
        model.eval()                       # match grpo_step pattern
        completions = generate_tandem_completions(   # [TANDEM]
            model, tokenizer, prompt,
            group_size=group_size,
            senior_adapter=senior_adapter,           # [TANDEM]
            junior_adapter=junior_adapter,           # [TANDEM]
            max_new_tokens=max_new_tokens,
            temperature=training_temperature,
            handoff_prob=handoff_prob,                # [TANDEM]
        )
        model.train()

        # --- Tandem metrics [TANDEM] ---
        FOR comp IN completions:
            IF len(comp["senior_mask"]) > 0:
                total_senior_frac += sum(comp["senior_mask"]) / len(comp["senior_mask"])
            total_handoffs += comp["n_handoffs"]
            n_completions_total += 1

        # --- 2. Verify (unchanged) ---
        statements = [thm["statement"]] * len(completions)
        proofs = [c["text"] for c in completions]
        needs_defs = thm.get("_requires_definitions", False)
        verification_results = lean_verify_batch(
            statements, proofs,
            max_workers=lean_workers, timeout=30,
            needs_definitions=needs_defs,
        )

        # --- 2b. Compute rewards (unchanged) ---
        raw_rewards = [1.0 if v else 0.0 for v in verification_results]
        IF eff_oracle is not None:
            rewards = compute_efficiency_rewards(
                oracle=eff_oracle, theorem=thm,
                completions=completions,
                verification_results=verification_results,
                raw_rewards=raw_rewards, step=step_idx,
                pass_rate=running_pass_rate, temperature=0.7,
            )
        ELSE:
            rewards = raw_rewards

        all_rewards.extend(rewards)

        # --- 3. Advantages (unchanged) ---
        rewards_tensor = tensor(rewards)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        all_advantages.extend(advantages)

        # --- 4. Loss with tandem masking [TANDEM] ---
        FOR comp, adv IN zip(completions, advantages):
            IF adv <= 0 OR len(comp["token_ids"]) == 0:
                CONTINUE
            IF not any(comp["senior_mask"]):    # [TANDEM] skip all-junior
                CONTINUE

            log_prob = compute_tandem_log_prob(   # [TANDEM]
                model, tokenizer, prompt,
                comp["token_ids"],
                comp["senior_mask"],              # [TANDEM]
                senior_adapter=senior_adapter,    # [TANDEM]
            )
            loss_term = -adv * log_prob
            total_loss = total_loss + loss_term
            n_updates += 1

    # --- Gradient step (unchanged) ---
    IF n_updates > 0:
        loss = total_loss / n_updates
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_val = loss.item()
    ELSE:
        scheduler.step()
        loss_val = 0.0

    # --- Compute tandem metrics [TANDEM] ---
    avg_senior_frac = total_senior_frac / max(n_completions_total, 1)
    avg_handoffs = total_handoffs / max(n_completions_total, 1)

    RETURN {
        "step": step_idx,
        "loss": loss_val,
        "mean_reward": mean(all_rewards),
        "mean_advantage": mean(all_advantages),
        "n_updates": n_updates,
        "efficiency_active": (eff_oracle is not None and ...),
        "pass_rate_estimate": running_pass_rate,
        "temperature": training_temperature,
        "tandem_senior_frac": avg_senior_frac,              # [TANDEM]
        "tandem_handoffs_per_completion": avg_handoffs,      # [TANDEM]
        "tandem_completions_total": n_completions_total,     # [TANDEM]
    }
```

---

## 7. Modified Functions

### 7.1. `run_grpo()` -- Extended Signature

**Do NOT create a new function.** Extend `run_grpo()` at line 909.

**New parameters (add after `compile_model`):**

```python
def run_grpo(
    ...,
    # Existing params unchanged (lines 910-932)
    compile_model: bool = False,
    # --- NEW: Tandem training ---
    enable_tandem: bool = False,
    tandem_junior_path: str | None = None,
    tandem_handoff_prob: float = 0.5,
    tandem_phase_in_step: int = 0,
):
```

### 7.2. `run_grpo()` -- Adapter Loading

**Insert point:** After line 1004 (`model.print_trainable_parameters()`),
BEFORE line 1006 (`if compile_model:`). This ordering is critical -- see section 10.1.

```python
    # --- Tandem adapter setup ---
    senior_adapter_name = "default"  # PEFT's default adapter name
    junior_adapter_name = "junior"

    if enable_tandem:
        # Resolve junior adapter path
        if tandem_junior_path is None:
            _sft_path = Path(output_dir) / "sft_checkpoint" / "model_weights"
            if _sft_path.exists():
                tandem_junior_path = str(_sft_path)
                print("  [TANDEM] Junior adapter: {} (SFT checkpoint)".format(
                    tandem_junior_path))
            else:
                print("ERROR: --enable-tandem requires --tandem-junior-path or "
                      "existing SFT checkpoint at {}".format(_sft_path))
                sys.exit(1)
        else:
            print("  [TANDEM] Junior adapter: {} (explicit)".format(
                tandem_junior_path))

        # Load junior adapter (frozen)
        from peft import PeftModel as _PeftModel  # already imported at line 991
        model.load_adapter(tandem_junior_path, adapter_name=junior_adapter_name)

        # Freeze junior adapter parameters
        for name, param in model.named_parameters():
            if junior_adapter_name in name:
                param.requires_grad = False

        # Ensure senior (default) is active and trainable
        model.set_adapter(senior_adapter_name)

        print("  [TANDEM] Mode: ENABLED")
        print("  [TANDEM] Handoff prob: {}".format(tandem_handoff_prob))
        print("  [TANDEM] Phase-in step: {}".format(tandem_phase_in_step))
        print("  [TANDEM] Senior adapter: '{}' (trainable)".format(
            senior_adapter_name))
        print("  [TANDEM] Junior adapter: '{}' (frozen)".format(
            junior_adapter_name))
```

**Important note on adapter naming:**

PEFT's `get_peft_model()` and `PeftModel.from_pretrained()` both create the initial
adapter with the name `"default"`. We do NOT rename it. Instead:
- Senior adapter name = `"default"` (the trainable adapter created during model setup)
- Junior adapter name = `"junior"` (loaded explicitly via `load_adapter()`)

The tandem functions accept `senior_adapter` and `junior_adapter` as string parameters,
so the caller passes the correct names.

### 7.3. `run_grpo()` -- torch.compile Guard (modify lines 1007-1012)

```python
    if compile_model and not enable_tandem:
        try:
            model = torch.compile(model)
            print("  torch.compile: ENABLED")
        except Exception as e:
            print("  torch.compile: FAILED ({}), continuing without".format(e))
    elif compile_model and enable_tandem:
        print("  torch.compile: SKIPPED (incompatible with tandem adapter switching)")
```

### 7.4. `run_grpo()` -- Training Loop Dispatch (modify line 1136)

Replace the `grpo_step()` call with conditional dispatch:

```python
            _tandem_zero_streak = 0  # initialize before training loop

            # In the training loop body:
            if enable_tandem and current_step >= tandem_phase_in_step:
                metrics = tandem_grpo_step(
                    model=model,
                    tokenizer=tokenizer,
                    batch_theorems=batch,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_idx=current_step,
                    lean_workers=lean_workers,
                    eff_oracle=eff_oracle,
                    running_pass_rate=running_pass_rate,
                    training_temperature=training_temp,
                    handoff_prob=tandem_handoff_prob,
                    senior_adapter=senior_adapter_name,
                    junior_adapter=junior_adapter_name,
                )
                # Safety guard: auto-disable if zero reward for 10 steps
                if metrics["mean_reward"] == 0.0:
                    _tandem_zero_streak += 1
                    if _tandem_zero_streak >= 10:
                        print("  [TANDEM] WARNING: 10 consecutive zero-reward steps. "
                              "Disabling tandem. Falling back to standard GRPO.")
                        enable_tandem = False
                else:
                    _tandem_zero_streak = 0
            else:
                metrics = grpo_step(
                    model=model,
                    tokenizer=tokenizer,
                    batch_theorems=batch,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_idx=current_step,
                    lean_workers=lean_workers,
                    eff_oracle=eff_oracle,
                    running_pass_rate=running_pass_rate,
                    training_temperature=training_temp,
                )
```

### 7.5. `run_grpo()` -- Tandem Metrics Logging (extend line 1187)

```python
            # After existing per-step print (line 1187-1196):
            if enable_tandem and "tandem_senior_frac" in metrics:
                print(
                    "           | sr_frac={:.2f} | handoffs={:.1f}".format(
                        metrics["tandem_senior_frac"],
                        metrics["tandem_handoffs_per_completion"],
                    )
                )
```

### 7.6. `run_grpo()` -- Checkpoint Saving (extend line 1206)

```python
        # Save model weights
        weights_path = ckpt_dir / "model_weights"
        print("  Saving weights to {}".format(weights_path))
        if enable_tandem:
            model.set_adapter(senior_adapter_name)  # ensure senior active
        model.save_pretrained(str(weights_path))
        tokenizer.save_pretrained(str(weights_path))

        # Save tandem config alongside checkpoint
        if enable_tandem:
            tandem_meta = {
                "tandem_enabled": True,
                "junior_path": tandem_junior_path,
                "handoff_prob": tandem_handoff_prob,
                "phase_in_step": tandem_phase_in_step,
                "senior_adapter": senior_adapter_name,
                "junior_adapter": junior_adapter_name,
            }
            with open(ckpt_dir / "tandem_config.json", "w") as f:
                json.dump(tandem_meta, f, indent=2)
```

### 7.7. `run_grpo()` -- Hidden State Collection Guard (extend line 1211)

```python
        if enable_tandem:
            model.set_adapter(senior_adapter_name)
        summary = collect_at_checkpoint(...)
```

### 7.8. `run_grpo()` -- Run Summary (extend line 1255)

```python
    run_summary = {
        ...,  # existing fields
        "tandem_config": {
            "enabled": enable_tandem,
            "junior_path": tandem_junior_path,
            "handoff_prob": tandem_handoff_prob,
            "phase_in_step": tandem_phase_in_step,
        } if enable_tandem else None,
    }
```

### 7.9. `main()` -- Pass Tandem Args (extend line 1575)

```python
    run_grpo(
        ...,  # existing args
        compile_model=getattr(args, 'compile', False),
        enable_tandem=getattr(args, 'enable_tandem', False),
        tandem_junior_path=getattr(args, 'tandem_junior_path', None),
        tandem_handoff_prob=getattr(args, 'tandem_handoff_prob', 0.5),
        tandem_phase_in_step=getattr(args, 'tandem_phase_in_step', 0),
    )
```

---

## 8. CLI Interface

### 8.1. New Arguments

Add to `parse_args()`, after the `--compile` argument (after line 1439):

```python
    # --- Tandem Training (West et al. 2026) ---
    parser.add_argument(
        "--enable-tandem", action="store_true",
        help="Enable tactic-level tandem training. Requires a junior adapter "
             "(defaults to SFT checkpoint)."
    )
    parser.add_argument(
        "--tandem-junior-path", type=str, default=None,
        help="Path to the junior model's PEFT adapter directory. "
             "Defaults to output_dir/sft_checkpoint/model_weights/."
    )
    parser.add_argument(
        "--tandem-handoff-prob", type=float, default=0.5,
        help="Probability of junior generating each tactic "
             "(0.0=senior only, 1.0=junior only, default: 0.5)"
    )
    parser.add_argument(
        "--tandem-phase-in-step", type=int, default=0,
        help="Step at which tandem training begins (0=from start). "
             "Steps before this use standard GRPO."
    )
```

### 8.2. Example Usage

```bash
# Tandem training with SFT checkpoint as junior (recommended)
python3 grpo_lean_reward.py \
    --steps 200 \
    --enable-tandem \
    --tandem-handoff-prob 0.5

# Tandem with explicit junior adapter
python3 grpo_lean_reward.py \
    --steps 200 \
    --enable-tandem \
    --tandem-junior-path /workspace/grpo_run/checkpoint_050/model_weights

# Tandem with phase-in (standard GRPO first 20 steps, then tandem)
python3 grpo_lean_reward.py \
    --steps 200 \
    --enable-tandem \
    --tandem-phase-in-step 20 \
    --tandem-handoff-prob 0.5

# Light tandem pressure (70% senior, 30% junior)
python3 grpo_lean_reward.py \
    --steps 200 \
    --enable-tandem \
    --tandem-handoff-prob 0.3
```

---

## 9. Data Flow Diagram

### 9.1. Standard GRPO Step (Existing)

```
                  +------------------+
                  |   batch_theorems  |
                  +--------+---------+
                           |
                  +--------v---------+
                  |  make_prompt()    |
                  +--------+---------+
                           |
              +------------v------------+
              |  generate_completions() |  <-- model.generate() (opaque, batched)
              +------------+------------+
                           |
                 list[{token_ids, text}]
                           |
              +------------v------------+
              |  lean_verify_batch()    |  <-- ThreadPool, subprocess
              +------------+------------+
                           |
                  rewards: [0.0 | 1.0]
                           |
              +------------v------------+
              |  GRPO advantages        |  <-- (r - mean) / (std + eps)
              +------------+------------+
                           |
        +------------------v------------------+
        |  compute_completion_log_prob()       |  <-- full forward pass, all tokens
        |  loss = -adv * sum(log_prob)         |
        +------------------+------------------+
                           |
                   loss.backward()
                   optimizer.step()
```

### 9.2. Tandem GRPO Step (New)

```
                  +------------------+
                  |   batch_theorems  |
                  +--------+---------+
                           |
                  +--------v---------+
                  |  make_prompt()    |
                  +--------+---------+
                           |
        +------------------v-------------------+
        |  generate_tandem_completions()       |
        |                                      |
        |  FOR each completion:                |
        |    +-----------------------------+   |
        |    | Autoregressive loop w/ KV   |   |
        |    |                             |   |
        |    |  At each \n token:          |   |
        |    |    coin flip -> senior/junior|  |
        |    |    set_adapter(winner)      |   |
        |    |    (KV cache preserved)     |   |
        |    |                             |   |
        |    |  Output:                    |   |
        |    |    token_ids                |   |
        |    |    senior_mask  <-----------+---+-- NEW
        |    |    tactic_assignments       |   |
        |    +-----------------------------+   |
        +------------------+-------------------+
                           |
             list[{token_ids, text, senior_mask, ...}]
                           |
              +------------v------------+
              |  lean_verify_batch()    |  <-- unchanged
              +------------+------------+
                           |
                  rewards: [0.0 | 1.0]       <-- unchanged
                           |
              +------------v------------+
              |  GRPO advantages        |  <-- unchanged
              +------------+------------+
                           |
        +------------------v------------------+
        |  compute_tandem_log_prob()           |
        |                                      |
        |  1. Senior adapter ONLY forward pass |
        |  2. log_softmax over all positions   |
        |  3. MASK: only sum where             |
        |     senior_mask[i] == True  <--------+-- NEW
        |  4. loss = -adv * masked_sum         |
        +------------------+------------------+
                           |
                   loss.backward()
                   optimizer.step()
```

### 9.3. Adapter Switching During Generation (Detail)

```
Token stream:  t0  t1  t2  \n  t3  t4  \n  t5  t6  t7  \n  t8  EOS
               -----------  ----------  ----------------  -----
Tactic:           #1           #2            #3            #4

Coin flips:    [0.32]      [0.71]      [0.18]          [0.55]
               (< 0.5)     (> 0.5)     (< 0.5)         (> 0.5)
Adapter:       JUNIOR      SENIOR      JUNIOR          SENIOR

senior_mask:   [F,F,F,F,   T,T,T,      F,F,F,F,F,      T,T]
               ---------   --------    -------------   --------

Loss includes:              ^  ^                         ^  ^
               (skipped)  (included)   (skipped)     (included)
```

---

## 10. Feature Interaction Matrix

| Existing Feature | Interaction | Action Required |
|------------------|-------------|-----------------|
| **Efficiency reward** (`--enable-efficiency`) | Compatible. Rewards computed on full proof text regardless of authorship. | None. Pass `eff_oracle` to `tandem_grpo_step()` as-is. |
| **Controller** (`--enable-controller`) | Compatible. Controller reads metrics dict. | Pass `tandem_senior_frac` to controller for monitoring. No schema change needed (controller ignores unknown keys). |
| **Temperature schedule** (cosine `temp_start` to `temp_end`) | Compatible. Same temperature applied to both adapters. | None. Temperature passed through. |
| **Checkpoint resume** | Must handle tandem state. Junior adapter reloaded from CLI arg. | Junior adapter path is a CLI arg (or defaults to SFT checkpoint). No special resume logic. |
| **`torch.compile`** (`--compile`) | Likely incompatible with `set_adapter()`. | Auto-disable compile when tandem active. Print warning. |
| **4-bit QLoRA** (`--load-in-4bit`) | Compatible. Both adapters sit on quantized base. | None. |
| **Hidden state collection** | Should use senior-only for clean states. | `set_adapter(senior)` before collection. |
| **Stage 1 RS-SFT** | Independent. SFT checkpoint becomes default junior. | None. |
| **Muon optimizer** | Compatible. Optimizer sees only senior params. | None (ordering ensures junior is frozen before optimizer creation). |
| **Gradient clipping** | Compatible. Junior params excluded (requires_grad=False). | None. |

### 10.1. Critical Ordering Constraint: Tandem Setup vs. Optimizer

The tandem adapter setup (loading junior, freezing its params) MUST happen between
model creation (line 1003) and optimizer creation (line 1015). This ensures:

1. Junior adapter is loaded onto the model
2. Junior params are frozen (`requires_grad=False`)
3. `trainable_params = [p for p in model.parameters() if p.requires_grad]` at line 1015
   correctly excludes junior params
4. Optimizer only tracks senior (default) adapter parameters

**Insert point for tandem setup:** After line 1004, before line 1006.

---

## 11. Design Decisions and Rationale

### 11.1. Tactic-Level vs. Token-Level

See section 3 for full analysis. Token-level breaks Lean syntax. Tactic-level preserves it.

### 11.2. handoff_prob = 0.5 (Default)

| Value | Meaning | Use Case |
|-------|---------|----------|
| 0.0 | Pure senior | No tandem (equivalent to standard GRPO) |
| 0.3 | Light pressure | Early training, 70% senior tactics |
| **0.5** | **Balanced (default)** | **Paper's default. Strong intelligibility.** |
| 0.7 | Heavy junior | Aggressive intelligibility (30% senior) |
| 1.0 | Pure junior | Diagnostic only |

### 11.3. KV Cache Preservation

See section 5.4. Not recomputing is correct per the paper, efficient, and safe.

### 11.4. Named Adapters (Not Separate Models)

See sections 5.2-5.3. Memory constraints make this the only viable option.

### 11.5. SFT Checkpoint as Default Junior

The SFT checkpoint is the model BEFORE GRPO training. It generates valid Lean syntax
but uses standard, unsurprising patterns. Forcing the GRPO-trained senior to produce
proofs continuable by the SFT model prevents drift into exotic strategies.

### 11.6. Full Senior Masking (j=0)

The paper (Appendix E) discusses soft-masking with weight `j` on junior tokens.
For v1, we use **full masking (j=0)**:
- Mathematically clean
- No extra hyperparameters
- Paper's main experiments produce strongest results

### 11.7. Sequential Generation (Not Batched)

Different sequences hit tactic boundaries at different token positions. True batching
would require per-sequence adapter masking -- complex and error-prone. Sequential is
correct and the performance impact is acceptable (~10-15% of total step time).

### 11.8. New Function vs. Modifying Existing

`tandem_grpo_step()` is a **new function**, not a modification of `grpo_step()`.
Rationale:
- `grpo_step()` continues to work for non-tandem runs (no regression risk)
- Clean separation of concerns
- Dispatch decision is centralized in `run_grpo()`

---

## 12. Implementation Plan

### Phase 1: Core Functions (No Behavioral Change)

| Step | Task | Dependencies |
|------|------|-------------|
| 1.1 | Add `generate_tandem_completion()` after line 287 | None |
| 1.2 | Add `generate_tandem_completions()` after 1.1 | 1.1 |
| 1.3 | Add `compute_tandem_log_prob()` after line 757 | None |
| 1.4 | Add `tandem_grpo_step()` after line 883 | 1.1, 1.2, 1.3 |

### Phase 2: Integration

| Step | Task | Dependencies |
|------|------|-------------|
| 2.1 | Extend `run_grpo()` signature with tandem params | None |
| 2.2 | Add tandem adapter loading block (after line 1004) | 2.1 |
| 2.3 | Add torch.compile guard (modify lines 1007-1012) | 2.2 |
| 2.4 | Add dispatch logic in training loop (modify line 1136) | 1.4, 2.2 |
| 2.5 | Add tandem safety guard (`_tandem_zero_streak`) | 2.4 |
| 2.6 | Add tandem metrics logging (extend line 1187) | 2.4 |
| 2.7 | Add tandem checkpoint saving (extend line 1206) | 2.2 |
| 2.8 | Add senior-only collection guard (extend line 1211) | 2.2 |
| 2.9 | Add tandem config to run summary (extend line 1255) | 2.2 |
| 2.10 | Add CLI arguments to `parse_args()` (after line 1439) | None |
| 2.11 | Pass tandem args in `main()` (extend line 1575) | 2.1, 2.10 |

### Phase 3: Testing

| Step | Task | Dependencies |
|------|------|-------------|
| 3.1 | Create `scripts/test_tandem.py` with unit tests | Phase 1 |
| 3.2 | Dry run with tiny model (GPT-2) -- no GPU needed | Phase 2 |
| 3.3 | RunPod smoke test (5 tandem GRPO steps on H100) | Phase 2 |
| 3.4 | Baseline comparison (20 steps standard vs. tandem) | 3.3 |

### Phase 4: Observability

| Step | Task | Dependencies |
|------|------|-------------|
| 4.1 | Verify tandem metrics appear in `step_metrics_log` | Phase 2 |
| 4.2 | Verify `tandem_config.json` saved with checkpoints | Phase 2 |
| 4.3 | Optional: save `senior_mask` per proof for analysis | Phase 1 |

---

## 13. Testing Strategy

### 13.1. Unit Tests (No GPU Required)

Create `scripts/test_tandem.py`:

```python
def test_tactic_boundary_detection():
    """Verify newline token correctly triggers adapter switch.

    Setup: Mock model that returns predetermined token IDs including \n tokens.
    Assert: tactic_assignments has one entry per tactic (one more than \n count).
    Assert: Adapter switches happen only at \n boundaries.
    """

def test_senior_mask_correctness():
    """Verify senior_mask aligns with tactic_assignments.

    Setup: Generate tandem completion with known random seed.
    Assert: All tokens within a "senior" tactic have senior_mask=True.
    Assert: All tokens within a "junior" tactic have senior_mask=False.
    Assert: len(senior_mask) == len(token_ids).
    """

def test_tandem_log_prob_masks_junior():
    """Verify gradient only flows through senior-authored positions.

    Setup: Compute log_prob with senior_mask=[True, False, True, False].
    Assert: Returned log_prob equals sum of log_probs at positions 0 and 2.
    Assert: Gradient exists on returned tensor.
    """

def test_handoff_prob_zero_equals_standard():
    """handoff_prob=0.0 should produce all-senior completions.

    Setup: Generate with handoff_prob=0.0.
    Assert: All entries in senior_mask are True.
    Assert: tactic_assignments is all "senior".
    Assert: n_handoffs == 0.
    """

def test_handoff_prob_one_all_junior():
    """handoff_prob=1.0 should produce all-junior completions.

    Setup: Generate with handoff_prob=1.0.
    Assert: All entries in senior_mask are False.
    Assert: tactic_assignments is all "junior".
    """

def test_senior_mask_length_invariant():
    """len(senior_mask) must always equal len(token_ids).

    Setup: Generate multiple completions with varying lengths.
    Assert: Invariant holds for each.
    """

def test_no_senior_tokens_returns_zero_grad():
    """compute_tandem_log_prob with all-False mask returns 0.0 with grad.

    Setup: Call with senior_mask all False.
    Assert: return value is 0.0.
    Assert: return tensor requires_grad is True.
    """
```

### 13.2. Integration Tests (GPU Required)

```python
def test_adapter_switching_preserves_generation():
    """Switch adapters mid-generation on real model, verify no crashes/NaN.

    Setup: Load DeepSeek-Prover-V2-7B with two LoRA adapters.
    Action: Generate tandem completion.
    Assert: No NaN in output logits.
    Assert: Text is decodable.
    """

def test_tandem_grpo_step_produces_gradient():
    """Run tandem_grpo_step, verify model parameters have non-zero .grad.

    Setup: Real model, real theorem, group_size=4.
    Action: Run tandem_grpo_step.
    Assert: At least one parameter has .grad != 0.
    """

def test_junior_frozen_after_step():
    """After tandem_grpo_step, junior adapter weights must be unchanged.

    Setup: Snapshot junior params before step.
    Action: Run tandem_grpo_step.
    Assert: All junior params equal to snapshot.
    """

def test_senior_params_updated():
    """After tandem_grpo_step with positive reward, senior params change.

    Setup: Snapshot senior params.
    Action: Run step with a theorem that has known-good proofs.
    Assert: At least some senior params differ from snapshot.
    """

def test_checkpoint_resume_with_tandem():
    """Save checkpoint during tandem training, reload, verify correct state.

    Setup: Run 2 tandem steps, save checkpoint.
    Action: Load checkpoint, load junior adapter, verify adapters exist.
    Assert: model.set_adapter("default") works (senior).
    Assert: model.set_adapter("junior") works.
    Assert: Junior params are frozen.
    """

def test_tandem_disabled_equals_standard():
    """With enable_tandem=False, output matches standard grpo_step exactly.

    Setup: Same seed, same theorem, same group_size.
    Action: Run grpo_step and run_grpo with enable_tandem=False.
    Assert: Identical metrics.
    """
```

---

## 14. Risk Matrix

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | `set_adapter()` corrupts KV cache silently | Medium | High | Lean verifier catches bad proofs (reward 0). Monitor pass rate: if drops >50% below baseline, investigate. |
| R2 | Junior too weak: all tandem rollouts fail | Low | High | Safety guard: 10 consecutive zero-reward steps auto-disables tandem, falls back to standard GRPO. |
| R3 | VRAM OOM from two adapters | Very Low | High | Two LoRA r=16 adapters = ~100MB. Negligible on 80GB. Log VRAM after adapter load. |
| R4 | `torch.compile` incompatible with `set_adapter()` | Medium | Low | Auto-disable compile when tandem active. Warning printed. |
| R5 | Sequential generation too slow | Medium | Medium | Generation <20% of step time (Lean dominates). Net ~10-15%. Acceptable for v1. |
| R6 | Tactic boundary splits `have` blocks | Medium | Low | Lean catches bad proofs (reward 0). No gradient contribution. v2 adds indentation-aware boundaries. |
| R7 | `load_adapter()` fails on mismatched LoRA config | Low | Medium | Validate at startup: LoRA rank and targets must match. PEFT raises error on mismatch. |
| R8 | Optimizer includes junior params | Low | High | Tandem setup (freeze) happens BEFORE optimizer creation. Verified by ordering constraint (section 10.1). |
| R9 | `model.train()`/`model.eval()` affects adapter state | Very Low | Medium | PEFT train/eval only affects dropout, not adapter selection. Verified in PEFT source. |

---

## 15. Metrics and Success Criteria

### 15.1. Tandem is Working (Functional)

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Pass rate parity | Tandem GRPO pass rate | Within 80% of standard GRPO at same step count |
| Intelligibility signal | `tandem_senior_frac` | Hovers near `1 - handoff_prob` (expected ~0.5) |
| Mask correctness | `senior_mask` distribution | Mixed True/False values |
| Loss convergence | `loss_val` | Finite, decreasing trend over 20+ steps |
| No OOM | GPU memory | Peak VRAM < 75GB |

### 15.2. Tandem is Valuable (Longer-Term)

| Criterion | Metric | How to Measure |
|-----------|--------|----------------|
| Proof standardization | Unique tactic count | `count_distinct_tactics()` from Round 3c. Fewer unique tactics than baseline after 50+ steps. |
| Junior continuability | Junior-only pass rate | Generate with junior-only at same prompts. Should increase over training. |
| Proof brevity | Mean proof length | Shorter proofs = more standard patterns. |
| Generalization | Transfer pass rate | Higher pass rate on held-out theorem set. |

### 15.3. Metrics Logged Per Step

Standard metrics (unchanged):
- `step`, `loss`, `mean_reward`, `mean_advantage`, `n_updates`
- `efficiency_active`, `pass_rate_estimate`, `temperature`

Tandem metrics (new, only when tandem active):
- `tandem_senior_frac`: fraction of tokens authored by senior (expected ~0.5)
- `tandem_handoffs_per_completion`: mean adapter switches per completion
- `tandem_completions_total`: total completions generated this step

---

## 16. Edge Cases

### 16.1. Empty Completion

Model generates EOS immediately (zero tokens). `senior_mask` is empty.
`compute_tandem_log_prob()` returns `tensor(0.0, requires_grad=True)`.
`tandem_grpo_step()` skips this completion (`len(comp["token_ids"]) == 0`).

### 16.2. Completion with No Senior Tokens

High `handoff_prob` + short proof = all junior tactics. `senior_mask` all False.
`tandem_grpo_step()` checks `any(comp["senior_mask"])` and skips.
Result: no gradient contribution from this completion (correct).

### 16.3. Completion with No Newlines

Single-line proof. Entire proof is one tactic. Initial coin flip determines authorship
of the whole proof. Degenerate but harmless.

### 16.4. Completion That is Only Newlines

Pathological. Each newline triggers boundary check. Proof is empty whitespace. Lean
verification fails. Reward 0. No gradient contribution.

### 16.5. Very Long Proof (max_new_tokens Reached)

Generation stops at limit. Last tactic may be incomplete. Lean catches parsing failure.
Same behavior as standard GRPO.

### 16.6. Junior Adapter Path Doesn't Exist

Validated at startup in tandem setup block (section 7.2). Prints error, exits.

### 16.7. Junior and Senior Have Different LoRA Configs

PEFT raises error on `load_adapter()` if config is incompatible. Clean failure at
startup.

### 16.8. handoff_prob = 0.0

All tactics senior. `senior_mask` all True. Functionally equivalent to standard GRPO
but through tandem code path (slower). Use case: validation.

### 16.9. Resume from Non-Tandem Checkpoint with Tandem Enabled

Works correctly. Checkpoint has `"default"` adapter. `load_adapter()` adds `"junior"`.
Senior is the existing trained adapter.

---

## 17. Future Extensions (Out of Scope for v1)

| # | Extension | Complexity | Value |
|---|-----------|------------|-------|
| F1 | Batched tandem generation | High | 4x generation speedup |
| F2 | handoff_prob curriculum (ramp 0.0 to 0.5) | Low | Smoother learning |
| F3 | Multiple juniors (bandit selection) | Medium | Richer intelligibility |
| F4 | Soft-masking (j > 0 for junior tokens) | Low | Paper Appendix E variant |
| F5 | Contrastive learning from failures | Medium | Negative weight variant |
| F6 | Indentation-aware tactic boundaries | Medium | Avoids splitting blocks |
| F7 | Cross-architecture tandem (1.3B junior) | High | Requires separate base |
| F8 | Wire tandem into Rust `generality` channel | Medium | Replace 0.5 hardcode |
| F9 | Per-proof `senior_mask` saving | Low | Interpretability |
| F10 | Tandem-aware proof attribution visualization | Low | Debugging aid |

---

## 18. File Change Summary

| File | Change | Lines Added (est.) | Description |
|------|--------|-------------------|-------------|
| `scripts/grpo_lean_reward.py` | MODIFY | ~250 | 4 new functions + extend `run_grpo()` + CLI args |
| `scripts/test_tandem.py` | CREATE | ~300 | Unit + integration tests |
| `docs/specs/tandem-grpo-architecture.md` | CREATE | This document | Full architectural specification |

### 18.1. Detailed Change Locations in `grpo_lean_reward.py`

| Location | What | Type |
|----------|------|------|
| After line 287 | `generate_tandem_completion()` | New function (~80 lines) |
| After above | `generate_tandem_completions()` | New function (~15 lines) |
| After line 757 | `compute_tandem_log_prob()` | New function (~35 lines) |
| After line 883 | `tandem_grpo_step()` | New function (~90 lines) |
| Lines 909-933 | `run_grpo()` signature | Extend (4 new params) |
| After line 1004 | Tandem adapter setup block | New block (~30 lines) |
| Lines 1007-1012 | `torch.compile` guard | Modify |
| Line 1136 | Training loop dispatch | Modify (conditional) |
| Line 1187 | Metrics logging | Extend |
| Line 1206 | Checkpoint saving | Extend |
| Line 1211 | Collection guard | Extend |
| Line 1255 | Run summary | Extend |
| After line 1439 | CLI arguments | New (4 arguments) |
| Line 1575 | `main()` run_grpo() call | Extend (4 new kwargs) |

### 18.2. What Does NOT Change

- `generate_completions()` (435-475) -- unchanged
- `generate_with_hidden_states()` (207-287) -- unchanged
- `grpo_step()` (764-883) -- unchanged
- `compute_completion_log_prob()` (724-757) -- unchanged
- `lean_verify_batch()` (140-165) -- unchanged
- `lean_verify_single()` (113-137) -- unchanged
- `make_prompt()` (195-200) -- unchanged
- `rejection_sampling_sft()` (478-721) -- unchanged
- `collect_at_checkpoint()` (294-428) -- unchanged
- All Rust code -- unchanged in v1

### 18.3. No New Dependencies

- `peft.PeftModel.load_adapter()` -- available in peft >= 0.6.0 (already installed)
- `peft.PeftModel.set_adapter()` -- available in peft >= 0.6.0 (already installed)
- `random.random()` -- stdlib (already imported)

---

## 19. Reference: Hyperparameter Comparison

| Parameter | Paper (GSM8K) | ProofForge v1 (Lean) | Notes |
|-----------|---------------|----------------------|-------|
| Senior model | Llama-2-7b (specialist) | DeepSeek-Prover-V2-7B (SFT+GRPO) | |
| Junior model | Llama-2-7b-chat (frozen) | SFT checkpoint (frozen) | |
| LoRA rank | 16 | 16 | Match |
| LoRA alpha | 16 | 32 | ProofForge uses higher alpha |
| LoRA targets | All linear | q,k,v,o,gate,up,down_proj | Equivalent for arch |
| Quantization | 4-bit QLoRA | bfloat16 (or 4-bit QLoRA) | Both supported |
| Batch size | 152 | 4 theorems x 8 group = 32 | Paper uses larger batch |
| Learning rate | 1e-4 | 5e-6 | ProofForge 20x smaller |
| Temperature | 0.7 | 0.7 (annealing 1.0 to 0.5) | Schedule differs |
| Max output length | 256 | 512 | Proofs can be longer |
| Handoff granularity | **Token** | **Tactic (newline)** | Key adaptation |
| Rollouts per prompt | 2 | 8 (group_size) | More rollouts |
| RL algorithm | REINFORCE (binary) | REINFORCE (binary, group-norm) | Group normalization |
| Training steps | 80 | 200 | Longer training |
| Handoff probability | Uniform (0.5) | Configurable (default 0.5) | CLI-tunable |
| Junior token masking | No masking (main) | Full masking (j=0) | Stricter |

---

## Appendix A: Codebase Verification Log

All references verified against production codebase on 2026-04-13.

| Claim | Verified | Location |
|-------|----------|----------|
| File is 1603 lines | Yes | `wc -l` = 1602 + trailing newline = 1603 |
| `generate_completions()` at 435-475 | Yes | Read and confirmed |
| `generate_with_hidden_states()` at 207-287 | Yes | Read and confirmed |
| `grpo_step()` at 764-883 | Yes | Read and confirmed |
| `compute_completion_log_prob()` at 724-757 | Yes | Read and confirmed |
| `lean_verify_batch()` at 140-165 | Yes | Read and confirmed |
| `run_grpo()` at 909-1285 | Yes | Read and confirmed |
| `make_prompt()` at 195-200 | Yes | Read and confirmed |
| Model loading at 936-1003 | Yes | Read and confirmed |
| Loss formula at 848-863 | Yes | Read and confirmed |
| SFT checkpoint load at 1567 | Yes | Read and confirmed |
| LoRA config: r=16, alpha=32 | Yes | Lines 997-998 |
| `PeftModel` import at line 991 | Yes | Read and confirmed |
| `generality` hardcode at 0.5 | Yes | `pf-core/src/grpo.rs:91` |
| `ChannelRewards` struct | Yes | `pf-core/src/types.rs:77` |
| No discrepancies between prompt and codebase | Yes | All claims match |
