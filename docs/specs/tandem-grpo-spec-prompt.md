# Architectural Specification Prompt: Tactic-Level Tandem GRPO for ProofForge

> **Purpose:** This document is the complete, self-contained prompt for an Architecture
> Sentinel (AS) mode session to produce the full architectural specification for
> integrating tactic-level tandem training into ProofForge's GRPO pipeline.
>
> **Source paper:** "Tandem Training for Language Models" — West, Anderson, Kamar, Horvitz
> (EPFL / U of Toronto / Microsoft, arXiv:2510.13551v2, Jan 2026)

---

## PROMPT START

You are the Architecture Sentinel for ProofForge, a Rust + Python system that trains
DeepSeek-Prover-V2-7B with GRPO to generate Lean 4 proofs. Your task is to produce a
**complete architectural specification** for integrating **tactic-level tandem training**
into the existing GRPO pipeline.

This spec must be implementation-ready: a kraken agent reading only this spec and the
current codebase should be able to implement the feature without ambiguity.

---

### 1. BACKGROUND: What Is Tandem Training?

Tandem training (West et al., 2026) modifies RL rollouts by randomly interleaving
tokens from a frozen "junior" (weaker) model during the trainable "senior" model's
generation. The key idea:

- During generation, at each step, a coin flip decides whether the **senior** (trainable)
  or the **junior** (frozen) generates the next token.
- Rewards are computed on the **full tandem rollout** (mixed senior + junior tokens).
- Because rewards are only high when the *tandem pair* co-produces correct solutions,
  the senior is incentivized to produce reasoning that the junior can *continue* —
  operationalizing "intelligibility" as **handoff robustness**.
- The loss gradient only flows through **senior-authored tokens** (junior tokens are
  detached / masked from the policy gradient).

**Formal definition of handoff robustness:** A senior model M_sen's solution is
intelligible to a junior model M_jun if M_jun can take over during randomly assigned
portions of the solution path without causing task failure.

**Key result:** On GSM8K, notational jargon drops 99% → 0% within 20 gradient updates.
Accuracy stays above junior baseline. Works with any RL algorithm (paper tested
REINFORCE; explicitly states compatibility with GRPO, PPO, etc.).

---

### 2. CRITICAL ADAPTATION: Tactic-Level, Not Token-Level

The paper interleaves at the **token level** for natural language math. This will NOT
work for Lean 4 proofs because:

- Lean 4 has rigid syntax. A random junior token mid-tactic produces unparseable garbage.
- Every tandem rollout would fail → zero reward signal → model learns nothing.
- The meaningful unit of proof composition is a **tactic**, not a token.

**Our adaptation:** Interleave at the **tactic level.** At each tactic boundary
(newline / semicolon in the generated proof), flip a coin: the senior or junior
generates the **next full tactic** (autoregressively, token by token, until the next
boundary or EOS). This preserves syntactic validity while enforcing handoff robustness
at the granularity that matters for proofs.

A "tactic boundary" is defined as:
1. A newline character (`\n`) that is NOT inside a `by`, `do`, `where`, or `have` block
   (i.e., not a continuation line that starts with more indentation than the previous tactic).
2. A semicolon (`;`) used as tactic separator in term-mode proofs.
3. The beginning of generation (the first tactic is always assigned by coin flip).

**Simplification for v1:** Treat every `\n` as a tactic boundary. DeepSeek-Prover-V2-7B
generates proofs as newline-separated tactic sequences in nearly all cases. Nested
`have`/`suffices` blocks that span multiple lines will occasionally get split mid-block,
but this is acceptable for v1 — the Lean verifier will catch any resulting failures, and
those rollouts simply receive reward 0.

---

### 3. EXISTING CODEBASE (What You're Modifying)

#### 3.1. File: `scripts/grpo_lean_reward.py` (1603 lines)

This is the **production training script** used on RunPod. It is self-contained Python —
no Rust orchestration needed.

**Key functions and their roles:**

| Function | Lines | Role |
|----------|-------|------|
| `generate_completions()` | 435–475 | Batched generation via `model.generate()`. Returns `[{token_ids, text}]`. Uses HuggingFace's `.generate()` — **opaque**, cannot interleave mid-generation. |
| `generate_with_hidden_states()` | 207–287 | **Manual autoregressive loop** with KV cache. Token-by-token control. Records hidden states. This is our skeleton for tandem generation. |
| `grpo_step()` | 764–883 | One GRPO gradient update. Calls `generate_completions()`, verifies with Lean, computes advantages, accumulates `-adv * log_prob` for positive-advantage completions. |
| `compute_completion_log_prob()` | 724–757 | Forward pass to get `sum(log_prob)` for a completion. Used in the loss term. Takes `(model, tokenizer, prompt, completion_token_ids)`. |
| `lean_verify_batch()` | 140–165 | Parallel Lean 4 verification via `ThreadPoolExecutor`. |
| `run_grpo()` | 909–1200+ | Outer training loop. Loads model (LoRA/PEFT), creates optimizer, iterates over steps calling `grpo_step()`. |
| `make_prompt()` | 195–200 | Builds prompt: `"Complete this Lean 4 proof. Output ONLY the tactic(s).\n\n" + statement + "\n"` |

**Model loading (run_grpo, lines 936–1003):**
- Detects PEFT checkpoint via `adapter_config.json`
- Loads base model with `AutoModelForCausalLM.from_pretrained()` (bfloat16 or 4-bit QLoRA)
- Wraps with LoRA: `r=16, alpha=32, targets=[q,k,v,o,gate,up,down]_proj`
- Flash Attention 2 if available, else SDPA

**Reward computation:**
- Binary: `1.0 if lean_verify_single(statement, proof) else 0.0`
- Optional efficiency reward (phase-in after step 50, blends proof length penalty)
- Elegance/generality channels exist in Rust types but are NOT wired into this script

**Loss formula (grpo_step, lines 848–863):**
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

**Checkpoint handling:**
- Stage 1 SFT checkpoint saved to `output_dir/sft_checkpoint/model_weights/`
- Stage 2 loads from SFT checkpoint if available (line 1567)
- Checkpoints saved at scheduled steps: `output_dir/checkpoint_{step:03d}/model_weights/`

#### 3.2. Key Existing Pattern: `generate_with_hidden_states()` (lines 207–287)

This function already implements the exact pattern we need:

```python
@torch.no_grad()
def generate_with_hidden_states(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, layer_idx=-1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    past_key_values = None

    for _step in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break

        current_ids = next_token.unsqueeze(0)
        # ... attention mask update ...
```

The tandem generation function will follow this structure but alternate between two
models at tactic boundaries.

#### 3.3. Hardware Constraints (RunPod H100 80GB)

- **Single GPU:** 80GB VRAM
- **Base model:** DeepSeek-Prover-V2-7B = ~14GB in bfloat16, ~4GB in 4-bit QLoRA
- **LoRA adapters:** ~50MB (negligible)
- **Two models simultaneously:** The senior (trainable LoRA) and junior (frozen) share
  the same base model weights. Only the LoRA adapters differ. We do NOT need to load
  two full copies of the base model.

**Critical architectural decision:** Both senior and junior are LoRA adapters on the
same base model. We can:
- Option A: Load two `PeftModel` instances wrapping the same base → 2× LoRA memory (~100MB), acceptable
- Option B: Use `model.set_adapter("senior")` / `model.set_adapter("junior")` — PEFT supports named adapters. Single `PeftModel` with two adapter sets. Switch between them at tactic boundaries. **Zero additional memory.**
- Option C: Load junior as a separate full model → doubles VRAM → NOT feasible in bfloat16 on 80GB

**Recommendation: Option B (named adapters).** This is PEFT's native multi-adapter
support. The base model weights are shared. Switching adapters is a pointer swap on the
LoRA matrices — effectively free.

```python
from peft import PeftModel

# Load base
base_model = AutoModelForCausalLM.from_pretrained(...)

# Load senior adapter (trainable)
model = PeftModel.from_pretrained(base_model, senior_path, adapter_name="senior", is_trainable=True)

# Load junior adapter (frozen)
model.load_adapter(junior_path, adapter_name="junior")

# Switch
model.set_adapter("senior")  # for senior tactic generation
model.set_adapter("junior")  # for junior tactic generation
```

**KV cache complication:** When switching adapters mid-generation, the KV cache from
prior tokens was computed with the *other* adapter's weights. Options:
1. **Invalidate cache on switch** (recompute from scratch) — correct but slow
2. **Keep cache across switches** — the KV cache reflects the base model + whichever
   adapter was active. For LoRA rank 16 on a 7B model, the adapter perturbation to
   attention weights is small. The paper (West et al.) does NOT recompute cache on
   switches and reports no issues. Lean verification catches any failures.
3. **Hybrid:** Keep cache but recompute the last N tokens on switch

**Recommendation for v1: Option 2 (keep cache).** The paper doesn't recompute, Lean
verifier catches failures, and the computational cost of cache invalidation would make
tandem training 2-5× slower (defeating the purpose of being "zero extra compute cost").

---

### 4. ARCHITECTURE SPECIFICATION

#### 4.1. New Function: `generate_tandem_completion()`

**Location:** `scripts/grpo_lean_reward.py`, after `generate_with_hidden_states()`

**Signature:**
```python
@torch.no_grad()
def generate_tandem_completion(
    model: PeftModel,          # Multi-adapter PEFT model
    tokenizer,
    prompt: str,
    senior_adapter: str = "senior",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,  # Probability of junior generating each tactic
    newline_token_id: int | None = None,  # Precomputed \n token ID
) -> dict:
    """Generate a single tandem completion.

    At each tactic boundary (newline), flips a coin with probability handoff_prob
    to decide whether the senior or junior generates the next tactic.

    Returns:
        {
            "token_ids": list[int],       # All generated token IDs
            "text": str,                  # Decoded proof text
            "senior_mask": list[bool],    # True for tokens generated by senior
            "tactic_assignments": list[str],  # ["senior", "junior", ...] per tactic
        }
    """
```

**Algorithm:**
```
1. Tokenize prompt → current_ids, attention_mask
2. Initialize: past_key_values = None, all_tokens = [], senior_mask = []
3. Determine newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
4. current_generator = coin_flip(handoff_prob) → "senior" or "junior"
5. model.set_adapter(current_generator)
6. LOOP for max_new_tokens steps:
   a. Forward pass with KV cache → logits
   b. Sample next token from logits/temperature
   c. Record: all_tokens.append(tok_id), senior_mask.append(current_generator == "senior")
   d. If tok_id == eos_token_id → break
   e. If tok_id == newline_token_id:
      - This is a tactic boundary
      - Flip coin: current_generator = "senior" if random() > handoff_prob else "junior"
      - model.set_adapter(current_generator)
      - (KV cache is NOT invalidated — see §3.3 rationale)
   f. Feed next token → continue
7. Decode all_tokens → text
8. Return {token_ids, text, senior_mask, tactic_assignments}
```

**Why `senior_mask` matters:** When computing the policy gradient, we must ONLY
backpropagate through senior-authored tokens. Junior tokens are sampled from a frozen
distribution — including them in the loss would be mathematically incorrect (trying to
update the senior's policy based on tokens it didn't generate). The `senior_mask` tells
`compute_tandem_log_prob()` which token positions to include.

#### 4.2. New Function: `generate_tandem_completions()`

**Wrapper for batched tandem generation (sequential, not GPU-batched):**

```python
def generate_tandem_completions(
    model: PeftModel,
    tokenizer,
    prompt: str,
    group_size: int,
    senior_adapter: str = "senior",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,
) -> list[dict]:
    """Generate group_size tandem completions sequentially."""
```

**Why sequential, not batched?** The standard `generate_completions()` uses
`model.generate(num_return_sequences=group_size)` which is GPU-batched. But tandem
generation requires per-sequence adapter switching at different tactic boundaries —
sequences in a batch won't hit boundaries at the same token position. True batching
would require masking logic per sequence in the batch. For v1, sequential generation
is correct and simpler. Optimization to batched tandem is a v2 concern.

**Performance note:** Sequential generation of `group_size=8` completions at
`max_new_tokens=512` on H100 with 7B model ≈ 8 × 2s = 16s per theorem. The existing
`generate_completions()` does the same in ≈ 4s (GPU-batched). This is a 4× slowdown
in generation, but generation is typically <20% of step time (Lean verification
dominates at 30s timeout × lean_workers). Net impact: ~10-15% step time increase.

#### 4.3. New Function: `compute_tandem_log_prob()`

**Modified version of `compute_completion_log_prob()` that masks junior tokens:**

```python
def compute_tandem_log_prob(
    model: PeftModel,
    tokenizer,
    prompt: str,
    completion_token_ids: list[int],
    senior_mask: list[bool],
    senior_adapter: str = "senior",
) -> torch.Tensor:
    """Compute sum of log probabilities for SENIOR-AUTHORED tokens only.

    The forward pass runs over the full sequence (prompt + all completion tokens),
    but only senior-authored token positions contribute to the returned log_prob sum.

    The model MUST have the senior adapter active (since we're computing the senior's
    policy gradient).
    """
```

**Implementation:**
```
1. model.set_adapter(senior_adapter)
2. Construct full_ids = [prompt_ids] + [completion_token_ids]
3. Forward pass → logits  (single forward, NOT per-token)
4. Extract pred_logits at completion positions
5. Compute log_softmax → per-token log probs
6. Apply senior_mask: only sum log_probs where senior_mask[i] == True
7. Return sum (scalar tensor, requires grad)
```

**Critical detail:** During the forward pass for log_prob computation, the model uses
the **senior adapter only** (not switching between adapters per token). This is correct:
we're computing π_senior(a|s) — the senior's probability of the *entire* completion
sequence — but only backpropagating through positions where the senior actually chose
the token. The senior adapter was active when those tokens were sampled, so the
log_prob is well-defined.

#### 4.4. Modified Function: `grpo_step()` → `tandem_grpo_step()`

**Create a new function (do NOT modify the existing `grpo_step()`):**

```python
def tandem_grpo_step(
    model: PeftModel,       # Multi-adapter model
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
```

**Changes from `grpo_step()`:**
1. Calls `generate_tandem_completions()` instead of `generate_completions()`
2. Verification is identical — `lean_verify_batch()` doesn't care who generated what
3. Reward computation is identical — binary correctness from Lean
4. Advantage computation is identical — group normalization
5. Loss computation calls `compute_tandem_log_prob()` instead of `compute_completion_log_prob()`, passing the `senior_mask`
6. Reports additional metrics: `tandem_senior_frac` (fraction of tokens authored by senior), `tandem_handoffs` (number of adapter switches per completion)

**Metrics to track (returned in dict):**
```python
{
    "step": step_idx,
    "loss": loss_val,
    "mean_reward": mean_reward,
    "mean_advantage": mean_adv,
    "n_updates": n_updates,
    "tandem_senior_frac": avg_senior_fraction,      # NEW
    "tandem_handoffs_per_completion": avg_handoffs,  # NEW
    "tandem_junior_pass_rate": junior_only_pass,     # NEW (diagnostic)
}
```

#### 4.5. Modified Function: `run_grpo()` → Extended with Tandem Support

**Do NOT create a new function. Extend `run_grpo()` with new parameters:**

```python
def run_grpo(
    ...,
    # New tandem parameters
    enable_tandem: bool = False,
    tandem_junior_path: str | None = None,  # Path to junior adapter checkpoint
    tandem_handoff_prob: float = 0.5,
    tandem_phase_in_step: int = 0,  # Step at which to start tandem (0 = from start)
):
```

**Modifications inside `run_grpo()`:**

1. **After LoRA model creation (line ~1003):** If `enable_tandem`:
   ```python
   # Rename current adapter to "senior"
   # Load junior adapter from tandem_junior_path
   model.load_adapter(tandem_junior_path, adapter_name="junior")
   # Freeze junior adapter
   for name, param in model.named_parameters():
       if "junior" in name:
           param.requires_grad = False
   # Ensure senior is active and trainable
   model.set_adapter("senior")
   ```

2. **In the training loop:** If `enable_tandem` and `step >= tandem_phase_in_step`:
   call `tandem_grpo_step()` instead of `grpo_step()`.

3. **Checkpoint saving:** Only save the senior adapter weights. The junior adapter is
   frozen and unchanging — saving it wastes disk space.

4. **Logging:** Print tandem metrics alongside existing metrics.

#### 4.6. CLI Arguments

**Add to `argparse` in `main()`:**

```python
# Tandem training
parser.add_argument("--enable-tandem", action="store_true",
    help="Enable tactic-level tandem training (West et al. 2026)")
parser.add_argument("--tandem-junior-path", type=str, default=None,
    help="Path to junior model's PEFT adapter. Defaults to SFT checkpoint.")
parser.add_argument("--tandem-handoff-prob", type=float, default=0.5,
    help="Probability of junior generating each tactic (0.0=senior only, 1.0=junior only)")
parser.add_argument("--tandem-phase-in-step", type=int, default=0,
    help="Step at which to begin tandem training (0=from start)")
```

**Default behavior:** If `--enable-tandem` is set without `--tandem-junior-path`, use
the SFT checkpoint (`output_dir/sft_checkpoint/model_weights/`) as the junior. This is
the natural choice: the junior is the model *before* GRPO training.

#### 4.7. Interaction with Existing Features

| Feature | Interaction | Action Required |
|---------|-------------|-----------------|
| Efficiency reward (`--enable-efficiency`) | Compatible. Efficiency rewards are computed on the full proof text regardless of who generated which tactic. | None |
| Controller (`--enable-controller`) | Compatible. Controller adjusts hyperparameters based on metrics, which tandem doesn't affect. | Pass `tandem_senior_frac` to controller for monitoring. |
| Temperature schedule | Compatible. Temperature applies to whoever is generating (senior or junior). | Use same temperature for both. |
| Checkpoint resume | Must handle tandem state. When resuming from checkpoint with `--enable-tandem`, reload junior adapter. | Add junior adapter reload in checkpoint resume path. |
| `torch.compile` | May conflict with `set_adapter()` dynamic dispatch. | Test. If compile breaks adapter switching, disable compile when tandem is active. |
| 4-bit QLoRA (`--load-in-4bit`) | Compatible. Both adapters sit on the quantized base model. | None |
| Hidden state collection (`collect_at_checkpoint`) | Collection should use senior-only generation (no tandem) to get clean hidden states. | Set `model.set_adapter("senior")` before collection, skip tandem. |

---

### 5. DESIGN DECISIONS AND RATIONALE

#### 5.1. Why Tactic-Level, Not Token-Level?

| Approach | Pros | Cons |
|----------|------|------|
| Token-level (paper's approach) | Finest granularity, maximally faithful to paper | Lean 4 syntax breaks instantly. Nearly 100% rollout failure rate. Zero learning signal. |
| Tactic-level (our approach) | Preserves syntactic validity per-tactic. Meaningful unit of proof composition. Junior can contribute complete, well-formed tactics. | Coarser granularity. Less pressure per token. But proofs are 5-20 tactics long, so ~50% handoff_prob still means many switches. |
| Block-level (multi-tactic) | Even coarser. Less tandem pressure. | Too coarse for short proofs (3-5 tactics). |

#### 5.2. Why handoff_prob = 0.5?

The paper uses uniform random at each token. For tactic-level, 0.5 means each tactic
has equal probability of being authored by senior or junior. For a proof with 10 tactics,
this means ~5 senior, ~5 junior — strong intelligibility pressure.

The `--tandem-handoff-prob` flag allows tuning:
- 0.0 = pure senior (no tandem, equivalent to standard GRPO)
- 0.3 = light tandem pressure (70% senior, good for early training)
- 0.5 = balanced (paper's default)
- 0.7 = heavy junior pressure (30% senior, aggressive intelligibility)

**Curriculum schedule (v2):** Start at 0.0, linearly increase to 0.5 over the first N
steps. This lets the senior learn basic proof structure before introducing handoff
pressure.

#### 5.3. Why NOT Recompute KV Cache on Adapter Switch?

1. The paper doesn't recompute and reports no issues.
2. LoRA rank 16 on a 7B model perturbs attention weights by <1%. The KV cache from
   the other adapter is a close approximation.
3. Recomputing means re-forwarding all prior tokens on every switch — for a 512-token
   sequence with 10 switches, that's ~5× more computation.
4. Lean verification is the ground truth. If stale cache causes a bad proof, it gets
   reward 0 and doesn't contribute to the gradient.

#### 5.4. Why Named Adapters, Not Two Separate Models?

- **Memory:** Two full 7B models = 28GB bfloat16 = won't fit alongside optimizer state
  and activations on 80GB. Two LoRA adapters = ~100MB total.
- **Speed:** `set_adapter()` is a pointer swap. Loading a second model takes 30s+.
- **Simplicity:** Single model object, single tokenizer, single optimizer.

#### 5.5. Why the SFT Checkpoint as Junior?

The SFT checkpoint represents the model's "common knowledge" before GRPO
specialization. It can generate valid Lean syntax (that's what SFT taught it) but uses
standard, unsurprising patterns. Forcing the GRPO-trained senior to produce proofs
continuable by the SFT model means the senior can't drift into exotic strategies.

Alternative juniors (v2):
- Base DeepSeek-Prover-V2-7B (no LoRA) — even weaker, tests raw intelligibility
- Earlier GRPO checkpoint — tests incremental intelligibility
- Different model (e.g., 1.3B) — requires separate base model, much more complex

#### 5.6. Policy Gradient Correctness: Why Mask Junior Tokens?

In standard GRPO, the loss is:
```
L = -A(s) * Σ_t log π(a_t | s, a_{<t})
```
where all tokens a_t are sampled from π (the senior).

In tandem GRPO, some tokens a_t are sampled from π_junior, not π_senior. Including
them in the loss would compute gradients for:
```
∂/∂θ log π_senior(a_t | ...) where a_t ~ π_junior
```
This is off-policy and introduces bias. The correct loss masks junior tokens:

```
L = -A(s) * Σ_{t ∈ senior_tokens} log π_senior(a_t | s, a_{<t})
```

The paper (Section 4.3, "soft-masking" variant in Appendix E) discusses this:
- Main experiments do NOT mask junior tokens (simpler, still works)
- Appendix E shows masking with multiplicative factor j<1 on junior token log-probs

**Our recommendation for v1:** Full masking (j=0). This is mathematically cleaner and
avoids introducing the (c, j) hyperparameter tuning the paper explores. If v1 works,
v2 can experiment with soft-masking (0 < j < 1).

---

### 6. IMPLEMENTATION PLAN (Ordered Steps)

#### Phase 1: Core Functions (No Behavioral Change)

1. **Add `generate_tandem_completion()`** — single tandem completion with tactic-level switching
2. **Add `generate_tandem_completions()`** — sequential wrapper for group_size completions
3. **Add `compute_tandem_log_prob()`** — masked log-prob computation
4. **Add `tandem_grpo_step()`** — full tandem GRPO step function
5. **Unit test:** Generate tandem completions with a mock model, verify `senior_mask` correctness

#### Phase 2: Integration

6. **Extend `run_grpo()`** — add tandem parameters, adapter loading, conditional dispatch
7. **Add CLI arguments** — `--enable-tandem`, `--tandem-junior-path`, etc.
8. **Extend checkpoint logic** — save only senior adapter, reload junior on resume

#### Phase 3: Validation

9. **Dry run (no GPU):** Verify adapter loading/switching with a tiny model (e.g., GPT-2)
10. **RunPod smoke test:** Run 5 tandem GRPO steps on H100, verify:
    - No OOM
    - `senior_mask` has mixed True/False values
    - Lean verification produces some successes (reward > 0)
    - Loss is finite and decreasing
    - Adapter switching doesn't corrupt generation
11. **Baseline comparison:** Run 20 steps standard GRPO vs 20 steps tandem GRPO, compare:
    - Pass rate (should be similar ± 5%)
    - Proof length distribution (tandem should be shorter / more standard)
    - Tactic diversity (tandem should use fewer unique tactics)

#### Phase 4: Observability

12. **Logging:** Per-step: `tandem_senior_frac`, `tandem_handoffs`, `tandem_junior_alone_pass_rate`
13. **Checkpoint metadata:** Save `tandem_config` dict alongside model weights
14. **Proof attribution:** Optionally save `senior_mask` per proof for post-hoc analysis

---

### 7. TESTING STRATEGY

#### 7.1. Unit Tests (No GPU)

```python
def test_tactic_boundary_detection():
    """Verify newline token correctly triggers adapter switch."""

def test_senior_mask_correctness():
    """Generate tandem completion, verify mask aligns with adapter assignments."""

def test_tandem_log_prob_masks_junior():
    """Compute log_prob with mask, verify gradient only flows through senior tokens."""

def test_handoff_prob_zero_equals_standard():
    """handoff_prob=0 should produce identical behavior to standard generation."""

def test_handoff_prob_one_all_junior():
    """handoff_prob=1 should produce all-junior completions (senior_mask all False)."""
```

#### 7.2. Integration Tests (GPU)

```python
def test_adapter_switching_preserves_generation():
    """Switch adapters mid-generation, verify no crashes or NaN outputs."""

def test_tandem_grpo_step_produces_gradient():
    """Run tandem_grpo_step, verify model.parameters() have non-zero .grad."""

def test_junior_frozen():
    """After tandem_grpo_step, verify junior adapter weights unchanged."""

def test_checkpoint_resume_with_tandem():
    """Save checkpoint, reload, verify tandem state (adapters, masks) correct."""
```

---

### 8. RISK MATRIX

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `set_adapter()` breaks KV cache silently | Medium | High (garbage proofs, wasted compute) | Lean verifier catches bad proofs. Monitor pass rate — if it drops >50% vs baseline, investigate. |
| Junior too weak → all tandem rollouts fail → zero gradient | Low (SFT checkpoint should be competent) | High (no learning) | Guard: if tandem pass rate < 2% for 10 consecutive steps, auto-disable tandem and fall back to standard GRPO. Log warning. |
| VRAM OOM from two adapters | Very Low (~100MB extra) | High (crash) | Monitor GPU memory at startup. Two LoRA r=16 adapters ≈ 100MB — negligible on 80GB. |
| `torch.compile` incompatible with `set_adapter()` | Medium | Low (just disable compile) | Check at startup. If `enable_tandem and compile_model`, warn and skip compile. |
| Sequential generation too slow | Medium (4× slower generation) | Medium (longer per-step time) | Generation is <20% of step time (Lean verification dominates). Net impact ~10-15%. Acceptable. |
| Tactic boundary detection wrong (splits `have` blocks) | Medium | Low (Lean catches it, reward=0) | v1 accepts this. v2 can add indentation-aware boundary detection. |

---

### 9. METRICS & SUCCESS CRITERIA

**Tandem training is working if:**

1. **Pass rate parity:** Tandem GRPO pass rate within 80% of standard GRPO pass rate at same step count
2. **Intelligibility signal:** `tandem_senior_frac` hovers near `1 - handoff_prob` (expected 0.5 at default)
3. **Proof standardization:** After 50+ tandem steps, proofs use fewer unique tactic types than baseline (measured by `count_distinct_tactics()` from existing Round 3c integration)
4. **Junior continuability:** `tandem_junior_alone_pass_rate` (proofs generated by junior alone at same prompts) should increase over training — meaning the senior's proofs are becoming more compatible with the junior's capabilities

**Tandem training is valuable if (longer-term):**

5. Proofs are shorter on average (fewer tactics needed when using standard patterns)
6. Proofs generalize better to unseen theorems (transfer learning benchmark)
7. The `generality` reward channel (currently hardcoded 0.5) can be replaced by tandem handoff robustness as a metric

---

### 10. FUTURE EXTENSIONS (Out of Scope for v1)

1. **Batched tandem generation:** Per-sequence adapter masking for GPU-parallel tandem rollouts
2. **Curriculum over handoff_prob:** Start at 0.0, linearly ramp to 0.5 over first 20 steps
3. **Junior pool:** Multiple juniors at different competence levels, selected by bandit
4. **Soft-masking:** Include junior tokens in loss with multiplicative weight j ∈ (0, 1)
5. **Contrastive learning from failures:** Down-weight rollouts where handoff caused failure (paper's Appendix E, modification 1: negative weight c < 0 for failed rollouts)
6. **Indentation-aware tactic boundaries:** Parse proof structure to avoid splitting `have`/`suffices`/`calc` blocks
7. **Cross-architecture tandem:** Use a smaller prover (1.3B) as junior — requires separate base model
8. **SOS integration:** Wire tandem handoff robustness into Rust-side `compute_channel_rewards()` as the `generality` channel, replacing the 0.5 hardcode

---

### 11. FILE CHANGE SUMMARY

| File | Change Type | Description |
|------|-------------|-------------|
| `scripts/grpo_lean_reward.py` | MODIFY | Add 4 new functions, extend `run_grpo()` signature and body, add CLI args |
| `scripts/test_tandem.py` | CREATE | Unit + integration tests for tandem functions |
| `docs/specs/tandem-grpo-architecture.md` | CREATE | This spec (output of AS mode) |

**No Rust changes in v1.** The Rust-side types (`ChannelRewards`, `RewardOracle`) and
the `generality` channel integration are v2/Phase 4 work.

**No new dependencies.** PEFT's `load_adapter()` and `set_adapter()` are available in
peft>=0.6.0 (already installed).

---

### 12. REFERENCE: Paper Hyperparameters (for comparison)

| Parameter | Paper (GSM8K) | ProofForge (Lean proofs) |
|-----------|---------------|--------------------------|
| Senior model | Llama-2-7b (GSM8K specialist) | DeepSeek-Prover-V2-7B (SFT+GRPO) |
| Junior model | Llama-2-7b-chat (frozen) | SFT checkpoint (frozen) |
| LoRA rank | 16 | 16 |
| LoRA alpha | 16 | 32 |
| LoRA targets | All linear | q,k,v,o,gate,up,down_proj |
| Quantization | 4-bit QLoRA | bfloat16 (or 4-bit QLoRA) |
| Batch size | 152 | 4 theorems × 8 group_size = 32 |
| Learning rate | 10⁻⁴ | 5×10⁻⁶ |
| Temperature | 0.7 | 0.7 (cosine schedule 1.0→0.5) |
| Max output length | 256 | 512 |
| Handoff granularity | Token (word) | Tactic (newline) |
| Rollouts per prompt | 2 | 8 (group_size) |
| RL algorithm | REINFORCE (binary) | REINFORCE (binary, group-normalized) |
| Training steps | 80 gradient updates | 200 steps |

---

## END OF PROMPT

**Instructions to AS mode:**
1. Read this prompt in full.
2. Read `scripts/grpo_lean_reward.py` in full to verify all line references.
3. Produce the architectural specification document at `docs/specs/tandem-grpo-architecture.md`.
4. The spec should be self-contained: a developer who has never seen the codebase should
   understand the full design from the spec alone.
5. Include all function signatures, algorithms, data flow diagrams (ASCII), and edge cases.
6. Flag any discrepancies between this prompt and the actual codebase state.
