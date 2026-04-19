# ProofForge Round 3 Training Spec: Two-Stage Pipeline

## Overview

Round 3 introduces a two-stage pipeline to break through the ~2% base-rate ceiling observed in
Rounds 1 and 2. The core insight from recent RL-for-theorem-proving literature is that pure
GRPO from a cold start wastes most compute on a reward signal that is nearly always zero. The
solution is a cold-start warm-up via Rejection Sampling SFT (RS-SFT) before GRPO, paired
with an unlikeliness penalty to discourage the model from mode-collapsing onto its most
likely (often trivially-wrong) completions.

---

## Stage 1: Rejection Sampling SFT (Cold Start)

### Goal

Boost the empirical pass rate from ~2% to a target of 10-20% so that GRPO in Stage 2 has
a rich non-zero reward signal to work with.

### Protocol

1. **Sampling.** For each of 50 training theorems, generate 200 candidate proofs at
   temperature T=1.2 (high entropy to maximise diversity).
   - Total completions: 50 x 200 = 10,000
   - Generation uses the base model (or the most recent GRPO checkpoint if running
     Stage 2 immediately after a prior run)

2. **Verification.** Verify all 10,000 completions with the Lean 4 type-checker.
   - Parallelism: 8 CPU workers via `ProcessPoolExecutor`
   - Timeout: 30 s per proof
   - Only proofs where Lean returns exit code 0 are retained

3. **Filtering.** Keep only Lean-verified successes.
   - If zero successes across all theorems, emit a warning and exit Stage 1 early
     (Stage 2 will then proceed from the base model)

4. **SFT Training.** Fine-tune the LoRA model on the successful (statement, proof) pairs.
   - Loss: standard next-token cross-entropy on proof tokens only (prompt tokens masked)
   - Optimizer: AdamW, lr=2e-5
   - Epochs: 2
   - Batch size: 4 per step
   - LR schedule: cosine warmup over first 5% of steps

5. **Checkpoint.** Save the SFT adapter weights to `{output_dir}/sft_checkpoint/`.

### Hyperparameters

| Parameter              | Value   |
|------------------------|---------|
| Theorems               | 50      |
| Completions / theorem  | 200     |
| Sampling temperature   | T = 1.2 |
| Lean workers           | 8       |
| SFT learning rate      | 2e-5    |
| SFT epochs             | 2       |
| LoRA rank              | 16      |
| LoRA alpha             | 32      |

---

## Stage 2: GRPO with Unlikeliness Reward

### Motivation

Standard GRPO computes advantages as `(r_i - mean_r) / std_r`.  When all completions in a
group are wrong the gradient is near zero.  When the model passes 10-20% of proofs
(after Stage 1) there is signal, but the model still tends to repeat its highest-probability
completions.  The "Rewarding the Unlikely" approach re-weights rewards so that rare,
lower-probability successes get amplified while common high-probability successes are
discounted.  This prevents mode collapse and encourages exploration.

### Unlikeliness Reward Formula

For a group of G completions of the same prompt, sorted so that rank 0 is the
*highest*-probability completion under the current policy:

```
r_i_unlike = r_i * (1 - beta_rank * (G - rank_i) / G)
```

Where:
- `r_i` is the binary Lean verification reward (0.0 or 1.0)
- `rank_i` is the rank of completion i by log-probability under the current policy
  (rank 0 = most probable, rank G-1 = least probable)
- `beta_rank` = 0.25 (default)
- `G` = group size

Effect:
- The most likely completion (rank 0) is penalized by factor `(1 - 0.25 * G/G)` = 0.75
- The least likely completion (rank G-1) retains full reward (1 - 0.25 * 0/G) = 1.0
- Mid-rank completions receive intermediate scaling

This encourages the model to diversify away from its mode while still rewarding correctness.

### Dynamic Sampling (Buffer Mechanism)

A key efficiency improvement: only perform gradient updates when there is a non-trivial
advantage signal.  A group with all rewards zero (or all rewards one) produces zero
advantage and a zero gradient -- updating on it wastes compute.

The buffer mechanism:
1. Generate a group of G completions and verify them with Lean
2. Compute unlikeliness-adjusted rewards and then advantages
3. If `max(|advantages|) > epsilon` (default epsilon=1e-4), add the group to the buffer
4. Only perform a gradient update when the buffer reaches `target_batch_size` groups
   (default: 4 theorem groups per update, i.e., the standard batch size)

This ensures every gradient update contains useful signal.

### Training Configuration

| Parameter              | Value                                     |
|------------------------|-------------------------------------------|
| Group size G           | 64 (fall back to 32 if OOM)               |
| Learning rate          | 1e-6                                      |
| LR schedule            | Cosine with 5% warmup                     |
| beta_rank              | 0.25                                      |
| Max new tokens         | 256                                       |
| Lean workers           | 8                                         |
| Training steps         | 200                                       |
| Batch size             | 4 theorems per update                     |
| Gradient clip          | 1.0                                       |

### Checkpoint Schedule

Hidden states are collected at steps: 0, 5, 10, 20, 30, 50, 75, 100, 150, 200

At each checkpoint:
- Model adapter weights saved to `{output_dir}/checkpoint_{step:03d}/model_weights/`
- Hidden states and metadata saved as `.npz` files (one per theorem x temperature)
- Summary JSON saved to `{output_dir}/checkpoint_{step:03d}/checkpoint_summary.json`
- Collection temperatures: [0.3, 0.7, 1.2]

### Hidden State Collection

Identical to Rounds 1 and 2:
- Manual token-by-token autoregressive loop with `output_hidden_states=True`
- Records the hidden state at the final-layer, last-token position at each decode step
- Output shape: `(T_steps, d_hidden)` float16 numpy array
- Saves `hidden_states`, `tokens`, `success`, `temperature`, `generation_time`,
  `statement`, `proof_text`, `theorem_id`, `step`, `d_hidden`, `n_layers`, `model_name`

---

## CLI Reference (`grpo_round3.py`)

```
python3 grpo_round3.py --stage both --model deepseek-ai/DeepSeek-Prover-V2-7B
python3 grpo_round3.py --stage 1 --n-completions-per-theorem 200 --sft-epochs 2
python3 grpo_round3.py --stage 2 --skip-sft  # GRPO from base model
python3 grpo_round3.py --stage 2 --group-size 32  # reduced group size for OOM
```

| Flag                         | Default                              | Description                                 |
|------------------------------|--------------------------------------|---------------------------------------------|
| `--model`                    | deepseek-ai/DeepSeek-Prover-V2-7B   | HuggingFace model name or local path        |
| `--stage`                    | both                                 | 1, 2, or "both"                             |
| `--group-size`               | 64                                   | GRPO group size G                           |
| `--lr`                       | 1e-6                                 | GRPO learning rate                          |
| `--beta-rank`                | 0.25                                 | Unlikeliness penalty coefficient            |
| `--sft-epochs`               | 2                                    | SFT epochs in Stage 1                       |
| `--sft-lr`                   | 2e-5                                 | SFT learning rate in Stage 1                |
| `--n-completions-per-theorem`| 200                                  | Completions generated per theorem in Stage 1|
| `--output-dir`               | /workspace/grpo_round3               | Root output directory                       |
| `--checkpoint-schedule`      | 0,5,10,20,30,50,75,100,150,200       | Steps at which to collect hidden states     |
| `--lean-workers`             | 8                                    | Parallel Lean worker processes              |
| `--max-new-tokens`           | 256                                  | Max tokens per generation                   |
| `--skip-sft`                 | (flag, default off)                  | Skip Stage 1, load base model for Stage 2   |
| `--theorems-path`            | /workspace/proofforge/data/theorems.json | Path to theorems dataset               |
| `--max-theorems`             | 50                                   | Max theorems to use                         |
| `--temperatures`             | 0.3 0.7 1.2                          | Collection temperatures for hidden states   |

---

## Expected Outcomes

| Stage        | Metric                     | Target       |
|--------------|----------------------------|--------------|
| Baseline     | Pass rate                  | ~2%          |
| After Stage 1| Pass rate                  | 10-20%       |
| After Stage 2| Pass rate at step 200      | 25-40%       |
| Exploration  | Reward variance in groups  | Higher than Round 2 (unlikeliness effect) |

---

## Literature Basis

- **Rejection Sampling SFT / Cold Start**: Commonly used before GRPO in recent work
  (e.g., DeepSeek-R1, STILL-2, Sky-T1) to bootstrap a non-trivial base rate.
- **Unlikeliness Reward**: Inspired by "Rewarding the Unlikely" (reward re-weighting by
  rank under policy to suppress mode collapse in RL fine-tuning).
- **Dynamic Sampling Buffer**: Standard trick to avoid zero-gradient steps; first
  described formally in GRPO++ and related extensions.
