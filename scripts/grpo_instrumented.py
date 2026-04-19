#!/usr/bin/env python3
"""
ProofForge -- Instrumented GRPO Training with Hidden State Collection

Runs GRPO training on DeepSeek-Prover-V2-7B with LoRA, collecting hidden states
at each checkpoint in the schedule. Designed for RunPod H100.

Usage:
  python3 grpo_instrumented.py --steps 200
  python3 grpo_instrumented.py --model deepseek-ai/DeepSeek-Prover-V2-7B --steps 200 --output-dir /workspace/grpo_run

Prerequisites (RunPod PyTorch 2.4 template):
  pip install transformers peft accelerate datasets numpy sentencepiece protobuf
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Lean tactic heuristics (no Lean binary needed on the pod)
# ---------------------------------------------------------------------------

VALID_TACTICS = {
    "rfl", "trivial", "decide", "simp", "omega", "ring", "norm_num",
    "exact", "intro", "apply", "constructor", "cases", "induction",
    "use", "exists", "left", "right", "assumption", "contradiction",
    "linarith", "nlinarith", "aesop", "tauto", "field_simp", "push_neg",
    "rcases", "obtain", "have", "suffices", "show", "change",
}

SORRY_PATTERNS = ["sorry", "admit", "native_decide"]


def heuristic_reward(proof_text: str, statement: str) -> float:
    """Binary heuristic reward: 1.0 if proof looks valid, 0.0 otherwise.

    Heuristic success criteria:
      1. Proof text is non-empty
      2. Does not contain 'sorry' or 'admit'
      3. Proof length < 200 tokens (very long = likely looping/broken)
      4. Contains at least one recognised Lean tactic keyword
      5. Does not restart the theorem statement inside the proof body
    """
    text = proof_text.strip()
    if not text:
        return 0.0

    # Reject sorry / admit
    for bad in SORRY_PATTERNS:
        if bad in text.lower():
            return 0.0

    # Reject suspiciously long outputs (likely degenerate)
    word_count = len(text.split())
    if word_count > 200:
        return 0.0

    # Must contain at least one known tactic
    text_lower = text.lower()
    has_tactic = any(t in text_lower for t in VALID_TACTICS)
    if not has_tactic:
        return 0.0

    # Reject if the model restarted the theorem statement
    stmt_keyword = "theorem "
    if stmt_keyword in text:
        return 0.0

    return 1.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_theorems(data_path: str) -> list:
    """Load theorems from the JSON file at data_path.

    Expected format:
      {
        "tiers": [
          { "theorems": [ {"id": "...", "statement": "...", "known_proof": "..."}, ... ] },
          ...
        ]
      }
    """
    with open(data_path) as f:
        raw = json.load(f)

    theorems = []
    for tier in raw.get("tiers", []):
        for thm in tier.get("theorems", []):
            theorems.append(thm)

    print(f"Loaded {len(theorems)} theorems from {data_path}")
    return theorems


def make_prompt(statement: str) -> str:
    """Build the prompt string for a theorem statement."""
    return (
        "Complete this Lean 4 proof. Output ONLY the tactic(s).\n\n"
        + statement + "\n"
    )


# ---------------------------------------------------------------------------
# Hidden state generation (manual autoregressive loop, matches collect.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    layer_idx: int = -1,
) -> dict:
    """Autoregressive generation with hidden state recording at each step.

    Returns:
      hidden_states : (T_steps, d_hidden)  float16 numpy array
      tokens        : (T_steps,)           int32 numpy array
      proof_text    : str
      generation_time : float
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    hidden_states_list = []
    generated_tokens = []
    t_start = time.time()

    for _step in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Record hidden state at the last token position, chosen layer
        layer_hidden = outputs.hidden_states[layer_idx]   # (1, seq_len, d_hidden)
        last_tok_hidden = layer_hidden[0, -1, :]           # (d_hidden,)
        hidden_states_list.append(
            last_tok_hidden.cpu().to(torch.float16).numpy()
        )

        # Sample next token
        logits = outputs.logits[0, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        tok_id = next_token.item()
        generated_tokens.append(tok_id)

        if tok_id == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype),
            ], dim=1)

    generation_time = time.time() - t_start

    if hidden_states_list:
        hidden_states = np.stack(hidden_states_list, axis=0)
    else:
        hidden_states = np.zeros((1, 1), dtype=np.float16)
    tokens = np.array(generated_tokens, dtype=np.int32)
    proof_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return {
        "hidden_states": hidden_states,
        "tokens": tokens,
        "proof_text": proof_text,
        "generation_time": generation_time,
    }


# ---------------------------------------------------------------------------
# Checkpoint collection phase
# ---------------------------------------------------------------------------

def collect_at_checkpoint(
    model,
    tokenizer,
    theorems,
    checkpoint_dir: Path,
    step: int,
    temperatures,
    max_new_tokens: int,
    d_hidden: int,
    n_layers: int,
    model_name: str,
) -> dict:
    """Run inference on all theorems x temperatures and save .npz files.

    Returns a summary dict with pass rate and per-theorem results.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Put model into eval mode for collection
    was_training = model.training
    model.eval()

    with torch.no_grad():
        total = 0
        passed = 0
        theorem_results = []

        print(
            f"\n  [Collect step={step}] {len(theorems)} theorems x "
            f"{len(temperatures)} temps = "
            f"{len(theorems) * len(temperatures)} trajectories"
        )

        for thm in theorems:
            thm_passed_any = False
            thm_results_by_temp = {}

            for temp in temperatures:
                total += 1
                prompt = make_prompt(thm["statement"])

                result = generate_with_hidden_states(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    layer_idx=-1,
                )

                reward = heuristic_reward(result["proof_text"], thm["statement"])
                success = reward > 0.5

                if success:
                    passed += 1
                    thm_passed_any = True

                # Save .npz
                traj_id = "{}_{:.1f}_step{:04d}".format(thm["id"], temp, step)
                save_path = checkpoint_dir / (traj_id + ".npz")
                np.savez_compressed(
                    save_path,
                    hidden_states=result["hidden_states"],
                    tokens=result["tokens"],
                    success=np.array(success),
                    temperature=np.array(temp),
                    generation_time=np.array(result["generation_time"]),
                    statement=np.array(thm["statement"]),
                    proof_text=np.array(result["proof_text"]),
                    theorem_id=np.array(thm["id"]),
                    step=np.array(step),
                    d_hidden=np.array(d_hidden),
                    n_layers=np.array(n_layers),
                    model_name=np.array(model_name),
                )

                thm_results_by_temp[str(temp)] = {
                    "success": success,
                    "proof_text": result["proof_text"][:200],
                    "generation_steps": int(result["hidden_states"].shape[0]),
                    "generation_time": round(result["generation_time"], 2),
                }

                status = "PASS" if success else "fail"
                print(
                    "    [{:4d}] {:40s} | {:3d} steps | {:.1f}s | {}".format(
                        total,
                        traj_id,
                        result["hidden_states"].shape[0],
                        result["generation_time"],
                        status,
                    )
                )

            theorem_results.append({
                "id": thm["id"],
                "statement": thm["statement"],
                "passed_any": thm_passed_any,
                "by_temp": thm_results_by_temp,
            })

    pass_rate = passed / total if total > 0 else 0.0
    print("  [Collect step={}] pass_rate={:.3f} ({}/{})".format(
        step, pass_rate, passed, total))

    summary = {
        "step": step,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "temperatures": list(temperatures),
        "theorem_results": theorem_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "max_new_tokens": max_new_tokens,
    }

    summary_path = checkpoint_dir / "checkpoint_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved: {}".format(summary_path))

    # Restore training mode if needed
    if was_training:
        model.train()

    return summary


# ---------------------------------------------------------------------------
# Manual GRPO generation helpers
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list:
    """Generate group_size completions for a single prompt.

    Returns a list of dicts with keys: token_ids (list[int]), text (str).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    completions = []
    for _ in range(group_size):
        current_ids = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.clone()

        generated = []

        for _step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tok_id = next_token.item()
            generated.append(tok_id)

            if tok_id == tokenizer.eos_token_id:
                break

            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype),
                ], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append({"token_ids": generated, "text": text})

    return completions


def compute_completion_log_prob(
    model,
    tokenizer,
    prompt: str,
    completion_token_ids: list,
) -> torch.Tensor:
    """Compute the sum of log probabilities for completion tokens under the model.

    Returns a scalar tensor (requires grad).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    completion_tensor = torch.tensor(
        [completion_token_ids], dtype=torch.long, device=model.device
    )
    # Full sequence: [prompt tokens] + [completion tokens]
    full_ids = torch.cat([prompt_ids, completion_tensor], dim=1)

    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)

    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Logits at positions [prompt_len-1 .. prompt_len+comp_len-2] predict
    # completion tokens at positions [prompt_len .. prompt_len+comp_len-1].
    pred_logits = logits[0, prompt_len - 1: prompt_len + comp_len - 1, :]  # (comp_len, vocab)
    targets = completion_tensor[0]  # (comp_len,)

    log_probs = F.log_softmax(pred_logits, dim=-1)  # (comp_len, vocab)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]
    return token_log_probs.sum()


# ---------------------------------------------------------------------------
# Manual GRPO training step
# ---------------------------------------------------------------------------

def grpo_step(
    model,
    tokenizer,
    batch_theorems: list,
    group_size: int,
    max_new_tokens: int,
    optimizer,
    scheduler,
    step_idx: int,
) -> dict:
    """Perform one GRPO gradient update over a batch of theorems.

    For each theorem:
      1. Generate group_size completions
      2. Score with heuristic reward
      3. Compute advantages: (r - mean_r) / (std_r + 1e-8)
      4. For completions with positive advantage, accumulate -advantage * log_prob
      5. Backprop and step

    Returns a metrics dict.
    """
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_updates = 0
    all_rewards = []
    all_advantages = []

    for thm in batch_theorems:
        prompt = make_prompt(thm["statement"])

        # --- 1. Generate group_size completions (no grad needed here) ---
        model.eval()
        completions = generate_completions(
            model, tokenizer, prompt,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
        model.train()

        # --- 2. Score ---
        rewards = [
            heuristic_reward(c["text"], thm["statement"])
            for c in completions
        ]
        all_rewards.extend(rewards)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item()

        # --- 3. Advantages ---
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        all_advantages.extend(advantages)

        # --- 4. Loss over positive-advantage completions ---
        for comp, adv in zip(completions, advantages):
            if adv <= 0 or len(comp["token_ids"]) == 0:
                continue

            log_prob = compute_completion_log_prob(
                model, tokenizer, prompt, comp["token_ids"]
            )
            loss_term = -adv * log_prob
            total_loss = total_loss + loss_term
            n_updates += 1

    if n_updates > 0:
        loss = total_loss / n_updates
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_val = loss.item()
    else:
        # All advantages were non-positive — no update, still step scheduler
        scheduler.step()
        loss_val = 0.0

    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    mean_adv = float(np.mean(all_advantages)) if all_advantages else 0.0

    return {
        "step": step_idx,
        "loss": loss_val,
        "mean_reward": mean_reward,
        "mean_advantage": mean_adv,
        "n_updates": n_updates,
    }


# ---------------------------------------------------------------------------
# GRPO training with custom checkpoint schedule
# ---------------------------------------------------------------------------

def run_grpo(
    model_name: str,
    theorems: list,
    total_steps: int,
    group_size: int,
    output_dir: Path,
    checkpoint_schedule: list,
    temperatures: list,
    max_new_tokens: int,
    batch_size: int = 4,
):
    """Main training + collection loop."""

    print("\n=== Loading tokenizer: {} ===".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== Loading model: {} ===".format(model_name))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    d_hidden = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print("Model: {} layers, d_hidden={}".format(n_layers, d_hidden))

    # Wrap with LoRA
    print("=== Applying LoRA ===")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Optimizer and cosine LR schedule
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-6,
    )
    # Warmup for first 5% of steps
    num_warmup = max(1, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
    )

    print("\n=== Manual GRPO Training Config ===")
    print("  Model:        {}".format(model_name))
    print("  Total steps:  {}".format(total_steps))
    print("  Group size:   {}".format(group_size))
    print("  Batch size:   {}".format(batch_size))
    print("  Checkpoints:  {}".format(checkpoint_schedule))
    print("  Theorems:     {}".format(len(theorems)))
    print("  Temperatures: {}".format(temperatures))
    print("  Output dir:   {}".format(output_dir))
    print()

    # Collect at step 0 (baseline, before any training)
    if 0 in checkpoint_schedule:
        ckpt_dir = output_dir / "checkpoint_000"
        print("\n=== Checkpoint step=0 (baseline) ===")
        collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=0, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
        )
        # Save model weights at step 0
        weights_path = ckpt_dir / "model_weights"
        print("  Saving weights to {}".format(weights_path))
        model.save_pretrained(str(weights_path))
        tokenizer.save_pretrained(str(weights_path))

    # Build sorted non-zero schedule points
    schedule_points = sorted(s for s in checkpoint_schedule if s > 0)
    all_summaries = []

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    current_step = 0
    theorem_pool = list(theorems)  # mutable pool for sampling
    step_metrics_log = []

    model.train()

    for target_step in schedule_points:
        if target_step > total_steps:
            break

        steps_this_segment = target_step - current_step
        if steps_this_segment <= 0:
            continue

        print("\n=== Training steps {} -> {} ({} steps) ===".format(
            current_step, target_step, steps_this_segment))

        for _ in range(steps_this_segment):
            current_step += 1

            # Sample a batch of theorems
            if len(theorem_pool) >= batch_size:
                batch = random.sample(theorem_pool, batch_size)
            else:
                # Fewer theorems than batch size — use all with repetition
                batch = [random.choice(theorem_pool) for _ in range(batch_size)]

            metrics = grpo_step(
                model=model,
                tokenizer=tokenizer,
                batch_theorems=batch,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                optimizer=optimizer,
                scheduler=scheduler,
                step_idx=current_step,
            )
            step_metrics_log.append(metrics)

            print(
                "  step {:4d} | loss={:.4f} | mean_reward={:.3f} | "
                "mean_adv={:.4f} | n_updates={}".format(
                    current_step,
                    metrics["loss"],
                    metrics["mean_reward"],
                    metrics["mean_advantage"],
                    metrics["n_updates"],
                )
            )

        # -----------------------------------------------------------------------
        # Checkpoint: collect hidden states
        # -----------------------------------------------------------------------
        print("\n=== Checkpoint step={} ===".format(target_step))
        ckpt_dir = output_dir / "checkpoint_{:03d}".format(target_step)

        # Save model weights
        weights_path = ckpt_dir / "model_weights"
        print("  Saving weights to {}".format(weights_path))
        model.save_pretrained(str(weights_path))
        tokenizer.save_pretrained(str(weights_path))

        # Collect hidden states with the current trained model
        summary = collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=target_step, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
        )
        all_summaries.append(summary)

        # Print running pass rate curve
        print("\n  === Pass Rate Curve So Far ===")
        for s in all_summaries:
            bar = "#" * int(s["pass_rate"] * 40)
            print("  step {:4d}: {:.3f} |{}".format(s["step"], s["pass_rate"], bar))

        model.train()

    # Save overall training summary
    run_summary = {
        "model_name": model_name,
        "total_steps": total_steps,
        "group_size": group_size,
        "batch_size": batch_size,
        "checkpoint_schedule": checkpoint_schedule,
        "temperatures": list(temperatures),
        "n_theorems": len(theorems),
        "max_new_tokens": max_new_tokens,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "checkpoints": all_summaries,
        "step_metrics": step_metrics_log,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run_summary_path = output_dir / "run_summary.json"
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print("\n=== Run complete. Summary saved to {} ===".format(run_summary_path))

    # Print final pass rate table
    print("\n" + "=" * 60)
    print("{:>6}  {:>10}  {:>8}  {:>6}".format("Step", "Pass Rate", "Passed", "Total"))
    print("-" * 60)
    for s in all_summaries:
        print("{:>6}  {:>10.3f}  {:>8}  {:>6}".format(
            s["step"], s["pass_rate"], s["passed"], s["total"]))
    print("=" * 60)

    return run_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProofForge Instrumented GRPO Training with Hidden State Collection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total GRPO training steps",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO group size G (number of completions per prompt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/grpo_run",
        help="Root output directory",
    )
    parser.add_argument(
        "--checkpoint-schedule",
        type=str,
        default="0,5,10,20,30,50,75,100,150,200",
        help="Comma-separated step numbers at which to collect hidden states",
    )
    parser.add_argument(
        "--theorems-path",
        type=str,
        default="/workspace/proofforge/data/theorems.json",
        help="Path to theorems.json dataset",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=50,
        help="Max theorems to use for collection (training uses all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per trajectory during collection",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.3, 0.7, 1.2],
        help="Sampling temperatures for collection (not training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProofForge -- Instrumented GRPO Training")
    print("=" * 60)
    print("  Model:       {}".format(args.model))
    print("  Steps:       {}".format(args.steps))
    print("  Group size:  {}".format(args.group_size))
    print("  Output dir:  {}".format(args.output_dir))
    print("  Checkpoints: {}".format(args.checkpoint_schedule))
    print("  Theorems:    {}".format(args.theorems_path))
    print("  Temps:       {}".format(args.temperatures))
    print("  Max tokens:  {}".format(args.max_new_tokens))
    print()

    # Parse checkpoint schedule
    checkpoint_schedule = [
        int(x.strip())
        for x in args.checkpoint_schedule.split(",")
        if x.strip()
    ]
    # Ensure total_steps is in the schedule
    if args.steps not in checkpoint_schedule:
        checkpoint_schedule.append(args.steps)
    checkpoint_schedule = sorted(set(checkpoint_schedule))
    print("  Final schedule: {}".format(checkpoint_schedule))
    print()

    # Validate theorems path
    if not os.path.exists(args.theorems_path):
        print("ERROR: theorems.json not found at {}".format(args.theorems_path))
        print("Expected location: /workspace/proofforge/data/theorems.json")
        sys.exit(1)

    # Load theorems
    all_theorems = load_theorems(args.theorems_path)
    collection_theorems = all_theorems[: args.max_theorems]
    print("Using {} theorems for collection (dataset has {} total)".format(
        len(collection_theorems), len(all_theorems)))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Report GPU state
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print("  GPU {}: {} ({:.1f} GB)".format(
                i, torch.cuda.get_device_name(i), gb))
    else:
        print("WARNING: No CUDA detected. Running on CPU will be very slow.")
    print()

    # Set HuggingFace cache to persistent volume if available
    if os.path.exists("/workspace"):
        os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
        Path("/workspace/hf_cache").mkdir(parents=True, exist_ok=True)
        print("HF_HOME set to /workspace/hf_cache")

    run_grpo(
        model_name=args.model,
        theorems=collection_theorems,
        total_steps=args.steps,
        group_size=args.group_size,
        output_dir=output_dir,
        checkpoint_schedule=checkpoint_schedule,
        temperatures=args.temperatures,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
