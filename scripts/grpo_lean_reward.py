#!/usr/bin/env python3
"""
ProofForge -- Instrumented GRPO Training with Lean 4 Verification Reward

Runs GRPO training on DeepSeek-Prover-V2-7B with LoRA, collecting hidden states
at each checkpoint in the schedule. Designed for RunPod H100.

Uses real Lean 4 type-checker for rewards instead of heuristics, with parallel
verification across CPU cores via ProcessPoolExecutor.

Usage:
  python3 grpo_lean_reward.py --steps 200
  python3 grpo_lean_reward.py --model deepseek-ai/DeepSeek-Prover-V2-7B --steps 200 --output-dir /workspace/grpo_run
  python3 grpo_lean_reward.py --steps 200 --lean-workers 16

Prerequisites (RunPod PyTorch 2.4 template):
  pip install transformers peft accelerate datasets numpy sentencepiece protobuf
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# --- App 6 + App 8 integration (Round 3c) ---
# Ensure proofforge is importable: pip install -e python/
try:
    from grpo_round3_integration import (
        create_efficiency_oracle,
        create_controller,
        compute_efficiency_rewards,
        controller_step,
        RetentionTracker,
        count_distinct_tactics,
        extract_first_tactic,
    )
    APPS_AVAILABLE = True
except ImportError:
    APPS_AVAILABLE = False
    print("WARNING: grpo_round3_integration not importable. Running without Apps 6/8.")


# ---------------------------------------------------------------------------
# Lean 4 setup
# ---------------------------------------------------------------------------

def setup_lean():
    """Install Lean 4 (elan) if not available."""
    if shutil.which("lean"):
        result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
        print(f"  Lean already installed: {result.stdout.strip()}")
        return True
    print("  Installing elan (Lean version manager)...")
    subprocess.run(
        ["bash", "-c",
         "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
         "| sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0"],
        check=True,
    )
    os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"]
    result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
    print(f"  Lean installed: {result.stdout.strip()}")
    return True


# ---------------------------------------------------------------------------
# Custom definitions preamble for Level 4+ theorems
# ---------------------------------------------------------------------------

_LEAN_DEFS_CACHE: str | None = None

def _get_lean_defs() -> str:
    """Load custom Lean definitions (myDouble, myPow2, etc.) from env var or file."""
    global _LEAN_DEFS_CACHE
    if _LEAN_DEFS_CACHE is not None:
        return _LEAN_DEFS_CACHE
    defs_path = os.environ.get("PROOFFORGE_LEAN_DEFS", "")
    if defs_path and os.path.exists(defs_path):
        with open(defs_path) as f:
            _LEAN_DEFS_CACHE = f.read()
        print(f"  Loaded custom Lean definitions from {defs_path}")
    else:
        _LEAN_DEFS_CACHE = ""
    return _LEAN_DEFS_CACHE


# ---------------------------------------------------------------------------
# Lean 4 reward: single-proof verification and batch parallel verification
# ---------------------------------------------------------------------------

def lean_verify_single(statement: str, proof_text: str, timeout: int = 30,
                        needs_definitions: bool = False) -> bool:
    """Verify a single proof with Lean 4. Returns True if type-checks."""
    cleaned = proof_text.replace('\u010a', '\n').replace('\u0120', ' ').strip()
    if not cleaned:
        return False
    # Send FULL proof text — multi-tactic proofs verified as-is
    proof_lines = cleaned.split('\n')
    indented = '\n'.join(f'  {line}' for line in proof_lines)
    preamble = ""
    if needs_definitions:
        defs = _get_lean_defs()
        if defs:
            preamble = defs + "\n\n"
    source = f"{preamble}{statement}\n{indented}\n"
    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(source)
        tmpfile = f.name
    try:
        result = subprocess.run(["lean", tmpfile], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(tmpfile)


def lean_verify_batch(
    statements: list,
    proofs: list,
    max_workers: int = 8,
    timeout: int = 30,
    needs_definitions: bool | list[bool] = False,
) -> list:
    """Verify a batch of proofs in parallel across CPU cores."""
    if isinstance(needs_definitions, bool):
        defs_list = [needs_definitions] * len(statements)
    else:
        defs_list = needs_definitions
    results = [False] * len(statements)
    # Use threads — Lean verification is subprocess I/O, avoids CUDA fork crash
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(lean_verify_single, stmt, proof, timeout, nd): i
            for i, (stmt, proof, nd) in enumerate(zip(statements, proofs, defs_list))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    return results


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

        # With KV cache, only feed the new token
        current_ids = next_token.unsqueeze(0)
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
# Tandem generation (West et al. 2026 -- tactic-level interleaving)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_tandem_completion(
    model,
    tokenizer,
    prompt: str,
    senior_adapter: str = "default",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    current_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    if newline_token_id is None:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    past_key_values = None
    all_token_ids = []
    senior_mask = []
    tactic_assignments = []
    n_handoffs = 0

    # First tactic assignment (coin flip)
    current_gen = "senior" if random.random() > handoff_prob else "junior"
    model.set_adapter(senior_adapter if current_gen == "senior" else junior_adapter)
    tactic_assignments.append(current_gen)

    for _step in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[0, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        tok_id = next_token.item()

        all_token_ids.append(tok_id)
        senior_mask.append(current_gen == "senior")

        if tok_id == tokenizer.eos_token_id:
            break

        # Check tactic boundary (newline)
        if tok_id == newline_token_id:
            new_gen = "senior" if random.random() > handoff_prob else "junior"
            if new_gen != current_gen:
                n_handoffs += 1
                model.set_adapter(
                    senior_adapter if new_gen == "senior" else junior_adapter
                )
            current_gen = new_gen
            tactic_assignments.append(current_gen)

        # Prepare next input (KV cache: only feed new token)
        current_ids = next_token.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype),
            ], dim=1)

    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)

    # Restore senior adapter (callers expect senior active)
    model.set_adapter(senior_adapter)

    return {
        "token_ids": all_token_ids,
        "text": text,
        "senior_mask": senior_mask,
        "tactic_assignments": tactic_assignments,
        "n_handoffs": n_handoffs,
    }


def generate_tandem_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    senior_adapter: str = "default",
    junior_adapter: str = "junior",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    handoff_prob: float = 0.5,
) -> list:
    """Generate group_size tandem completions for a single prompt.

    Calls generate_tandem_completion() sequentially. GPU batching is not
    possible because different sequences hit tactic boundaries at different
    token positions, requiring per-sequence adapter switching.

    Returns: list of dicts (same schema as generate_tandem_completion output).
    """
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    completions = []
    for _i in range(group_size):
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

    return completions


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
    lean_workers: int = 8,
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

                verified = lean_verify_single(
                    thm["statement"], result["proof_text"],
                    needs_definitions=thm.get("_requires_definitions", False),
                )
                success = verified

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

    # Free GPU memory between collection and training resumption
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    Uses batched generation via model.generate() for GPU parallelism.
    All group_size sequences are generated simultaneously, amortizing
    model weight reads across the batch.

    Returns a list of dicts with keys: token_ids (list[int]), text (str).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for i in range(group_size):
        # Extract generated tokens (after the prompt)
        gen_ids = output_ids[i, prompt_len:].tolist()
        # Trim at EOS and padding
        if tokenizer.eos_token_id in gen_ids:
            eos_idx = gen_ids.index(tokenizer.eos_token_id)
            gen_ids = gen_ids[:eos_idx]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append({"token_ids": gen_ids, "text": text})

    return completions


def rejection_sampling_sft(
    model,
    tokenizer,
    theorems: list,
    output_dir: Path,
    n_completions_per_theorem: int = 200,
    sft_epochs: int = 2,
    sft_lr: float = 2e-5,
    lean_workers: int = 8,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> Path:
    """Stage 1: Rejection Sampling SFT -- cold-start bootstrap.

    Generates many completions per theorem at high temperature, verifies
    ALL with Lean, then SFT-trains on verified proofs to boost the base
    rate from ~1%% to ~10-20%%. Solves the cold-start problem where GRPO
    gets zero gradient signal at very low pass rates.

    Returns the path to the saved SFT checkpoint directory.
    """
    print("\n" + "=" * 60)
    print("  Stage 1: Rejection Sampling SFT (Cold Start)")
    print("=" * 60)

    sft_ckpt_dir = output_dir / "sft_checkpoint"
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Generate completions at high temperature for diversity
    # -------------------------------------------------------------------
    total_completions = n_completions_per_theorem * len(theorems)
    print("\n[Stage 1] Generating {} completions x {} theorems = {} total at T=1.2...".format(
        n_completions_per_theorem, len(theorems), total_completions))

    all_statements = []
    all_proofs = []
    all_needs_defs = []

    model.eval()
    t0 = time.time()

    for thm_idx, thm in enumerate(theorems):
        prompt = make_prompt(thm["statement"])
        needs_defs = thm.get("_requires_definitions", False)

        # Generate in sub-batches to avoid OOM
        sub_batch = min(32, n_completions_per_theorem)
        thm_completions = []
        remaining = n_completions_per_theorem

        while remaining > 0:
            n = min(sub_batch, remaining)
            comps = generate_completions(
                model, tokenizer, prompt,
                group_size=n,
                max_new_tokens=max_new_tokens,
                temperature=1.2,
            )
            thm_completions.extend(comps)
            remaining -= n

        for comp in thm_completions:
            all_statements.append(thm["statement"])
            all_proofs.append(comp["text"])
            all_needs_defs.append(needs_defs)

        if (thm_idx + 1) % 10 == 0 or thm_idx == 0:
            print("  Theorem {}/{}: {} -- {} completions ({:.0f}s elapsed)".format(
                thm_idx + 1, len(theorems), thm["id"],
                len(thm_completions), time.time() - t0))

    print("\n[Stage 1] Generation complete: {} completions in {:.1f}s".format(
        len(all_proofs), time.time() - t0))

    # -------------------------------------------------------------------
    # 2. Verify all completions in parallel with Lean
    # -------------------------------------------------------------------
    print("\n[Stage 1] Verifying {} completions with Lean ({} workers)...".format(
        len(all_proofs), lean_workers))
    t_verify = time.time()

    verification_results = lean_verify_batch(
        all_statements, all_proofs,
        max_workers=lean_workers,
        timeout=30,
        needs_definitions=all_needs_defs,
    )

    n_success = sum(verification_results)
    print("[Stage 1] Verification complete in {:.1f}s: {}/{} passed ({:.1f}%)".format(
        time.time() - t_verify, n_success, len(verification_results),
        100.0 * n_success / max(1, len(verification_results))))

    # -------------------------------------------------------------------
    # 3. Filter to successful proofs + deduplicate
    # -------------------------------------------------------------------
    successful_pairs = [
        (stmt, proof)
        for stmt, proof, ok in zip(all_statements, all_proofs, verification_results)
        if ok
    ]

    seen = set()
    deduped = []
    for stmt, proof in successful_pairs:
        key = (stmt, proof.strip())
        if key not in seen:
            seen.add(key)
            deduped.append((stmt, proof))
    n_before = len(successful_pairs)
    successful_pairs = deduped
    print("[Stage 1] Deduplicated: {} -> {} unique proofs".format(n_before, len(successful_pairs)))

    from collections import Counter
    thm_counts = Counter()
    for stmt, _ in successful_pairs:
        thm_counts[stmt] += 1
    n_thms_with_proof = len(thm_counts)
    print("[Stage 1] Theorems with at least 1 proof: {}/{}".format(
        n_thms_with_proof, len(theorems)))

    sampling_summary = {
        "n_theorems": len(theorems),
        "n_completions_per_theorem": n_completions_per_theorem,
        "total_completions": len(all_proofs),
        "n_successes": n_success,
        "n_unique": len(successful_pairs),
        "n_theorems_with_proof": n_thms_with_proof,
        "pass_rate": n_success / max(1, len(all_proofs)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(sft_ckpt_dir / "sampling_summary.json", "w") as f:
        json.dump(sampling_summary, f, indent=2)

    if not successful_pairs:
        print("\n[Stage 1] WARNING: Zero successful proofs found. "
              "Skipping SFT -- Stage 2 will use the base model.")
        return sft_ckpt_dir

    # -------------------------------------------------------------------
    # 4. SFT training on successful proofs
    # -------------------------------------------------------------------
    print("\n[Stage 1] Training SFT on {} successful proofs for {} epochs, lr={}...".format(
        len(successful_pairs), sft_epochs, sft_lr))

    model.train()
    sft_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_lr,
    )

    global_sft_step = 0
    sft_loss_log = []

    for epoch in range(sft_epochs):
        random.shuffle(successful_pairs)
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_start in range(0, len(successful_pairs), batch_size):
            batch = successful_pairs[batch_start: batch_start + batch_size]
            if not batch:
                continue

            sft_optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            n_tokens = 0

            for stmt, proof_text in batch:
                prompt = make_prompt(stmt)
                prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
                proof_ids_list = tokenizer(proof_text, add_special_tokens=False)["input_ids"]

                if not proof_ids_list:
                    continue

                proof_tensor = torch.tensor(
                    [proof_ids_list], dtype=torch.long, device=model.device
                )
                full_ids = torch.cat([prompt_ids, proof_tensor], dim=1)

                outputs = model(input_ids=full_ids, use_cache=False)
                logits = outputs.logits

                prompt_len = prompt_ids.shape[1]
                comp_len = len(proof_ids_list)

                pred_logits = logits[0, prompt_len - 1: prompt_len + comp_len - 1, :]
                targets = proof_tensor[0]

                loss_per_token = F.cross_entropy(pred_logits, targets, reduction="sum")
                batch_loss = batch_loss + loss_per_token
                n_tokens += comp_len

            if n_tokens > 0:
                loss = batch_loss / n_tokens
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                sft_optimizer.step()

                loss_val = loss.item()
                epoch_loss_sum += loss_val
                epoch_steps += 1
                global_sft_step += 1
                sft_loss_log.append({"step": global_sft_step, "loss": loss_val})

                if global_sft_step % 10 == 0 or global_sft_step == 1:
                    print("  [SFT] epoch={} step={} loss={:.4f}".format(
                        epoch + 1, global_sft_step, loss_val))

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        print("\n[Stage 1] Epoch {}/{} complete -- avg loss={:.4f} over {} steps".format(
            epoch + 1, sft_epochs, avg_epoch_loss, epoch_steps))

    # -------------------------------------------------------------------
    # 5. Save SFT checkpoint
    # -------------------------------------------------------------------
    weights_path = sft_ckpt_dir / "model_weights"
    print("\n[Stage 1] Saving SFT weights to {}".format(weights_path))
    model.save_pretrained(str(weights_path))
    tokenizer.save_pretrained(str(weights_path))

    sft_summary = {
        "n_unique_proofs": len(successful_pairs),
        "n_theorems_with_proof": n_thms_with_proof,
        "sft_epochs": sft_epochs,
        "sft_lr": sft_lr,
        "total_steps": global_sft_step,
        "loss_log": sft_loss_log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(sft_ckpt_dir / "sft_training_summary.json", "w") as f:
        json.dump(sft_summary, f, indent=2)

    print("[Stage 1] Done. SFT checkpoint saved to {}".format(sft_ckpt_dir))

    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[Stage 1] GPU cache cleared for Stage 2")

    return sft_ckpt_dir


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


def compute_tandem_log_prob(
    model,
    tokenizer,
    prompt: str,
    completion_token_ids: list,
    senior_mask: list,
    senior_adapter: str = "default",
) -> torch.Tensor:
    """Compute sum of log probabilities for SENIOR-AUTHORED tokens only.

    Runs a single forward pass over the full sequence (prompt + all completion
    tokens), then masks the log_prob sum to include only positions where
    senior_mask[i] is True.

    The model uses the senior adapter for the forward pass (set explicitly).
    The returned tensor requires grad for backpropagation.

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
    model.set_adapter(senior_adapter)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    completion_tensor = torch.tensor(
        [completion_token_ids], dtype=torch.long, device=model.device
    )
    full_ids = torch.cat([prompt_ids, completion_tensor], dim=1)

    n_senior = sum(senior_mask)
    if n_senior == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)

    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Logits at [prompt_len-1 .. prompt_len+comp_len-2] predict tokens at
    # [prompt_len .. prompt_len+comp_len-1]
    pred_logits = logits[0, prompt_len - 1: prompt_len + comp_len - 1, :]  # (comp_len, vocab)
    targets = completion_tensor[0]  # (comp_len,)

    log_probs = F.log_softmax(pred_logits, dim=-1)  # (comp_len, vocab)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]

    # Apply senior mask: only sum log_probs where senior generated the token
    mask_tensor = torch.tensor(senior_mask, dtype=torch.bool, device=model.device)
    masked_log_probs = token_log_probs[mask_tensor]

    return masked_log_probs.sum()


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
    lean_workers: int = 8,
    eff_oracle=None,
    running_pass_rate: float = 0.0,
    training_temperature: float = 0.7,
) -> dict:
    """Perform one GRPO gradient update over a batch of theorems.

    For each theorem:
      1. Generate group_size completions
      2. Verify all completions in parallel with lean_verify_batch
      3. rewards = [1.0 if verified else 0.0 for verified in verification_results]
      4. Compute advantages: (r - mean_r) / (std_r + 1e-8)
      5. For completions with positive advantage, accumulate -advantage * log_prob
      6. Backprop and step

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
            temperature=training_temperature,
        )
        model.train()

        # --- 2. Verify all completions in parallel with Lean 4 ---
        statements = [thm["statement"]] * len(completions)
        proofs = [c["text"] for c in completions]
        needs_defs = thm.get("_requires_definitions", False)
        verification_results = lean_verify_batch(
            statements, proofs,
            max_workers=lean_workers,
            timeout=30,
            needs_definitions=needs_defs,
        )
        # --- 2b. Compute rewards (binary or efficiency-weighted) ---
        raw_rewards = [1.0 if verified else 0.0 for verified in verification_results]

        if eff_oracle is not None:
            rewards = compute_efficiency_rewards(
                oracle=eff_oracle,
                theorem=thm,
                completions=completions,
                verification_results=verification_results,
                raw_rewards=raw_rewards,
                step=step_idx,
                pass_rate=running_pass_rate,
                temperature=0.7,
            )
        else:
            rewards = raw_rewards

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
        # All advantages were non-positive -- no update, still step scheduler
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
        "efficiency_active": eff_oracle is not None and eff_oracle.current_step >= eff_oracle.config.phase_in_step,
        "pass_rate_estimate": running_pass_rate,
        "temperature": training_temperature,
    }


# ---------------------------------------------------------------------------
# Tandem GRPO training step (West et al. 2026)
# ---------------------------------------------------------------------------

def tandem_grpo_step(
    model,
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
    senior_adapter: str = "default",
    junior_adapter: str = "junior",
) -> dict:
    """Perform one tandem GRPO gradient update over a batch of theorems.

    Identical to grpo_step() except:
    1. Uses generate_tandem_completions() for generation
    2. Uses compute_tandem_log_prob() with senior_mask for loss
    3. Reports tandem-specific metrics
    """
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_updates = 0
    all_rewards = []
    all_advantages = []
    total_senior_frac = 0.0
    total_handoffs = 0
    n_completions_total = 0

    for thm in batch_theorems:
        prompt = make_prompt(thm["statement"])

        # --- 1. Generate tandem completions ---
        model.set_adapter(senior_adapter)
        model.eval()
        completions = generate_tandem_completions(
            model, tokenizer, prompt,
            group_size=group_size,
            senior_adapter=senior_adapter,
            junior_adapter=junior_adapter,
            max_new_tokens=max_new_tokens,
            temperature=training_temperature,
            handoff_prob=handoff_prob,
        )
        model.train()

        # --- Tandem metrics ---
        for comp in completions:
            if len(comp["senior_mask"]) > 0:
                total_senior_frac += sum(comp["senior_mask"]) / len(comp["senior_mask"])
            total_handoffs += comp["n_handoffs"]
            n_completions_total += 1

        # --- 2. Verify all completions in parallel with Lean 4 ---
        statements = [thm["statement"]] * len(completions)
        proofs = [c["text"] for c in completions]
        needs_defs = thm.get("_requires_definitions", False)
        verification_results = lean_verify_batch(
            statements, proofs,
            max_workers=lean_workers,
            timeout=30,
            needs_definitions=needs_defs,
        )

        # --- 2b. Compute rewards ---
        raw_rewards = [1.0 if verified else 0.0 for verified in verification_results]

        if eff_oracle is not None:
            rewards = compute_efficiency_rewards(
                oracle=eff_oracle,
                theorem=thm,
                completions=completions,
                verification_results=verification_results,
                raw_rewards=raw_rewards,
                step=step_idx,
                pass_rate=running_pass_rate,
                temperature=0.7,
            )
        else:
            rewards = raw_rewards

        all_rewards.extend(rewards)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item()

        # --- 3. Advantages ---
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        all_advantages.extend(advantages)

        # --- 4. Loss with tandem masking ---
        for comp, adv in zip(completions, advantages):
            if adv <= 0 or len(comp["token_ids"]) == 0:
                continue
            if not any(comp["senior_mask"]):
                continue  # skip all-junior completions

            log_prob = compute_tandem_log_prob(
                model, tokenizer, prompt,
                comp["token_ids"],
                comp["senior_mask"],
                senior_adapter=senior_adapter,
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
        scheduler.step()
        loss_val = 0.0

    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    mean_adv = float(np.mean(all_advantages)) if all_advantages else 0.0
    avg_senior_frac = total_senior_frac / max(n_completions_total, 1)
    avg_handoffs = total_handoffs / max(n_completions_total, 1)

    return {
        "step": step_idx,
        "loss": loss_val,
        "mean_reward": mean_reward,
        "mean_advantage": mean_adv,
        "n_updates": n_updates,
        "efficiency_active": eff_oracle is not None and eff_oracle.current_step >= eff_oracle.config.phase_in_step,
        "pass_rate_estimate": running_pass_rate,
        "temperature": training_temperature,
        "tandem_senior_frac": avg_senior_frac,
        "tandem_handoffs_per_completion": avg_handoffs,
        "tandem_completions_total": n_completions_total,
    }


# ---------------------------------------------------------------------------
# GRPO training with custom checkpoint schedule
# ---------------------------------------------------------------------------

def _make_cosine_with_floor(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with a floor at lr * min_lr_ratio.

    Adapted from PufferLib torch_pufferl.py:261-266:
        lr_min + 0.5*(lr - lr_min) * (1 + cos(pi * step/total))
    Unlike HuggingFace get_cosine_schedule_with_warmup which decays to 0,
    this floors at min_lr_ratio * base_lr to preserve late-training gradient signal.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


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
    lean_workers: int = 8,
    enable_efficiency: bool = False,
    enable_controller: bool = False,
    efficiency_phase_in_step: int = 50,
    efficiency_phase_in_rate: float = 0.20,
    controller_bias_correction: float = 0.0,
    min_lr_ratio: float = 0.1,
    optimizer_type: str = "adamw",
    muon_momentum: float = 0.95,
    learning_rate: float = 5e-6,
    temp_start: float = 1.0,
    temp_end: float = 0.5,
    load_in_4bit: bool = False,
    compile_model: bool = False,
    # --- Tandem training (West et al. 2026) ---
    enable_tandem: bool = True,
    tandem_junior_path: str | None = None,
    tandem_handoff_prob: float = 0.5,
    tandem_phase_in_step: int = 0,
):
    """Main training + collection loop."""

    # Check if model_name points to a LoRA/PEFT checkpoint (from Stage 1)
    _peft_checkpoint = Path(model_name) / "adapter_config.json"
    _is_peft = _peft_checkpoint.exists()

    if _is_peft:
        with open(_peft_checkpoint) as _f:
            adapter_cfg = json.load(_f)
        _base_model_name = adapter_cfg.get("base_model_name_or_path", model_name)
        print("\n=== Loading from PEFT checkpoint ===")
        print("  Base model: {}".format(_base_model_name))
        print("  LoRA adapter: {}".format(model_name))
        _tokenizer_source = model_name
    else:
        _base_model_name = model_name
        _tokenizer_source = model_name

    print("\n=== Loading tokenizer: {} ===".format(_tokenizer_source))
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== Loading model: {} ===".format(_base_model_name))
    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
    )
    # Try Flash Attention 2 if available, fall back to default (SDPA)
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Flash Attention 2: enabled")
    except ImportError:
        print("  Flash Attention 2: not installed, using default SDPA")
    if load_in_4bit and BNB_AVAILABLE:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("  Loading in 4-bit (QLoRA mode)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(_base_model_name, **load_kwargs)

    d_hidden = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print("Model: {} layers, d_hidden={}".format(n_layers, d_hidden))

    if load_in_4bit and BNB_AVAILABLE:
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model)

    # Wrap with LoRA (fresh or from checkpoint)
    if _is_peft:
        from peft import PeftModel
        print("=== Loading LoRA from SFT checkpoint ===")
        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)
    else:
        print("=== Applying fresh LoRA ===")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --- Tandem adapter setup (must happen BEFORE optimizer creation) ---
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
                print("  [TANDEM] No junior adapter found (no SFT checkpoint at "
                      "{}, no --tandem-junior-path). Disabling tandem.".format(
                          _sft_path))
                enable_tandem = False
        else:
            print("  [TANDEM] Junior adapter: {} (explicit)".format(
                tandem_junior_path))

    if enable_tandem:
        # Load junior adapter (frozen)
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

    # Optional torch.compile for forward/backward speedup
    if compile_model and not enable_tandem:
        try:
            model = torch.compile(model)
            print("  torch.compile: ENABLED")
        except Exception as e:
            print("  torch.compile: FAILED ({}), continuing without".format(e))
    elif compile_model and enable_tandem:
        print("  torch.compile: SKIPPED (incompatible with tandem adapter switching)")

    # Optimizer selection
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == "muon":
        from proofforge.optim.muon import Muon
        optimizer = Muon(trainable_params, lr=learning_rate, momentum=muon_momentum)
        print("  Optimizer: Muon (momentum={}, lr={})".format(muon_momentum, learning_rate))
    else:
        optimizer = AdamW(trainable_params, lr=learning_rate)
        print("  Optimizer: AdamW (lr={})".format(learning_rate))

    # Cosine LR schedule with floor at lr * min_lr_ratio
    # Adapted from PufferLib torch_pufferl.py:261-266.
    # Unlike get_cosine_schedule_with_warmup (decays to 0), this floors at
    # min_lr_ratio to preserve late-training gradient signal.
    num_warmup = max(1, int(0.05 * total_steps))
    scheduler = _make_cosine_with_floor(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

    print("\n=== Manual GRPO Training Config ===")
    print("  Model:        {}".format(model_name))
    print("  Total steps:  {}".format(total_steps))
    print("  Group size:   {}".format(group_size))
    print("  Batch size:   {}".format(batch_size))
    print("  Lean workers: {}".format(lean_workers))
    print("  Checkpoints:  {}".format(checkpoint_schedule))
    print("  Theorems:     {}".format(len(theorems)))
    print("  Temperatures: {}".format(temperatures))
    print("  Output dir:   {}".format(output_dir))
    print()

    # --- App 6 + App 8 init ---
    eff_oracle = None
    controller = None
    retention_tracker = None
    running_pass_rate = 0.0  # rolling estimate for efficiency phase-in
    eff_lr_was_reset = False  # flag: has LR schedule been restarted for efficiency?

    if APPS_AVAILABLE and enable_efficiency:
        eff_oracle = create_efficiency_oracle(
            phase_in_step=efficiency_phase_in_step,
            phase_in_pass_rate=efficiency_phase_in_rate,
            max_proof_tokens=max_new_tokens,
        )
        print("  App 6: Efficiency reward ENABLED (phase-in at step={}, rate={})".format(
            efficiency_phase_in_step, efficiency_phase_in_rate))
    else:
        print("  App 6: Efficiency reward DISABLED (pure binary)")

    if APPS_AVAILABLE and enable_controller:
        controller = create_controller(
            output_dir=str(output_dir),
            verbose=True,
        )
        # OBSERVATION MODE: override auto_apply to False for all intervention types
        controller.config.auto_apply_efficiency = False
        controller.config.auto_apply_checkpoints = False
        controller.phase_detector.reward_bias_correction = controller_bias_correction
        retention_tracker = RetentionTracker()
        print("  App 8: Controller ENABLED (observation mode -- no auto-apply, bias_correction={})".format(
            controller_bias_correction))
    else:
        print("  App 8: Controller DISABLED")

    # Collect at step 0 (baseline, before any training)
    if 0 in checkpoint_schedule:
        ckpt_dir = output_dir / "checkpoint_000"
        print("\n=== Checkpoint step=0 (baseline) ===")
        collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=0, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
            lean_workers=lean_workers,
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
    _tandem_zero_streak = 0  # safety guard for tandem auto-disable

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

            # Temperature annealing: linear decay from temp_start to temp_end
            progress = current_step / total_steps
            training_temp = temp_start + (temp_end - temp_start) * progress

            # Sample a batch of theorems
            if len(theorem_pool) >= batch_size:
                batch = random.sample(theorem_pool, batch_size)
            else:
                # Fewer theorems than batch size -- use all with repetition
                batch = [random.choice(theorem_pool) for _ in range(batch_size)]

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
            step_metrics_log.append(metrics)

            # Update rolling pass rate for efficiency phase-in
            running_pass_rate = 0.9 * running_pass_rate + 0.1 * metrics["mean_reward"]

            # --- LR schedule restart when efficiency activates ---
            if (eff_oracle is not None
                    and metrics.get("efficiency_active", False)
                    and not eff_lr_was_reset):
                remaining_steps = total_steps - current_step
                if remaining_steps > 10:
                    scheduler = _make_cosine_with_floor(
                        optimizer,
                        num_warmup_steps=max(1, int(0.05 * remaining_steps)),
                        num_training_steps=remaining_steps,
                        min_lr_ratio=min_lr_ratio,
                    )
                    eff_lr_was_reset = True
                    print("  [LR RESET] Efficiency activated at step {}. "
                          "Restarting cosine schedule (floor={}) for {} remaining steps.".format(
                              current_step, min_lr_ratio, remaining_steps))

            # --- App 8: Controller observation ---
            if controller is not None:
                interventions = controller_step(
                    controller=controller,
                    step=current_step,
                    reward=metrics["mean_reward"],
                    pass_rate=running_pass_rate,
                    loss=metrics["loss"],
                    proof_length=None,       # safe: metrics.py:126 guards with `if v is not None`
                    retention_rate=None,      # computed at checkpoints only
                    tactic_diversity=None,    # safe: metrics.py:130 guards with `if v is not None`
                )
                for intv in interventions:
                    print("  [CONTROLLER] {}".format(intv))

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
            if enable_tandem and "tandem_senior_frac" in metrics:
                print(
                    "           | sr_frac={:.2f} | handoffs={:.1f}".format(
                        metrics["tandem_senior_frac"],
                        metrics["tandem_handoffs_per_completion"],
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

        # Collect hidden states with the current trained model
        if enable_tandem:
            model.set_adapter(senior_adapter_name)  # senior-only for clean states
        summary = collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=target_step, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
            lean_workers=lean_workers,
        )
        all_summaries.append(summary)

        # --- App 8: Retention tracking ---
        if retention_tracker is not None and summary.get("theorem_results"):
            for tr in summary["theorem_results"]:
                retention_tracker.record(tr["id"], tr["passed_any"])
            retention_rate = retention_tracker.compute_and_rotate()
            print("  [RETENTION] rate={:.3f}".format(retention_rate))

        # --- App 6: Feed multi-temperature breadth data ---
        if eff_oracle is not None and summary.get("theorem_results"):
            for tr in summary["theorem_results"]:
                thm_id = tr["id"]
                for temp_str, temp_result in tr.get("by_temp", {}).items():
                    eff_oracle.breadth_tracker.record(
                        theorem_id=thm_id,
                        temperature=float(temp_str),
                        success=temp_result["success"],
                        proof_tokens=temp_result.get("generation_steps", 0),
                    )
            # Log breadth summary
            mastery = eff_oracle.breadth_tracker.mastery_summary()
            n_mastered = sum(1 for v in mastery.values() if v["breadth"] >= 3)
            n_moderate = sum(1 for v in mastery.values() if v["breadth"] == 2)
            n_fragile = sum(1 for v in mastery.values() if v["breadth"] < 2)
            print("  [BREADTH] mastered={} moderate={} fragile={}".format(
                n_mastered, n_moderate, n_fragile))

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
        "lean_workers": lean_workers,
        "checkpoint_schedule": checkpoint_schedule,
        "temperatures": list(temperatures),
        "n_theorems": len(theorems),
        "max_new_tokens": max_new_tokens,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "checkpoints": all_summaries,
        "step_metrics": step_metrics_log,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tandem_config": {
            "enabled": enable_tandem,
            "junior_path": tandem_junior_path,
            "handoff_prob": tandem_handoff_prob,
            "phase_in_step": tandem_phase_in_step,
        } if enable_tandem else None,
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
        description="ProofForge GRPO Training with Lean 4 Verification Reward"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="HuggingFace model name or local path",
    )
    # --- Stage selection ---
    parser.add_argument(
        "--stage", type=str, default="both", choices=["1", "2", "both"],
        help="Which stage: '1' (RS-SFT only), '2' (GRPO only), 'both' (default)"
    )
    parser.add_argument(
        "--skip-sft", action="store_true",
        help="Skip Stage 1 and go straight to GRPO with the base model"
    )
    parser.add_argument(
        "--n-completions-per-theorem", type=int, default=200,
        help="Completions per theorem in Stage 1 RS-SFT (default: 200)"
    )
    parser.add_argument(
        "--sft-epochs", type=int, default=2,
        help="SFT training epochs in Stage 1 (default: 2)"
    )
    parser.add_argument(
        "--sft-lr", type=float, default=2e-5,
        help="SFT learning rate in Stage 1 (default: 2e-5)"
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
    parser.add_argument(
        "--lean-workers",
        type=int,
        default=16,
        help="Number of parallel worker threads for Lean 4 verification",
    )
    # --- App 6: Efficiency Reward ---
    parser.add_argument(
        "--enable-efficiency", action="store_true",
        help="Enable App 6 efficiency reward (adaptive per-theorem alpha based on temperature breadth)"
    )
    parser.add_argument(
        "--efficiency-phase-in-step", type=int, default=50,
        help="Step at which efficiency reward activates (default: 50)"
    )
    parser.add_argument(
        "--efficiency-phase-in-rate", type=float, default=0.20,
        help="Pass rate threshold for efficiency activation (default: 0.20)"
    )
    # --- App 8: Adaptive Controller ---
    parser.add_argument(
        "--enable-controller", action="store_true",
        help="Enable App 8 adaptive controller (observation mode -- logs phase transitions, no auto-apply)"
    )
    parser.add_argument(
        "--controller-bias-correction", type=float, default=0.0,
        help="Reward bias correction for phase detector when Apps 2+6 depress observed reward (default: 0.0)"
    )
    # --- LR floor (PufferLib min_lr_ratio) ---
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1,
        help="Cosine LR schedule floor as fraction of base LR (default: 0.1). "
             "Prevents LR from decaying to 0 in late training."
    )
    # --- Optimizer selection ---
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
        help="Optimizer: 'adamw' (default) or 'muon' (Newton-Schulz orthogonal projection)"
    )
    parser.add_argument(
        "--muon-momentum", type=float, default=0.95,
        help="HeavyBall momentum for Muon optimizer (default: 0.95)"
    )
    # --- Learning rate (overridable for sweep) ---
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6,
        help="Base learning rate (default: 5e-6)"
    )
    # --- Temperature annealing ---
    parser.add_argument(
        "--temp-start", type=float, default=1.0,
        help="Starting sampling temperature (exploration phase, default: 1.0)"
    )
    parser.add_argument(
        "--temp-end", type=float, default=0.5,
        help="Ending sampling temperature (exploitation phase, default: 0.5)"
    )
    # --- QLoRA (4-bit quantization) ---
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load base model in 4-bit (QLoRA). Halves VRAM. Requires bitsandbytes."
    )
    # --- torch.compile ---
    parser.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile() to model. ~10-30%% forward/backward speedup. "
             "Requires PyTorch 2.4+. May not work with all PEFT versions."
    )
    # --- Tandem Training (West et al. 2026) ---
    parser.add_argument(
        "--no-tandem", action="store_true",
        help="Disable tactic-level tandem training (tandem is ON by default)."
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

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProofForge -- GRPO Training with Lean 4 Reward")
    print("=" * 60)
    print("  Model:        {}".format(args.model))
    print("  Stage:        {}".format(args.stage))
    print("  Steps:        {}".format(args.steps))
    print("  Group size:   {}".format(args.group_size))
    print("  Output dir:   {}".format(args.output_dir))
    print("  Checkpoints:  {}".format(args.checkpoint_schedule))
    print("  Theorems:     {}".format(args.theorems_path))
    print("  Temps:        {}".format(args.temperatures))
    print("  Max tokens:   {}".format(args.max_new_tokens))
    print("  Lean workers: {}".format(args.lean_workers))
    if args.stage in ("1", "both") and not args.skip_sft:
        print("  RS-SFT:       {} completions/thm x {} epochs @ lr={}".format(
            args.n_completions_per_theorem, args.sft_epochs, args.sft_lr))
    print()

    # Set up Lean 4
    print("=== Setting up Lean 4 ===")
    setup_lean()
    print()

    # Parse checkpoint schedule
    checkpoint_schedule = [
        int(x.strip())
        for x in args.checkpoint_schedule.split(",")
        if x.strip()
    ]
    if args.steps not in checkpoint_schedule:
        checkpoint_schedule.append(args.steps)
    checkpoint_schedule = sorted(set(checkpoint_schedule))
    print("  Final schedule: {}".format(checkpoint_schedule))
    print()

    # Validate theorems path
    if not os.path.exists(args.theorems_path):
        print("ERROR: theorems.json not found at {}".format(args.theorems_path))
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
        print("WARNING: No CUDA detected.")
    print()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if os.path.exists("/workspace"):
        os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
        Path("/workspace/hf_cache").mkdir(parents=True, exist_ok=True)

    stage = args.stage
    sft_weights_path = output_dir / "sft_checkpoint" / "model_weights"

    # ─── Stage 1: RS-SFT ───
    if stage in ("1", "both") and not args.skip_sft:
        print("\n=== Loading model for Stage 1 RS-SFT ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = dict(device_map="auto", trust_remote_code=True)
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
        load_kwargs["torch_dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        rejection_sampling_sft(
            model=model,
            tokenizer=tokenizer,
            theorems=collection_theorems,
            output_dir=output_dir,
            n_completions_per_theorem=args.n_completions_per_theorem,
            sft_epochs=args.sft_epochs,
            sft_lr=args.sft_lr,
            lean_workers=args.lean_workers,
            max_new_tokens=args.max_new_tokens,
        )

        # Free Stage 1 model
        del model, base_model
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    if stage == "1":
        print("\n=== Stage 1 only. Exiting. ===")
        return

    # ─── Stage 2: GRPO ───
    model_for_grpo = args.model
    if sft_weights_path.exists() and not args.skip_sft:
        model_for_grpo = str(sft_weights_path)
        print("\n[Stage 2] Loading from SFT checkpoint: {}".format(model_for_grpo))
    elif args.skip_sft:
        print("\n[Stage 2] --skip-sft: using base model {}".format(args.model))
    else:
        print("\n[Stage 2] No SFT checkpoint found. Using base model.")

    run_grpo(
        model_name=model_for_grpo,
        theorems=collection_theorems,
        total_steps=args.steps,
        group_size=args.group_size,
        output_dir=output_dir,
        checkpoint_schedule=checkpoint_schedule,
        temperatures=args.temperatures,
        max_new_tokens=args.max_new_tokens,
        lean_workers=args.lean_workers,
        enable_efficiency=getattr(args, 'enable_efficiency', False),
        enable_controller=getattr(args, 'enable_controller', False),
        efficiency_phase_in_step=getattr(args, 'efficiency_phase_in_step', 50),
        efficiency_phase_in_rate=getattr(args, 'efficiency_phase_in_rate', 0.20),
        controller_bias_correction=getattr(args, 'controller_bias_correction', 0.0),
        min_lr_ratio=getattr(args, 'min_lr_ratio', 0.1),
        optimizer_type=getattr(args, 'optimizer', 'adamw'),
        muon_momentum=getattr(args, 'muon_momentum', 0.95),
        learning_rate=getattr(args, 'learning_rate', 5e-6),
        temp_start=getattr(args, 'temp_start', 1.0),
        temp_end=getattr(args, 'temp_end', 0.5),
        load_in_4bit=getattr(args, 'load_in_4bit', False),
        compile_model=getattr(args, 'compile', False),
        enable_tandem=not getattr(args, 'no_tandem', False),
        tandem_junior_path=getattr(args, 'tandem_junior_path', None),
        tandem_handoff_prob=getattr(args, 'tandem_handoff_prob', 0.5),
        tandem_phase_in_step=getattr(args, 'tandem_phase_in_step', 0),
    )


if __name__ == "__main__":
    main()
