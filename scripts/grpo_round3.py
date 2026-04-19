#!/usr/bin/env python3
"""
ProofForge -- Round 3: Two-Stage Training Pipeline

Stage 1: Rejection Sampling SFT (Cold Start)
  - Generate 200 completions per theorem x 50 theorems = 10,000 completions at T=1.2
  - Verify ALL with parallel Lean verification (8 workers)
  - Filter to only Lean-verified successes
  - SFT on successful proofs for 2 epochs, lr=2e-5
  - Goal: boost base rate from 2% to 10-20%

Stage 2: GRPO with Unlikeliness Reward
  - group_size=64 (or 32 if OOM)
  - lr=1e-6
  - Unlikeliness reward: r_i = R(x, y_i) * (1 - beta_rank * (G - rank_i) / G)
    where rank is by log-probability under current policy (most likely = rank 0)
  - Dynamic sampling: buffer of nonzero-advantage groups
  - Binary Lean verification as primary reward
  - 200 training steps with checkpoint schedule 0,5,10,20,30,50,75,100,150,200

Usage:
  python3 grpo_round3.py --stage both
  python3 grpo_round3.py --stage 1 --n-completions-per-theorem 200 --sft-epochs 2
  python3 grpo_round3.py --stage 2 --skip-sft
  python3 grpo_round3.py --stage 2 --group-size 32  # reduced for OOM

Prerequisites (RunPod PyTorch 2.4 template):
  pip install transformers peft accelerate datasets numpy sentencepiece protobuf
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


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
# Lean 4 reward: single-proof verification and batch parallel verification
# ---------------------------------------------------------------------------

# Custom definitions preamble for Level 4+ theorems (loaded from env or file)
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


def lean_verify_single(statement: str, proof_text: str, timeout: int = 30,
                        needs_definitions: bool = False) -> bool:
    """Verify a single proof with Lean 4. Returns True if type-checks."""
    cleaned = proof_text.replace('\u010a', '\n').replace('\u0120', ' ').strip()
    if not cleaned:
        return False
    proof_lines = cleaned.split('\n')
    indented = '\n'.join(f'  {line}' for line in proof_lines)

    # Prepend custom definitions for Level 4+ theorems
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
    results = [False] * len(statements)
    # needs_definitions can be a single bool or per-proof list
    if isinstance(needs_definitions, bool):
        defs_list = [needs_definitions] * len(statements)
    else:
        defs_list = needs_definitions
    # Use threads for Lean verification (subprocess I/O, no GIL contention)
    # Avoids fork+CUDA conflict that crashes on Linux/RunPod
    from concurrent.futures import ThreadPoolExecutor
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


def load_theorems_by_tier(data_path: str) -> list[list]:
    """Load theorems grouped by tier (for curriculum mixing).

    Returns a list of tier-lists, e.g. [[tier0_thm, ...], [tier1_thm, ...]].
    """
    with open(data_path) as f:
        raw = json.load(f)
    tiers = []
    for tier in raw.get("tiers", []):
        thms = tier.get("theorems", [])
        if thms:
            tiers.append(thms)
    return tiers


def build_curriculum_pool(tiers: list[list], easy_mix_ratio: float, max_theorems: int) -> list:
    """Build a weighted theorem pool for curriculum mixing.

    Tier 0 (easy / already-solved Level-1 theorems) are oversampled to
    easy_mix_ratio of the pool to prevent catastrophic forgetting.
    Remaining slots fill with tier 1+ (harder Level-2 theorems).

    If only one tier exists, falls back to a flat list.
    """
    if len(tiers) < 2:
        flat = [t for tier in tiers for t in tier]
        return flat[:max_theorems]

    easy = tiers[0]
    hard = [t for tier in tiers[1:] for t in tier]

    # Determine hard count then back-calculate easy count from ratio
    hard_count = min(len(hard), max_theorems)
    easy_count = min(
        len(easy),
        max(1, round(hard_count * easy_mix_ratio / max(1e-9, 1.0 - easy_mix_ratio))),
    )

    # Oversample easy by repeating if needed
    easy_pool = (easy * ((easy_count // len(easy)) + 1))[:easy_count]
    hard_pool = hard[:hard_count]

    pool = easy_pool + hard_pool
    print(
        f"Curriculum pool: {len(easy_pool)} easy (tier-0) + {len(hard_pool)} hard "
        f"= {len(pool)} total (easy_mix_ratio={easy_mix_ratio:.2f})"
    )
    return pool


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
    past_key_values = None
    t_start = time.time()

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

    if was_training:
        model.train()

    return summary


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_completions_no_grad(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> list:
    """Generate group_size completions for a single prompt (no gradient).

    Returns a list of dicts with keys: token_ids (list[int]), text (str).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    completions = []
    for _ in range(group_size):
        current_ids = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.clone()

        generated = []
        past_key_values = None
        for _step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tok_id = next_token.item()
            generated.append(tok_id)

            if tok_id == tokenizer.eos_token_id:
                break

            # With KV cache, only feed the new token
            current_ids = next_token.unsqueeze(0)
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
    full_ids = torch.cat([prompt_ids, completion_tensor], dim=1)

    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_token_ids)

    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    pred_logits = logits[0, prompt_len - 1: prompt_len + comp_len - 1, :]  # (comp_len, vocab)
    targets = completion_tensor[0]  # (comp_len,)

    log_probs = F.log_softmax(pred_logits, dim=-1)  # (comp_len, vocab)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]
    return token_log_probs.sum()


# ---------------------------------------------------------------------------
# Stage 1: Rejection Sampling SFT
# ---------------------------------------------------------------------------

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
    """Stage 1: Rejection sampling + SFT cold start.

    Generates n_completions_per_theorem completions per theorem at T=1.2,
    verifies them all with Lean, then SFT-trains on the successful proofs.

    Returns the path to the saved SFT checkpoint directory.
    """
    print("\n" + "=" * 60)
    print("  Stage 1: Rejection Sampling SFT (Cold Start)")
    print("=" * 60)

    sft_ckpt_dir = output_dir / "sft_checkpoint"
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Generate completions at high temperature for diversity
    # -----------------------------------------------------------------------
    print(f"\n[Stage 1] Generating {n_completions_per_theorem} completions x "
          f"{len(theorems)} theorems = "
          f"{n_completions_per_theorem * len(theorems)} total at T=1.2...")

    all_statements = []
    all_proofs = []
    all_needs_defs = []

    model.eval()  # noqa: B010 — PyTorch model method, not builtin eval
    t0 = time.time()

    for thm_idx, thm in enumerate(theorems):
        prompt = make_prompt(thm["statement"])
        completions = generate_completions_no_grad(
            model, tokenizer, prompt,
            group_size=n_completions_per_theorem,
            max_new_tokens=max_new_tokens,
            temperature=1.2,
        )
        needs_defs = thm.get("_requires_definitions", False)
        for comp in completions:
            all_statements.append(thm["statement"])
            all_proofs.append(comp["text"])
            all_needs_defs.append(needs_defs)

        print(f"  Theorem {thm_idx + 1}/{len(theorems)}: {thm['id']} -- "
              f"{len(completions)} completions generated "
              f"({time.time() - t0:.0f}s elapsed)")

    print(f"\n[Stage 1] Generation complete: {len(all_proofs)} completions in "
          f"{time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Verify all completions in parallel with Lean
    # -----------------------------------------------------------------------
    print(f"\n[Stage 1] Verifying {len(all_proofs)} completions with Lean "
          f"({lean_workers} workers)...")
    t_verify = time.time()

    verification_results = lean_verify_batch(
        all_statements, all_proofs,
        max_workers=lean_workers,
        timeout=30,
        needs_definitions=all_needs_defs,
    )

    n_success = sum(verification_results)
    print(f"[Stage 1] Verification complete in {time.time() - t_verify:.1f}s: "
          f"{n_success}/{len(verification_results)} passed "
          f"({100.0 * n_success / max(1, len(verification_results)):.1f}%)")

    # -----------------------------------------------------------------------
    # 3. Filter to successful proofs
    # -----------------------------------------------------------------------
    successful_pairs = [
        (stmt, proof)
        for stmt, proof, ok in zip(all_statements, all_proofs, verification_results)
        if ok
    ]

    # Deduplicate proofs per theorem to avoid overfitting to common tactics
    seen = set()
    deduped = []
    for stmt, proof in successful_pairs:
        key = (stmt, proof.strip())
        if key not in seen:
            seen.add(key)
            deduped.append((stmt, proof))
    n_before = len(successful_pairs)
    successful_pairs = deduped
    print(f"[Stage 1] Deduplicated: {n_before} → {len(successful_pairs)} unique proofs")

    # Save sampling summary
    sampling_summary = {
        "n_theorems": len(theorems),
        "n_completions_per_theorem": n_completions_per_theorem,
        "total_completions": len(all_proofs),
        "n_successes": n_success,
        "pass_rate": n_success / max(1, len(all_proofs)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(sft_ckpt_dir / "sampling_summary.json", "w") as f:
        json.dump(sampling_summary, f, indent=2)

    if not successful_pairs:
        print("\n[Stage 1] WARNING: Zero successful proofs found. "
              "Skipping SFT -- Stage 2 will use the base model.")
        return sft_ckpt_dir

    print(f"\n[Stage 1] Training SFT on {len(successful_pairs)} successful proofs "
          f"for {sft_epochs} epochs, lr={sft_lr}...")

    # -----------------------------------------------------------------------
    # 4. SFT training on successful proofs
    # -----------------------------------------------------------------------
    model.train()

    sft_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_lr,
    )

    steps_per_epoch = max(1, len(successful_pairs) // batch_size)
    total_sft_steps = sft_epochs * steps_per_epoch
    num_warmup = max(1, int(0.05 * total_sft_steps))
    sft_scheduler = get_cosine_schedule_with_warmup(
        sft_optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_sft_steps,
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
                logits = outputs.logits  # (1, seq_len, vocab)

                prompt_len = prompt_ids.shape[1]
                comp_len = len(proof_ids_list)

                # Cross-entropy loss on proof tokens only (prompt masked)
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
                sft_scheduler.step()

                loss_val = loss.item()
                epoch_loss_sum += loss_val
                epoch_steps += 1
                global_sft_step += 1
                sft_loss_log.append({"step": global_sft_step, "loss": loss_val})

                if global_sft_step % 10 == 0 or global_sft_step == 1:
                    print(f"  [SFT] epoch={epoch + 1} step={global_sft_step} "
                          f"loss={loss_val:.4f}")

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        print(f"\n[Stage 1] Epoch {epoch + 1}/{sft_epochs} complete -- "
              f"avg loss={avg_epoch_loss:.4f} over {epoch_steps} steps")

    # -----------------------------------------------------------------------
    # 5. Save SFT checkpoint
    # -----------------------------------------------------------------------
    weights_path = sft_ckpt_dir / "model_weights"
    print(f"\n[Stage 1] Saving SFT weights to {weights_path}")
    model.save_pretrained(str(weights_path))
    tokenizer.save_pretrained(str(weights_path))

    sft_training_summary = {
        "n_successes": len(successful_pairs),
        "sft_epochs": sft_epochs,
        "sft_lr": sft_lr,
        "total_steps": global_sft_step,
        "loss_log": sft_loss_log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(sft_ckpt_dir / "sft_training_summary.json", "w") as f:
        json.dump(sft_training_summary, f, indent=2)

    print(f"[Stage 1] Done. SFT checkpoint saved to {sft_ckpt_dir}")
    return sft_ckpt_dir


# ---------------------------------------------------------------------------
# Stage 2: GRPO with Unlikeliness Reward
# ---------------------------------------------------------------------------

def compute_unlikeliness_rewards(
    model,
    tokenizer,
    prompt: str,
    completions: list,
    lean_rewards: list,
    beta_rank: float = 0.25,
) -> list:
    """Compute unlikeliness-adjusted rewards for a group of completions.

    For each completion, computes log-probability under the current policy,
    ranks completions (0 = most likely), then applies:
        r_i_unlike = r_i * (1 - beta_rank * (G - rank_i) / G)

    Args:
        model: Current policy model (in eval mode for log-prob scoring)
        tokenizer: Tokenizer
        prompt: The prompt string
        completions: List of dicts with 'token_ids' and 'text'
        lean_rewards: List of float rewards (0.0 or 1.0) from Lean verification
        beta_rank: Penalty coefficient for high-probability completions (default 0.25)

    Returns:
        List of unlikeliness-adjusted float rewards, same length as completions
    """
    G = len(completions)
    if G == 0:
        return []

    # Compute log-prob for each completion under current policy (no grad needed)
    log_probs = []
    with torch.no_grad():
        for comp in completions:
            if len(comp["token_ids"]) == 0:
                log_probs.append(float("-inf"))
                continue
            lp = compute_completion_log_prob(
                model, tokenizer, prompt, comp["token_ids"]
            )
            log_probs.append(lp.item())

    # Rank by log-prob descending: index 0 = most likely (rank 0)
    sorted_indices = sorted(range(G), key=lambda i: log_probs[i], reverse=True)
    rank_of = [0] * G
    for rank_pos, comp_idx in enumerate(sorted_indices):
        rank_of[comp_idx] = rank_pos

    # Apply unlikeliness scaling
    unlikeliness_rewards = []
    for i, r in enumerate(lean_rewards):
        rank_i = rank_of[i]
        # rank_i = 0  -> most likely -> factor = 1 - beta_rank * G/G = 1 - beta_rank
        # rank_i = G-1 -> least likely -> factor = 1 - beta_rank * 1/G approx 1.0
        factor = 1.0 - beta_rank * (G - rank_i) / G
        unlikeliness_rewards.append(r * factor)

    return unlikeliness_rewards


def grpo_with_unlikeliness(
    model,
    tokenizer,
    theorems: list,
    output_dir: Path,
    model_name: str,
    d_hidden: int,
    n_layers: int,
    total_steps: int = 200,
    group_size: int = 64,
    lr: float = 1e-6,
    beta_rank: float = 0.25,
    checkpoint_schedule: list = None,
    temperatures: list = None,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    lean_workers: int = 8,
    buffer_target_size: int = 4,
    advantage_epsilon: float = 1e-4,
) -> dict:
    """Stage 2: GRPO training with unlikeliness reward.

    Includes dynamic sampling: groups are buffered until buffer_target_size
    groups with nonzero advantage accumulate, then one gradient update is made.

    Args:
        model: LoRA-wrapped model (should be loaded from SFT checkpoint)
        tokenizer: Tokenizer
        theorems: List of theorem dicts
        output_dir: Root output directory
        model_name: Model name string (for metadata)
        d_hidden: Hidden state dimension
        n_layers: Number of transformer layers
        total_steps: Total GRPO training steps
        group_size: Number of completions per prompt (G)
        lr: Learning rate
        beta_rank: Unlikeliness penalty coefficient
        checkpoint_schedule: Steps at which to collect hidden states
        temperatures: Sampling temperatures for hidden state collection
        max_new_tokens: Max tokens to generate
        batch_size: Theorems per gradient update (dynamic buffer target)
        lean_workers: Parallel Lean workers
        buffer_target_size: Number of nonzero-advantage groups to accumulate
            before performing a gradient update (dynamic sampling)
        advantage_epsilon: Minimum absolute advantage to count as nonzero

    Returns:
        Run summary dict
    """
    if checkpoint_schedule is None:
        checkpoint_schedule = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.2]

    print("\n" + "=" * 60)
    print("  Stage 2: GRPO with Unlikeliness Reward")
    print("=" * 60)
    print(f"  Total steps:   {total_steps}")
    print(f"  Group size:    {group_size}")
    print(f"  LR:            {lr}")
    print(f"  beta_rank:     {beta_rank}")
    print(f"  Lean workers:  {lean_workers}")
    print(f"  Checkpoints:   {checkpoint_schedule}")
    print(f"  Buffer target: {buffer_target_size}")
    print()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    num_warmup = max(1, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
    )

    # Collect at step 0 (baseline before GRPO)
    if 0 in checkpoint_schedule:
        ckpt_dir = output_dir / "checkpoint_000"
        print("\n=== Checkpoint step=0 (Stage 2 baseline) ===")
        step0_summary = collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=0, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
            lean_workers=lean_workers,
        )
        weights_path = ckpt_dir / "model_weights"
        print(f"  Saving weights to {weights_path}")
        model.save_pretrained(str(weights_path))
        tokenizer.save_pretrained(str(weights_path))

    schedule_points = sorted(s for s in checkpoint_schedule if s > 0)
    all_summaries = []
    # Include step 0 baseline if it was collected
    if 0 in checkpoint_schedule:
        all_summaries.append(step0_summary)
    step_metrics_log = []

    current_step = 0
    theorem_pool = list(theorems)
    model.train()

    # Dynamic sampling buffer: accumulate nonzero-advantage groups
    # before performing gradient updates
    buffer_groups = []   # list of theorem groups waiting for an update

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

            # Sample a single theorem for this step
            thm = random.choice(theorem_pool)
            prompt = make_prompt(thm["statement"])

            # --- Generate a group of completions ---
            model.eval()
            completions = generate_completions_no_grad(
                model, tokenizer, prompt,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # --- Verify all completions in parallel with Lean ---
            needs_defs = thm.get("_requires_definitions", False)
            verification_results = lean_verify_batch(
                [thm["statement"]] * len(completions),
                [c["text"] for c in completions],
                max_workers=lean_workers,
                timeout=30,
                needs_definitions=needs_defs,
            )
            raw_rewards = [1.0 if v else 0.0 for v in verification_results]

            # --- Compute unlikeliness-adjusted rewards ---
            unlike_rewards = compute_unlikeliness_rewards(
                model, tokenizer, prompt, completions, raw_rewards, beta_rank=beta_rank
            )

            unlike_tensor = torch.tensor(unlike_rewards, dtype=torch.float32)
            mean_r = unlike_tensor.mean().item()
            std_r = unlike_tensor.std().item()
            advantages = [(r - mean_r) / (std_r + 1e-8) for r in unlike_rewards]

            max_abs_adv = max(abs(a) for a in advantages) if advantages else 0.0
            has_nonzero = max_abs_adv > advantage_epsilon

            # Add to buffer if this group has useful signal
            if has_nonzero:
                buffer_groups.append({
                    "thm": thm,
                    "prompt": prompt,
                    "completions": completions,
                    "advantages": advantages,
                    "raw_rewards": raw_rewards,
                    "unlike_rewards": unlike_rewards,
                })

            # --- Perform gradient update when buffer reaches target size ---
            loss_val = 0.0
            n_updates = 0

            if len(buffer_groups) >= buffer_target_size:
                model.train()
                optimizer.zero_grad()
                accum_loss = 0.0
                # Count total positive-advantage completions first
                n_total_pos = sum(
                    1 for group in buffer_groups
                    for comp, adv in zip(group["completions"], group["advantages"])
                    if adv > 0 and len(comp["token_ids"]) > 0
                )
                for group in buffer_groups:
                    g_prompt = group["prompt"]
                    g_completions = group["completions"]
                    g_advantages = group["advantages"]

                    for comp, adv in zip(g_completions, g_advantages):
                        if adv <= 0 or len(comp["token_ids"]) == 0:
                            continue
                        log_prob = compute_completion_log_prob(
                            model, tokenizer, g_prompt, comp["token_ids"]
                        )
                        per_sample_loss = -adv * log_prob / max(1, n_total_pos)
                        per_sample_loss.backward()  # graph freed immediately
                        accum_loss += per_sample_loss.item()
                        n_updates += 1

                if n_updates > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    loss_val = accum_loss

                buffer_groups = []  # flush buffer after update

            scheduler.step()  # Step every training step, not just on flush

            metrics = {
                "step": current_step,
                "loss": loss_val,
                "mean_raw_reward": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
                "mean_unlike_reward": float(np.mean(unlike_rewards)) if unlike_rewards else 0.0,
                "mean_advantage": float(np.mean(advantages)) if advantages else 0.0,
                "n_updates": n_updates,
                "buffer_size": buffer_target_size if n_updates > 0 else len(buffer_groups),
                "has_nonzero_advantage": has_nonzero,
            }
            step_metrics_log.append(metrics)

            print(
                "  step {:4d} | loss={:.4f} | raw_r={:.3f} | unlike_r={:.3f} | "
                "adv={:.4f} | n_upd={} | buf={}".format(
                    current_step,
                    metrics["loss"],
                    metrics["mean_raw_reward"],
                    metrics["mean_unlike_reward"],
                    metrics["mean_advantage"],
                    metrics["n_updates"],
                    metrics["buffer_size"],
                )
            )

        # Checkpoint: collect hidden states
        # Clear GPU cache before checkpoint collection (OOM prevention)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n=== Checkpoint step={} ===".format(target_step))
        ckpt_dir = output_dir / "checkpoint_{:03d}".format(target_step)

        weights_path = ckpt_dir / "model_weights"
        print(f"  Saving weights to {weights_path}")
        model.save_pretrained(str(weights_path))
        tokenizer.save_pretrained(str(weights_path))

        summary = collect_at_checkpoint(
            model, tokenizer, theorems, ckpt_dir,
            step=target_step, temperatures=temperatures,
            max_new_tokens=max_new_tokens,
            d_hidden=d_hidden, n_layers=n_layers, model_name=model_name,
            lean_workers=lean_workers,
        )
        all_summaries.append(summary)

        print("\n  === Pass Rate Curve So Far ===")
        for s in all_summaries:
            bar = "#" * int(s["pass_rate"] * 40)
            print("  step {:4d}: {:.3f} |{}".format(s["step"], s["pass_rate"], bar))

        model.train()

    # Save run summary
    run_summary = {
        "model_name": model_name,
        "total_steps": total_steps,
        "group_size": group_size,
        "lr": lr,
        "beta_rank": beta_rank,
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
    }
    run_summary_path = output_dir / "grpo_run_summary.json"
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"\n=== Stage 2 complete. Summary saved to {run_summary_path} ===")

    print("\n" + "=" * 60)
    print("{:>6}  {:>10}  {:>8}  {:>6}".format("Step", "Pass Rate", "Passed", "Total"))
    print("-" * 60)
    for s in all_summaries:
        print("{:>6}  {:>10.3f}  {:>8}  {:>6}".format(
            s["step"], s["pass_rate"], s["passed"], s["total"]))
    print("=" * 60)

    return run_summary


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_base_model_with_lora(model_name: str, device_map: str = "auto"):
    """Load tokenizer and base model wrapped with LoRA."""
    print(f"\n=== Loading tokenizer: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"=== Loading model: {model_name} ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    d_hidden = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print(f"Model: {n_layers} layers, d_hidden={d_hidden}")

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

    return model, tokenizer, d_hidden, n_layers


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Run Stage 1, Stage 2, or both depending on args.stage."""

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Random seed: {seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set HuggingFace cache to persistent volume if available
    if os.path.exists("/workspace"):
        os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
        Path("/workspace/hf_cache").mkdir(parents=True, exist_ok=True)
        print("HF_HOME set to /workspace/hf_cache")

    # Parse checkpoint schedule
    checkpoint_schedule = [
        int(x.strip())
        for x in args.checkpoint_schedule.split(",")
        if x.strip()
    ]
    if args.steps not in checkpoint_schedule:
        checkpoint_schedule.append(args.steps)
    checkpoint_schedule = sorted(set(checkpoint_schedule))
    print(f"  Final checkpoint schedule: {checkpoint_schedule}")

    # Validate theorems path
    if not os.path.exists(args.theorems_path):
        print(f"ERROR: theorems file not found at {args.theorems_path}")
        sys.exit(1)

    tiers = load_theorems_by_tier(args.theorems_path)
    all_theorems = [t for tier in tiers for t in tier]
    if args.easy_mix_ratio > 0 and len(tiers) > 1:
        theorems = build_curriculum_pool(tiers, args.easy_mix_ratio, args.max_theorems)
    else:
        theorems = all_theorems[: args.max_theorems]
    print(f"Using {len(theorems)} theorems (dataset has {len(all_theorems)} total)")

    # Report GPU state
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gb:.1f} GB)")
    else:
        print("WARNING: No CUDA detected. Running on CPU will be very slow.")

    stage = args.stage
    sft_weights_path = output_dir / "sft_checkpoint" / "model_weights"

    if stage in ("1", "both"):
        # Load base model + LoRA for Stage 1
        model, tokenizer, d_hidden, n_layers = load_base_model_with_lora(args.model)

        rejection_sampling_sft(
            model=model,
            tokenizer=tokenizer,
            theorems=theorems,
            output_dir=output_dir,
            n_completions_per_theorem=args.n_completions_per_theorem,
            sft_epochs=args.sft_epochs,
            sft_lr=args.sft_lr,
            lean_workers=args.lean_workers,
            max_new_tokens=args.max_new_tokens,
            batch_size=4,
        )

    # Clear GPU cache between stages to prevent fragmentation OOM
    if stage == "both" and torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[OOM prevention] GPU cache cleared between Stage 1 and Stage 2")

    if stage in ("2", "both"):
        if args.skip_sft or not sft_weights_path.exists():
            if args.skip_sft:
                print("\n[Stage 2] --skip-sft specified. Loading base model.")
            else:
                print("\n[Stage 2] No SFT checkpoint found. Loading base model.")
            model, tokenizer, d_hidden, n_layers = load_base_model_with_lora(args.model)
        elif stage == "2":
            # Stage 2 only -- load from SFT checkpoint on disk
            print(f"\n[Stage 2] Loading SFT weights from {sft_weights_path}")
            model, tokenizer, d_hidden, n_layers = load_base_model_with_lora(
                str(sft_weights_path)
            )
        # else: stage == "both" -- model is already loaded and SFT-trained in memory

        grpo_with_unlikeliness(
            model=model,
            tokenizer=tokenizer,
            theorems=theorems,
            output_dir=output_dir,
            model_name=args.model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            total_steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            beta_rank=args.beta_rank,
            checkpoint_schedule=checkpoint_schedule,
            temperatures=args.temperatures,
            max_new_tokens=args.max_new_tokens,
            batch_size=4,
            lean_workers=args.lean_workers,
            buffer_target_size=4,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProofForge Round 3: Two-Stage GRPO Pipeline (RS-SFT + Unlikeliness Reward)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Which stage to run: 1 (RS-SFT), 2 (GRPO), or both (default: both)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total GRPO training steps (Stage 2)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="GRPO group size G (default: 64, reduce to 32 if OOM)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="GRPO learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--beta-rank",
        type=float,
        default=0.25,
        help="Unlikeliness penalty coefficient (default: 0.25)",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=2,
        help="SFT training epochs in Stage 1 (default: 2)",
    )
    parser.add_argument(
        "--sft-lr",
        type=float,
        default=2e-5,
        help="SFT learning rate in Stage 1 (default: 2e-5)",
    )
    parser.add_argument(
        "--n-completions-per-theorem",
        type=int,
        default=200,
        help="Number of completions to generate per theorem in Stage 1 (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/grpo_round3",
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
        default="/workspace/proofforge/data/theorems_level1_2.json",
        help="Path to theorems dataset (default: Level 1-2 curriculum)",
    )
    parser.add_argument(
        "--easy-mix-ratio",
        type=float,
        default=0.3,
        help="Fraction of curriculum pool from tier-0 easy theorems (prevents forgetting, default: 0.3)",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=50,
        help="Max theorems to use",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per trajectory",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.3, 0.7, 1.2],
        help="Sampling temperatures for hidden state collection",
    )
    parser.add_argument(
        "--lean-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes for Lean 4 verification",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        default=False,
        help="Skip Stage 1 SFT; load base model directly for Stage 2",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProofForge -- Round 3 Training Pipeline")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Stage:          {args.stage}")
    print(f"  Steps:          {args.steps}")
    print(f"  Group size:     {args.group_size}")
    print(f"  LR (GRPO):      {args.lr}")
    print(f"  beta_rank:      {args.beta_rank}")
    print(f"  SFT epochs:     {args.sft_epochs}")
    print(f"  SFT LR:         {args.sft_lr}")
    print(f"  Completions/thm:{args.n_completions_per_theorem}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Checkpoints:    {args.checkpoint_schedule}")
    print(f"  Theorems:       {args.theorems_path}")
    print(f"  Max theorems:   {args.max_theorems}")
    print(f"  Easy mix ratio: {args.easy_mix_ratio}")
    print(f"  Temps:          {args.temperatures}")
    print(f"  Max tokens:     {args.max_new_tokens}")
    print(f"  Lean workers:   {args.lean_workers}")
    print(f"  Skip SFT:       {args.skip_sft}")
    print()

    print("=== Setting up Lean 4 ===")
    setup_lean()
    print()

    run_pipeline(args)


if __name__ == "__main__":
    main()
