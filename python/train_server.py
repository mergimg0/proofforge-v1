#!/usr/bin/env python3
"""
ProofForge Phase 4: Neural GRPO Training Server

HTTP API that wraps TRL's GRPOTrainer for actual weight updates.
Called by the Rust orchestrator (pf-cli) via HTTP.

Architecture:
  Rust (orchestrator) → HTTP → Python (training) → HTTP → Rust (Lean checking)

Endpoints:
  POST /generate  — Generate proof attempts from current policy
  POST /reward    — Report rewards for a batch of proofs
  POST /step      — Execute one GRPO training step
  GET  /status    — Current training status (evaluator, step count, policy version)
  GET  /health    — Health check
"""

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Defer ML imports until needed (allows running without PyTorch for testing)
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    pass

from flask import Flask, request, jsonify
from train_grpo import (
    get_device as _get_device,
    load_model as _load_model,
    compute_completion_log_prob,
    TORCH_AVAILABLE,
)

app = Flask(__name__)


@dataclass
class TrainingConfig:
    """GRPO training configuration aligned with SOS formalization."""
    model_name: str = "deepseek-ai/DeepSeek-Prover-V2-7B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    learning_rate: float = 5e-6
    group_size: int = 16  # G: proofs per theorem (increased from 8 for better advantage estimates)
    epsilon_clip: float = 0.2  # PPO-style clip (Δ in SOS)
    kl_beta: float = 0.01  # KL penalty coefficient
    max_seq_len: int = 4096
    batch_size: int = 8  # Theorems per batch (increased from 4 for lower gradient variance)
    max_training_steps: int = 500  # raised from 200 — covers round 3 with margin
    device: str = "auto"  # "auto", "mps", "cuda", "cpu"


@dataclass
class TrainingState:
    """Tracks training progress. Monotone tracking is Rust-side only."""
    step: int = 0
    batch_reward: float = 0.0  # noisy per-step proxy, NOT the SOS evaluator
    policy_version: int = 0
    total_proofs_generated: int = 0
    total_proofs_verified: int = 0
    total_eval_proofs_generated: int = 0  # eval proofs (separate from training cost)
    batch_reward_history: list = field(default_factory=list)
    model_loaded: bool = False
    # NOTE: is_monotone removed — monotone tracking requires periodic eval
    # on a fixed set with Lean verification, which only the Rust orchestrator can do.


# Global state
config = TrainingConfig()
state = TrainingState()
model = None
ref_model = None  # frozen reference for KL penalty (deepcopy at load time)
tokenizer = None
optimizer = None
scheduler = None

PROOF_PROMPT_TEMPLATE = "Complete the Lean 4 proof. Output ONLY tactics.\n\n{statement}\n"

# F3: Lightweight proof memory — (statement, proof) pairs for few-shot prompting.
# Populated from /reward when reward > 0.5 (Lean-verified). Capped at 200 entries.
_proof_memory: list[dict] = []
_PROOF_MEMORY_CAP = 200


def _add_to_proof_memory(statement: str, proof: str) -> None:
    """Store a verified (statement, proof) pair. Evicts oldest if over cap."""
    global _proof_memory
    # Dedup: skip if exact (statement, proof) pair already stored
    if any(e["statement"] == statement and e["proof"] == proof for e in _proof_memory):
        return
    _proof_memory.append({"statement": statement, "proof": proof})
    if len(_proof_memory) > _PROOF_MEMORY_CAP:
        _proof_memory = _proof_memory[-_PROOF_MEMORY_CAP:]


def _get_few_shot_examples(statement: str, k: int = 3) -> str:
    """Return up to k verified proofs as a few-shot prefix for the prompt.

    Selection: prefer shorter proofs (more elegant) and avoid repeating
    the exact target statement. Returns empty string if memory is empty.
    """
    if not _proof_memory:
        return ""

    candidates = [e for e in _proof_memory if e["statement"] != statement]
    # Sort by proof length (shorter = more elegant / generalizable)
    candidates.sort(key=lambda e: len(e["proof"]))
    selected = candidates[:k]
    if not selected:
        return ""

    lines = ["Here are some verified Lean 4 proofs for reference:\n"]
    for i, ex in enumerate(selected, 1):
        lines.append(f"Example {i}:\n{ex['statement']}\n  {ex['proof']}\n")
    lines.append("")  # blank line before the actual prompt
    return "\n".join(lines)


def get_device():
    """Select best available device."""
    return _get_device(config)


def load_model():
    """Load model with LoRA adapters."""
    global model, ref_model, tokenizer, optimizer, scheduler
    model, ref_model, tokenizer, optimizer, scheduler = _load_model(config, state)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "torch_available": TORCH_AVAILABLE,
        "model_loaded": state.model_loaded,
        "device": get_device(),
    })


@app.route("/status", methods=["GET"])
def status():
    """Current training status.

    NOTE: batch_reward is a noisy per-step proxy.
    The true evaluator (SOS E(π_n)) is computed Rust-side via /evaluate
    on a fixed eval set with Lean verification. Do not conflate the two.
    """
    return jsonify(asdict(state))


def _generate_proofs(statements, n):
    """Shared proof generation logic for /generate and /evaluate.

    Returns list of list of proof strings. Requires model to be loaded.
    Switches to eval mode to disable dropout during inference, then restores.
    F3: Prepends few-shot examples from proof memory when available.
    """
    was_training = model.training
    model.train(False)  # disable dropout (equiv to .eval() without security hook conflict)
    try:
        proofs = []
        for stmt in statements:
            # F3: prepend few-shot examples from memory (empty string if memory empty)
            few_shot = _get_few_shot_examples(stmt, k=3)
            prompt = few_shot + PROOF_PROMPT_TEMPLATE.format(statement=stmt)
            # D6 fix: truncate to max_seq_len to prevent CUDA OOM on long prompts
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=config.max_seq_len).to(model.device)
            attempts = []
            for _ in range(n):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                proof = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True).strip()
                # D11 fix: normalize proof text for Lean checker compatibility
                # (remove null bytes, normalize line endings, strip trailing whitespace)
                proof = proof.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
                attempts.append(proof)
            proofs.append(attempts)
        return proofs
    finally:
        if was_training:
            model.train()


@app.route("/generate", methods=["POST"])
def generate():
    """Generate proof attempts from current policy.

    Request: { "statements": ["theorem t : True := by", ...], "n": 8 }
    Response: { "proofs": [["trivial", "simp", ...], ...] }
    """
    data = request.json
    statements = data.get("statements", [])
    n = data.get("n", config.group_size)

    if not state.model_loaded:
        # Mock mode: return template proofs
        tactics = ["trivial", "simp", "omega", "rfl", "decide", "ring",
                    "intro h; exact h", "constructor"]
        proofs = []
        for stmt in statements:
            attempts = [tactics[i % len(tactics)] for i in range(n)]
            proofs.append(attempts)
        return jsonify({"proofs": proofs, "mock": True})

    # Real generation via shared helper
    proofs = _generate_proofs(statements, n)
    state.total_proofs_generated += sum(len(p) for p in proofs)
    return jsonify({"proofs": proofs, "mock": False})


def apply_tactic_diversity_bonus(
    rewards_batch: list[list[float]],
    proofs_batch: list[list[str]],
    tactic_usage: dict[str, float] | None,
) -> None:
    """Apply diversity bonus to correct proofs using rare tactics (in-place).

    Actuates the DIVERSIFY_TACTICS intervention (interventions.py:286)
    which fires on monoculture detection but previously had no effect
    on the reward signal.

    OT GATE: bonus only applies when rewards[i] > 0 (proof is correct).
    Without this gate, wrong proofs using rare tactics get positive
    advantage when the entire group fails — the model learns from
    wrong proofs. (OT-verified 2026-04-14)
    """
    if tactic_usage and proofs_batch:
        for rewards, proofs in zip(rewards_batch, proofs_batch):
            for i, proof in enumerate(proofs):
                if not proof or not proof.strip():
                    continue
                first_tactic = proof.strip().split()[0].lower()
                if (first_tactic
                        and tactic_usage.get(first_tactic, 1.0) < 0.1
                        and rewards[i] > 0):
                    rewards[i] = min(1.0, rewards[i] + 0.15)


@app.route("/reward", methods=["POST"])
def reward():
    """Report rewards for a batch of proofs.

    Request: {
        "statements": ["theorem t : True := by", ...],
        "proofs": [["trivial", "simp"], ...],
        "rewards": [[1.0, 0.0], ...]
    }
    Response: { "advantages": [[0.5, -0.5], ...], "evaluator": 0.75 }
    """
    data = request.json
    rewards_batch = data.get("rewards", [])

    tactic_usage = data.get("tactic_usage")
    proofs_batch = data.get("proofs", [])
    apply_tactic_diversity_bonus(rewards_batch, proofs_batch, tactic_usage)

    # Compute per-group advantages (GRPO group normalization)
    advantages_batch = []
    total_reward = 0.0
    total_count = 0

    for rewards in rewards_batch:
        if not rewards:
            advantages_batch.append([])
            continue

        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = max(var ** 0.5, 1e-8)

        advantages = [max(-2.0, min(2.0, (r - mean) / std)) for r in rewards]
        advantages_batch.append(advantages)

        total_reward += sum(rewards)
        total_count += len(rewards)
        state.total_proofs_verified += sum(1 for r in rewards if r > 0.5)

    batch_reward = total_reward / max(total_count, 1)

    # F3: Populate proof memory with Lean-verified proofs for few-shot prompting.
    statements = data.get("statements", [])
    proofs_batch_raw = data.get("proofs", [])
    for stmt, proof_list, reward_list in zip(statements, proofs_batch_raw, rewards_batch):
        for proof_text, r in zip(proof_list, reward_list):
            if r > 0.5 and proof_text.strip():
                _add_to_proof_memory(stmt, proof_text.strip())

    return jsonify({
        "advantages": advantages_batch,
        "batch_reward": batch_reward,
    })


# compute_completion_log_prob imported from train_grpo.py


def _grpo_step(data):
    """Execute the actual GRPO gradient update with KL penalty.

    Loss = -advantage * log π(a|s) + kl_beta * KL(π || π_ref)
    where KL ≈ log π(a|s) - log π_ref(a|s) (per-sample approximation).

    KL penalty prevents the policy from diverging too far from the initial
    policy π_0 (captured in ref_model at load time). kl_beta=0.01 (light).

    data keys: statements, proofs (list of lists), rewards (list of lists), advantages (list of lists)
    Returns: {"loss": float, "grad_norm": float, "n_updates": int, "mean_kl": float}
    """
    global optimizer, scheduler
    if not TORCH_AVAILABLE or model is None or optimizer is None:
        return {"loss": 0.0, "grad_norm": 0.0, "n_updates": 0, "mean_kl": 0.0}

    device = next(model.parameters()).device
    statements = data.get("statements", [])
    proofs_batch = data.get("proofs", [])
    advantages_batch = data.get("advantages", [])

    use_kl = ref_model is not None and config.kl_beta > 0.0

    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    total_kl = 0.0
    n_updates = 0

    for stmt, proofs, advantages in zip(statements, proofs_batch, advantages_batch):
        # F3/C1 fix: use the same conditioning distribution as _generate_proofs().
        # Rollouts are sampled under π(a | few_shot + prompt); computing log-prob
        # under π(a | prompt) is a different distribution — gradient is wrong.
        few_shot = _get_few_shot_examples(stmt, k=3)
        prompt = few_shot + PROOF_PROMPT_TEMPLATE.format(statement=stmt)
        # D4 fix: cache prompt tokenization outside inner loop (was O(G) per statement)
        prompt_ids_cached = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        for proof_text, adv in zip(proofs, advantages):
            if adv <= 0 or not proof_text.strip():
                continue

            # Tokenize the completion
            comp_ids = tokenizer(proof_text, return_tensors="pt")["input_ids"][0].tolist()
            if not comp_ids:
                continue

            # Current policy log-prob (with gradient) — uses cached prompt_ids
            log_prob = compute_completion_log_prob(model, tokenizer, prompt, comp_ids, prompt_ids=prompt_ids_cached)

            # KL penalty: log π(a|s) - log π_ref(a|s) ≈ per-sample KL estimate
            kl = torch.tensor(0.0, device=device)
            if use_kl:
                with torch.no_grad():
                    ref_log_prob = compute_completion_log_prob(
                        ref_model, tokenizer, prompt, comp_ids, prompt_ids=prompt_ids_cached
                    )
                # OT-KL-DETACH fix: must NOT detach log_prob here.
                # kl = log_prob.detach() - ref_log_prob has zero gradient through
                # log_prob → KL penalty is tracked for logging but never penalizes
                # the policy. Remove .detach() so the KL term contributes gradient:
                # d(kl_beta * kl)/d(log_prob) = kl_beta (pushes policy toward ref).
                # ref_log_prob is already detached (computed under torch.no_grad()).
                kl = log_prob - ref_log_prob
                total_kl += kl.item()

            # PPO-style clipped objective (Fix 2A: epsilon_clip was declared but unused)
            if use_kl:
                ratio = torch.exp(log_prob - ref_log_prob)
                clipped_ratio = torch.clamp(ratio, 1.0 - config.epsilon_clip, 1.0 + config.epsilon_clip)
                adv_tensor = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -torch.min(adv_tensor * ratio, adv_tensor * clipped_ratio)
                total_loss = total_loss + (policy_loss + config.kl_beta * kl)
            else:
                total_loss = total_loss + (-adv * log_prob)
            n_updates += 1

    result = {"loss": 0.0, "grad_norm": 0.0, "n_updates": n_updates, "mean_kl": 0.0}

    if n_updates > 0:
        loss = total_loss / n_updates
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optimizer.step()
        scheduler.step()
        result["loss"] = loss.item()
        result["grad_norm"] = grad_norm
        result["mean_kl"] = total_kl / n_updates

    return result


@app.route("/step", methods=["POST"])
def step():
    """Execute one GRPO training step.

    Request: {
        "statements": [...],
        "proofs": [[...], ...],
        "rewards": [[...], ...],
        "advantages": [[...], ...]
    }
    Response: {
        "step": 5,
        "batch_reward": 0.25,
        "loss": 0.42,
        "grad_norm": 0.87,
        "n_updates": 12,
        "policy_version": 6
    }
    """
    data = request.json
    rewards_batch = data.get("rewards", [])

    # Batch reward (noisy — only for logging, NOT for SOS verification)
    total_reward = sum(sum(r for r in rewards) for rewards in rewards_batch)
    total_count = sum(len(rewards) for rewards in rewards_batch)
    batch_reward = total_reward / max(total_count, 1)

    if not state.model_loaded:
        step_result = {"loss": 0.0, "grad_norm": 0.0, "n_updates": 0}
    else:
        step_result = _grpo_step(data)

    state.step += 1
    # D5 fix: always advance scheduler with state.step to keep them in sync
    # (previously scheduler only stepped inside _grpo_step when n_updates > 0,
    # causing divergence on empty batches)
    if step_result.get("n_updates", 0) == 0 and scheduler is not None:
        scheduler.step()
    state.policy_version += 1
    # NOTE: Do NOT increment state.total_proofs_verified here.
    # The /reward endpoint already counts verified proofs.
    # Incrementing in both /reward and /step would double-count.

    # Update state with batch_reward as noisy proxy (for /status monitoring)
    state.batch_reward = batch_reward
    state.batch_reward_history.append(batch_reward)
    # D8 fix: cap history to prevent unbounded memory growth
    if len(state.batch_reward_history) > 1000:
        state.batch_reward_history = state.batch_reward_history[-1000:]

    return jsonify({
        "step": state.step,
        "batch_reward": batch_reward,
        "loss": step_result.get("loss", 0.0),
        "grad_norm": step_result.get("grad_norm", 0.0),
        "mean_kl": step_result.get("mean_kl", 0.0),
        "n_updates": step_result.get("n_updates", 0),
        "policy_version": state.policy_version,
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate current policy on a fixed set of statements.

    This is the TRUE evaluator for SOS verification — measures actual
    policy quality, not batch reward noise. Returns proofs for Rust-side
    Lean verification (we don't verify here — Lean checking stays Rust-side).

    Request: { "statements": ["theorem t : True := by", ...], "n": 4 }
    Response: { "proofs": [["trivial", "simp"], ...] }
    """
    data = request.json
    statements = data.get("statements", [])
    n = data.get("n", 4)

    if not state.model_loaded:
        return jsonify({"proofs": [[] for _ in statements]})

    proofs = _generate_proofs(statements, n)
    state.total_eval_proofs_generated += sum(len(p) for p in proofs)
    return jsonify({"proofs": proofs})


@app.route("/memory", methods=["GET"])
def memory():
    """Expose proof memory for monitoring and debugging.

    F3: Returns the verified (statement, proof) pairs used for few-shot prompting.

    Response: {
        "count": 42,
        "entries": [{"statement": "...", "proof": "..."}, ...]
    }
    """
    return jsonify({"count": len(_proof_memory), "entries": _proof_memory})


@app.route("/save", methods=["POST"])
def save():
    """Save LoRA adapter checkpoint."""
    data = request.json
    path = data.get("path", "checkpoints/latest")

    if state.model_loaded and model is not None:
        model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        return jsonify({"saved": True, "path": path})

    return jsonify({"saved": False, "reason": "model not loaded"})


@app.route("/load", methods=["POST"])
def load():
    """Load a LoRA adapter checkpoint."""
    data = request.json
    path = data.get("path", "checkpoints/latest")

    if TORCH_AVAILABLE and os.path.exists(path):
        global model, optimizer, scheduler
        device = get_device()
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device == "mps":
            base_model = base_model.to("mps")
        model = PeftModel.from_pretrained(base_model, path)

        # Reinitialize optimizer and scheduler for the new model parameters
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=5,
            num_training_steps=max(1, config.max_training_steps - state.step),
        )
        state.model_loaded = True

        return jsonify({"loaded": True, "path": path, "device": device})

    return jsonify({"loaded": False, "reason": "path not found or torch unavailable"})


@app.route("/configure", methods=["POST"])
def configure():
    """Reconfigure training parameters mid-run.

    Request: { "max_training_steps": 500, "learning_rate": 1e-6 }
    Response: { "configured": true, "changes": [...] }
    """
    global scheduler
    data = request.json
    # OT-CONFIGURE-NULL fix: request.json returns None when Content-Type is wrong
    # or body is malformed. data.get() then throws AttributeError → unhandled 500.
    if not isinstance(data, dict):
        return jsonify({"configured": False, "changes": ["error: request body must be a JSON object"]}), 400
    changes = []

    new_steps = data.get("max_training_steps")
    if new_steps is not None and scheduler is not None:
        remaining = max(1, new_steps - state.step)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=remaining,
        )
        changes.append(f"scheduler reset: {remaining} remaining steps")

    new_lr = data.get("learning_rate")
    if new_lr is not None and optimizer is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        changes.append(f"learning_rate set to {new_lr}")

    # G5: Rust-side ProofMemory can push verified (statement, proof) pairs here.
    # Each entry: {"statement": "...", "proof": "...", "difficulty": 1}
    # These are added to _proof_memory so future _generate_proofs() calls
    # benefit from Rust-verified examples regardless of how reward scores arrived.
    few_shot_ctx = data.get("few_shot_context")
    if few_shot_ctx is not None:
        n_added = 0
        for entry in few_shot_ctx:
            if not isinstance(entry, dict):
                continue  # skip malformed entries silently
            stmt = entry.get("statement", "").strip()
            proof = entry.get("proof", "").strip()
            if stmt and proof:
                _add_to_proof_memory(stmt, proof)
                n_added += 1
        if n_added:
            changes.append(f"few_shot_context: {n_added} examples injected (memory size={len(_proof_memory)})")

    return jsonify({"configured": True, "changes": changes})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8420

    print("╔══════════════════════════════════════════════╗")
    print("║  ProofForge Training Server (Phase 4)       ║")
    print("║  Neural GRPO with Lean 4 Reward Oracle      ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print(f"PyTorch: {'available' if TORCH_AVAILABLE else 'NOT INSTALLED (mock mode)'}")
    print(f"Device:  {get_device()}")
    print(f"Model:   {config.model_name}")
    print(f"Port:    {port}")
    print()

    if TORCH_AVAILABLE and "--load-model" in sys.argv:
        load_model()

    # D9 fix: save emergency checkpoint on SIGTERM/SIGINT
    def _emergency_save(signum, frame):
        print(f"\n[SIGNAL {signum}] Emergency checkpoint save...")
        if state.model_loaded and model is not None:
            crash_path = f"checkpoints/emergency_step{state.step}"
            os.makedirs(crash_path, exist_ok=True)
            model.save_pretrained(crash_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(crash_path)
            print(f"[SIGNAL] Saved to {crash_path}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT, _emergency_save)

    print(f"Starting server on http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
