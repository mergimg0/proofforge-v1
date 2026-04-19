#!/usr/bin/env python3
"""App 1: GRPO Training with Code Verification Reward.

Applies the ProofForge pattern to code generation:
  - Generate Python code solutions (GPU, HuggingFace model + LoRA)
  - Verify via sandboxed test suite execution (CPU, binary reward)
  - GRPO update with group normalization

Demonstrates the infrastructure learning hypothesis: the model learns
coding PATTERNS that transfer across tasks, producing the same
three-phase dynamics observed in Lean proof training.

Usage:
  python3 grpo_code_verify.py --steps 100 --group-size 8
  python3 grpo_code_verify.py --model deepseek-ai/deepseek-coder-7b-instruct-v1.5

Prerequisites:
  pip install transformers peft accelerate numpy sentencepiece protobuf
  pip install -e python/
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from proofforge.code_verify.task_loader import TaskLoader
from proofforge.code_verify.tactic_registry import TacticRegistry
from proofforge.rewards.test_suite import TestSuiteOracle


def compute_log_prob(model, tokenizer, prompt: str, completion_ids: list) -> torch.Tensor:
    """Compute sum of log probabilities for completion tokens. Returns scalar (requires grad)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    comp_tensor = torch.tensor([completion_ids], dtype=torch.long, device=model.device)
    full_ids = torch.cat([prompt_ids, comp_tensor], dim=1)
    outputs = model(input_ids=full_ids, use_cache=False)
    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]
    comp_len = len(completion_ids)
    if comp_len == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    pred_logits = logits[0, prompt_len - 1: prompt_len + comp_len - 1, :]
    targets = comp_tensor[0]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(comp_len, device=model.device), targets]
    return token_log_probs.sum()


def generate_solutions(model, tokenizer, prompt: str, n: int,
                       temperature: float = 0.7, max_tokens: int = 512) -> list[dict]:
    """Generate n code solutions. Returns list of {token_ids, text}."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    solutions = []
    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output[0][inputs["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        solutions.append({"token_ids": gen_ids, "text": text})
    return solutions


def main():
    parser = argparse.ArgumentParser(description="App 1: GRPO Code Verification Training")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-7b-instruct-v1.5")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="/workspace/grpo_code_verify")
    parser.add_argument("--tasks-json", default=None, help="Extra tasks JSON file")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model + LoRA ---
    print("=== Loading model: {} ===".format(args.model))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --- Set up task infrastructure ---
    loader = TaskLoader()
    if args.tasks_json:
        n_loaded = loader.load_from_json(args.tasks_json)
        print("Loaded {} extra tasks from {}".format(n_loaded, args.tasks_json))
    print("Task pool: {} tasks across {}".format(loader.task_count, loader.level_summary()))

    oracle = TestSuiteOracle(timeout=30, memory_mb=256, workers=4, require_safe_imports=True)
    for task in loader.all_tasks:
        oracle.register_task(task.task_id, task.test_code)

    registry = TacticRegistry()

    # --- Optimizer ---
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.05 * args.steps)),
        num_training_steps=args.steps)

    print("\n=== GRPO Code Verification Training ===")
    print("  Steps: {}, Group size: {}, Tasks: {}".format(args.steps, args.group_size, loader.task_count))

    step_log = []
    model.train()

    for step in range(1, args.steps + 1):
        task = random.choice(loader.all_tasks)

        # 1. Generate solutions
        model.eval()
        solutions = generate_solutions(
            model, tokenizer, task.prompt, args.group_size,
            temperature=0.7, max_tokens=args.max_tokens)
        model.train()

        # 2. Verify via test suite
        pairs = [(task.prompt, s["text"]) for s in solutions]
        results = oracle.batch_evaluate(pairs, task_ids=[task.task_id] * len(pairs))
        rewards = [r.reward for r in results]

        # 3. Track infrastructure patterns
        for sol, result in zip(solutions, results):
            registry.record(sol["text"], result.verified)

        # 4. GRPO advantages
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean().item()
        std_r = rewards_t.std().item()
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

        # 5. Gradient update (positive-advantage only)
        optimizer.zero_grad()
        n_updates = 0
        total_loss = torch.tensor(0.0, device=model.device)
        n_pos = sum(1 for _i, (sol, adv) in enumerate(zip(solutions, advantages))
                    if adv > 0 and len(sol["token_ids"]) > 0)

        for sol, adv in zip(solutions, advantages):
            if adv <= 0 or len(sol["token_ids"]) == 0:
                continue
            lp = compute_log_prob(model, tokenizer, task.prompt, sol["token_ids"])
            loss_term = -adv * lp / max(1, n_pos)
            loss_term.backward()
            n_updates += 1
            total_loss = total_loss + loss_term.detach()

        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        pass_rate = sum(rewards) / len(rewards)
        metrics = {"step": step, "task": task.task_id, "level": int(task.level),
                   "loss": total_loss.item(), "pass_rate": pass_rate,
                   "mean_reward": mean_r, "n_updates": n_updates}
        step_log.append(metrics)

        print("  step {:4d} | {:20s} | L{} | loss={:.4f} | pr={:.3f} | updates={}".format(
            step, task.task_id, int(task.level), total_loss.item(), pass_rate, n_updates))

        # Checkpoint every 25 steps
        if step % 25 == 0 or step == args.steps:
            ckpt_path = output_dir / "checkpoint_{:04d}".format(step)
            ckpt_path.mkdir(exist_ok=True)
            model.save_pretrained(str(ckpt_path / "model_weights"))
            tokenizer.save_pretrained(str(ckpt_path / "model_weights"))
            with open(ckpt_path / "metrics.json", "w") as f:
                json.dump({"step_log": step_log, "patterns": registry.summary()}, f, indent=2)
            print("  [CHECKPOINT] Saved to {}".format(ckpt_path))

    # Final summary
    summary = {"model": args.model, "steps": args.steps, "group_size": args.group_size,
               "task_count": loader.task_count, "pattern_masteries": registry.all_masteries(),
               "step_log": step_log}
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== Training complete. Summary: {} ===".format(output_dir / "run_summary.json"))


if __name__ == "__main__":
    main()
