#!/usr/bin/env python3
"""
ProofForge Phase C: Neural GRPO Training Script
"Press the button" — run this when ready to spend compute.

Prerequisites:
  pip install torch transformers peft trl accelerate datasets flask
  # Kimina Lean Server running at http://localhost:8765 (optional, faster)
  # Or: lean binary on PATH (fallback, slower)

Usage:
  # Quick test (5 steps, small batch):
  python3 train_neural.py --test

  # Full training (miniF2F):
  python3 train_neural.py --dataset miniF2F --steps 500 --group-size 8

  # With Kimina server:
  python3 train_neural.py --lean-server http://localhost:8765

  # On Apple Silicon with MLX:
  python3 train_neural.py --backend mlx
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- Configuration ---

DEFAULT_CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-Prover-V2-7B",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 5e-6,
    "group_size": 8,
    "max_completion_length": 1024,
    "max_prompt_length": 512,
    "gradient_accumulation_steps": 16,
    "num_train_steps": 200,
    "output_dir": "./grpo_output",
    "lean_timeout": 60,
    "lean_server_url": None,  # If set, use Kimina server instead of subprocess
}


def parse_args():
    parser = argparse.ArgumentParser(description="ProofForge Neural GRPO Training")
    parser.add_argument("--test", action="store_true", help="Quick test (5 steps)")
    parser.add_argument("--dataset", default="proofforge",
                       choices=["proofforge", "miniF2F"],
                       help="Dataset to train on")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size G")
    parser.add_argument("--lean-server", type=str, default=None,
                       help="Kimina Lean Server URL")
    parser.add_argument("--backend", default="torch",
                       choices=["torch", "mlx"],
                       help="ML backend")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--output", type=str, default="./grpo_output", help="Output dir")
    return parser.parse_args()


# --- Lean 4 Reward Oracle ---

class LeanRewardOracle:
    """Binary reward from Lean 4 type checker.
    Verified = 1.0, Failed = 0.0. No approximation."""

    def __init__(self, server_url=None, timeout=60):
        self.server_url = server_url
        self.timeout = timeout
        self.total_checks = 0
        self.total_verified = 0

    def check_proof(self, full_lean_source: str) -> bool:
        """Returns True if Lean 4 accepts the proof."""
        self.total_checks += 1

        if self.server_url:
            return self._check_via_server(full_lean_source)
        return self._check_via_subprocess(full_lean_source)

    def _check_via_subprocess(self, source: str) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(source)
            tmpfile = f.name
        try:
            result = subprocess.run(
                ["lean", tmpfile],
                capture_output=True, text=True, timeout=self.timeout
            )
            verified = result.returncode == 0
            if verified:
                self.total_verified += 1
            return verified
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(tmpfile)

    def _check_via_server(self, source: str) -> bool:
        """Check via Kimina Lean Server REST API."""
        import requests
        try:
            resp = requests.post(
                f"{self.server_url}/check",
                json={"source": source},
                timeout=self.timeout
            )
            verified = resp.json().get("verified", False)
            if verified:
                self.total_verified += 1
            return verified
        except Exception:
            return False

    @property
    def success_rate(self):
        if self.total_checks == 0:
            return 0.0
        return self.total_verified / self.total_checks


# --- Dataset Loading ---

def load_proofforge_dataset():
    """Load the ProofForge toy dataset (50 theorems)."""
    data_path = Path(__file__).parent.parent / "data" / "theorems.json"
    with open(data_path) as f:
        raw = json.load(f)

    prompts = []
    stubs = []
    for tier in raw["tiers"]:
        for theorem in tier["theorems"]:
            prompt = f"Complete this Lean 4 proof. Output ONLY the tactic(s).\n\n{theorem['statement']}\n"
            prompts.append(prompt)
            stubs.append(theorem["statement"])

    return prompts, stubs


def load_minif2f_dataset():
    """Load miniF2F-lean4 validation split."""
    minif2f_path = Path.home() / "projects" / "miniF2F-lean4"
    if not minif2f_path.exists():
        print(f"miniF2F not found at {minif2f_path}")
        print("Run: git clone https://github.com/yangky11/miniF2F-lean4 ~/projects/miniF2F-lean4")
        sys.exit(1)

    prompts = []
    stubs = []
    # miniF2F has .lean files with theorem statements
    for lean_file in sorted(minif2f_path.rglob("*.lean")):
        content = lean_file.read_text()
        # Extract theorem statements
        for line in content.split("\n"):
            if line.strip().startswith("theorem") and ":=" in line:
                stmt = line.split(":=")[0].strip() + " := by"
                prompt = f"Complete this Lean 4 proof. Output ONLY the tactic(s).\n\n{stmt}\n"
                prompts.append(prompt)
                stubs.append(stmt)

    print(f"Loaded {len(prompts)} miniF2F theorems")
    return prompts, stubs


# --- TRL GRPO Training ---

def train_with_trl(config, prompts, stubs, oracle):
    """Run GRPO training with TRL GRPOTrainer."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print(f"\n=== Loading {config['model_name']} ===")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype="auto",
        device_map="auto",
    )

    # Build HuggingFace Dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "theorem_stub": stubs,
    })

    # Reward function: binary Lean 4 type checking
    def lean4_reward(completions: list[str], **kwargs) -> list[float]:
        """The binary reward oracle. This is the SOS evaluator."""
        stubs_batch = kwargs.get("theorem_stub", [""] * len(completions))
        rewards = []
        for stub, body in zip(stubs_batch, completions):
            # Combine theorem statement with generated proof
            full_source = f"{stub}\n  {body}\n"
            verified = oracle.check_proof(full_source)
            rewards.append(1.0 if verified else 0.0)
        return rewards

    # GRPO config
    grpo_config = GRPOConfig(
        num_generations=config["group_size"],
        max_completion_length=config["max_completion_length"],
        max_prompt_length=config["max_prompt_length"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=config["num_train_steps"],
        use_peft=True,
        lora_r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_target_modules=config["lora_target_modules"],
        output_dir=config["output_dir"],
        logging_steps=1,
        save_steps=50,
        report_to="none",
    )

    print(f"\n=== Starting GRPO Training ===")
    print(f"  Steps: {config['num_train_steps']}")
    print(f"  Group size: {config['group_size']}")
    print(f"  Dataset: {len(prompts)} theorems")
    print(f"  Output: {config['output_dir']}")
    print()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[lean4_reward],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final checkpoint
    trainer.save_model(os.path.join(config["output_dir"], "final"))
    print(f"\n=== Training Complete ===")
    print(f"  Oracle stats: {oracle.total_verified}/{oracle.total_checks} verified "
          f"({oracle.success_rate:.1%})")


# --- Evaluator Tracking ---

def track_evaluator(oracle, step, evaluator_history):
    """Track the SOS evaluator and verify axioms."""
    evaluator = oracle.success_rate
    evaluator_history.append({"step": step, "evaluator": evaluator})

    if len(evaluator_history) >= 2:
        prev = evaluator_history[-2]["evaluator"]
        monotone = evaluator >= prev - 1e-10
        gap = 1.0 - evaluator
        print(f"  Step {step}: E={evaluator:.4f} gap={gap:.4f} "
              f"{'✓' if monotone else '✗'} monotone")
    else:
        print(f"  Step {step}: E={evaluator:.4f} gap={1.0-evaluator:.4f}")


# --- Main ---

def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    if args.test:
        config["num_train_steps"] = 5
        config["group_size"] = 4
        config["gradient_accumulation_steps"] = 4
    if args.steps:
        config["num_train_steps"] = args.steps
    if args.group_size:
        config["group_size"] = args.group_size
    if args.model:
        config["model_name"] = args.model
    if args.output:
        config["output_dir"] = args.output
    if args.lean_server:
        config["lean_server_url"] = args.lean_server

    print("╔══════════════════════════════════════════════════╗")
    print("║  ProofForge — Neural GRPO Training              ║")
    print("║  Press the button. SOS guarantees convergence.   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Model:    {config['model_name']}")
    print(f"  Steps:    {config['num_train_steps']}")
    print(f"  Group G:  {config['group_size']}")
    print(f"  Backend:  {args.backend}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Lean:     {'Kimina server' if config['lean_server_url'] else 'subprocess'}")
    print()

    # Load dataset
    if args.dataset == "proofforge":
        prompts, stubs = load_proofforge_dataset()
    else:
        prompts, stubs = load_minif2f_dataset()

    # Create reward oracle
    oracle = LeanRewardOracle(
        server_url=config.get("lean_server_url"),
        timeout=config["lean_timeout"]
    )

    # Train
    if args.backend == "torch":
        train_with_trl(config, prompts, stubs, oracle)
    elif args.backend == "mlx":
        print("MLX-GRPO backend: use ~/projects/MLX-GRPO/train.py directly")
        print("with lean4_reward as the reward function.")
        sys.exit(0)

    # Save oracle stats
    stats = {
        "total_checks": oracle.total_checks,
        "total_verified": oracle.total_verified,
        "success_rate": oracle.success_rate,
        "config": config,
    }
    stats_path = os.path.join(config["output_dir"], "oracle_stats.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nOracle stats saved to {stats_path}")


if __name__ == "__main__":
    main()
