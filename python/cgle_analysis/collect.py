#!/usr/bin/env python3
"""
Phase 2 -- Hidden State Collection Pipeline

Runs inference on theorem statements, records the final-layer hidden state
at each autoregressive generation step, and labels each trajectory with
the Lean 4 verification result.

Output: one .npz file per proof attempt containing:
  - hidden_states: (T_steps, d_hidden) float16 array
  - tokens: (T_steps,) int32 array of generated token IDs
  - success: bool -- did Lean 4 accept the proof?
  - temperature: float -- sampling temperature used
  - statement: str -- the theorem statement
  - proof_text: str -- the generated proof body

Usage:
  python3 -m cgle_analysis.collect --model deepseek-ai/DeepSeek-Prover-V2-7B
  python3 -m cgle_analysis.collect --model ./grpo_output/checkpoint-100
  python3 -m cgle_analysis.collect --test
  python3 -m cgle_analysis.collect --temperatures 0.3 0.7 1.2 --attempts 3
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="CGLE Phase 2: Hidden State Collection")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-Prover-V2-7B",
                        help="Model name or checkpoint path")
    parser.add_argument("--output-dir", type=str,
                        default="./cgle_data",
                        help="Directory to save trajectory data")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.7],
                        help="Sampling temperatures to sweep")
    parser.add_argument("--attempts", type=int, default=1,
                        help="Number of proof attempts per theorem per temperature")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate per proof attempt")
    parser.add_argument("--dataset", type=str, default="proofforge",
                        choices=["proofforge", "miniF2F"],
                        help="Which theorem dataset to use")
    parser.add_argument("--lean-server", type=str, default=None,
                        help="Kimina Lean Server URL (faster verification)")
    parser.add_argument("--lean-timeout", type=int, default=60,
                        help="Lean type-checking timeout in seconds")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 5 theorems, 1 temperature, 1 attempt")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda, mps, cpu)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer to record (-1 = last)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip Lean verification (label all as unverified)")
    return parser.parse_args()


# --- Lean 4 Verification ---

def verify_lean(source: str, server_url: str = None, timeout: int = 60) -> bool:
    """Check if Lean 4 accepts the proof."""
    if server_url:
        import requests
        try:
            resp = requests.post(
                f"{server_url}/check",
                json={"source": source},
                timeout=timeout,
            )
            return resp.json().get("verified", False)
        except Exception:
            return False

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(source)
        tmpfile = f.name
    try:
        result = subprocess.run(
            ["lean", tmpfile],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(tmpfile)


# --- Dataset Loading ---

def load_theorems(dataset_name: str) -> list[dict]:
    """Load theorem statements."""
    if dataset_name == "proofforge":
        data_path = Path(__file__).parent.parent.parent / "data" / "theorems.json"
        with open(data_path) as f:
            raw = json.load(f)
        theorems = []
        for tier in raw["tiers"]:
            for thm in tier["theorems"]:
                theorems.append(thm)
        return theorems
    elif dataset_name == "miniF2F":
        minif2f_path = Path.home() / "projects" / "miniF2F-lean4"
        if not minif2f_path.exists():
            print(f"miniF2F not found at {minif2f_path}")
            sys.exit(1)
        theorems = []
        for lean_file in sorted(minif2f_path.rglob("*.lean")):
            content = lean_file.read_text()
            for line in content.split("\n"):
                if line.strip().startswith("theorem") and ":=" in line:
                    stmt = line.split(":=")[0].strip() + " := by"
                    theorems.append({"id": lean_file.stem, "statement": stmt})
        return theorems
    return []


# --- Hidden State Extraction ---

@torch.no_grad()
def generate_with_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    layer_idx: int = -1,
) -> dict:
    """Generate a proof and record hidden states at each autoregressive step.

    Returns dict with:
      - hidden_states: (T_steps, d_hidden) numpy array (float16)
      - tokens: (T_steps,) numpy array of generated token IDs
      - proof_text: str
      - generation_time: float
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hidden_states_list = []
    generated_tokens = []

    current_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    t_start = time.time()

    for step in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Hidden state at the last token position of the specified layer
        layer_hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, d_hidden)
        last_token_hidden = layer_hidden[0, -1, :]  # (d_hidden,)
        hidden_states_list.append(last_token_hidden.cpu().to(torch.float16).numpy())

        # Sample next token
        logits = outputs.logits[0, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype),
            ], dim=1)

    generation_time = time.time() - t_start

    hidden_states = np.stack(hidden_states_list, axis=0)  # (T_steps, d_hidden)
    tokens = np.array(generated_tokens, dtype=np.int32)
    proof_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return {
        "hidden_states": hidden_states,
        "tokens": tokens,
        "proof_text": proof_text,
        "generation_time": generation_time,
    }


# --- Main Collection Loop ---

def collect(args):
    """Main collection pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theorems = load_theorems(args.dataset)
    if args.test:
        theorems = theorems[:5]
        args.temperatures = [0.7]
        args.attempts = 1
    print(f"Loaded {len(theorems)} theorems from {args.dataset}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    # Set model to inference mode
    model.requires_grad_(False)
    model.train(False)

    d_hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, d_hidden={d_hidden}, device={device}")

    total = 0
    verified_count = 0

    for temp in args.temperatures:
        for thm in theorems:
            for attempt in range(args.attempts):
                total += 1
                prompt = (
                    f"Complete this Lean 4 proof. "
                    f"Output ONLY the tactic(s).\n\n{thm['statement']}\n"
                )

                result = generate_with_hidden_states(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=temp,
                    layer_idx=args.layer,
                )

                if args.skip_verify:
                    success = False  # Will verify locally after download
                else:
                    full_source = f"{thm['statement']}\n  {result['proof_text']}\n"
                    success = verify_lean(
                        full_source,
                        server_url=args.lean_server,
                        timeout=args.lean_timeout,
                    )
                if success:
                    verified_count += 1

                traj_id = f"{thm['id']}_T{temp}_a{attempt}"
                save_path = output_dir / f"{traj_id}.npz"
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
                    layer_idx=np.array(args.layer),
                    d_hidden=np.array(d_hidden),
                    n_layers=np.array(n_layers),
                    model_name=np.array(args.model),
                )

                status = "VERIFIED" if success else "failed"
                steps = result["hidden_states"].shape[0]
                print(
                    f"[{total:4d}] {traj_id:30s} | "
                    f"{steps:3d} steps | "
                    f"{result['generation_time']:.1f}s | "
                    f"{status}"
                )

    print(f"\n{'='*60}")
    print(f"Collection complete: {total} trajectories")
    print(f"  Verified: {verified_count}/{total} ({100*verified_count/max(total,1):.1f}%)")
    print(f"  Temperatures: {args.temperatures}")
    print(f"  Hidden dim: {d_hidden}")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}")

    manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "temperatures": args.temperatures,
        "attempts_per_theorem": args.attempts,
        "total_trajectories": total,
        "verified": verified_count,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "layer_recorded": args.layer,
        "max_tokens": args.max_tokens,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    args = parse_args()
    collect(args)
