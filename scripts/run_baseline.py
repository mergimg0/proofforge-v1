#!/usr/bin/env python3
"""
1d: Zero-shot baseline — measure DS-Prover-V2-7B performance
on ProofForge theorems WITHOUT any training.

This establishes the number to beat with GRPO.
"""

import json
import subprocess
import tempfile
import os
import time
import sys
from pathlib import Path

OLLAMA_MODEL = "deepseek-prover-v2:7b-q4"
OLLAMA_URL = "http://127.0.0.1:11434"
DATASET = Path(__file__).parent.parent / "data" / "theorems.json"


def ollama_generate(prompt: str, timeout: int = 30) -> str:
    """Call Ollama API for proof generation."""
    import requests
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 256,
                }
            },
            timeout=timeout
        )
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def lean_check(source: str, timeout: int = 30) -> bool:
    """Type-check Lean 4 source."""
    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        result = subprocess.run(["lean", tmp], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except:
        return False
    finally:
        os.unlink(tmp)


def main():
    attempts_per_theorem = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    with open(DATASET) as f:
        dataset = json.load(f)

    all_theorems = []
    for tier in dataset["tiers"]:
        for t in tier["theorems"]:
            all_theorems.append(t)

    print(f"=== Zero-Shot Baseline: {OLLAMA_MODEL} ===")
    print(f"Theorems: {len(all_theorems)}")
    print(f"Attempts per theorem: {attempts_per_theorem}")
    print()

    results = []
    total_verified = 0
    total_attempts = 0
    by_tier = {}

    for theorem in all_theorems:
        stmt = theorem["statement"]
        diff = theorem["difficulty"]
        tid = theorem["id"]

        prompt = f"Complete this Lean 4 proof. Output ONLY the tactic(s), nothing else.\n\n{stmt}\n"

        best_proof = None
        verified_count = 0

        for attempt_idx in range(attempts_per_theorem):
            total_attempts += 1
            proof = ollama_generate(prompt)

            # Clean up the proof (remove markdown, extra text)
            proof = proof.replace("```lean", "").replace("```", "").strip()
            lines = proof.split("\n")
            # Take only lines that look like tactics
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("--") and not stripped.startswith("theorem"):
                    clean_lines.append(line)
            proof = "\n".join(clean_lines) if clean_lines else proof

            source = f"{stmt}\n  {proof}\n"
            verified = lean_check(source)

            if verified:
                verified_count += 1
                total_verified += 1
                if best_proof is None or len(proof) < len(best_proof):
                    best_proof = proof

        # Track by tier
        if diff not in by_tier:
            by_tier[diff] = {"total": 0, "solved": 0}
        by_tier[diff]["total"] += 1
        if verified_count > 0:
            by_tier[diff]["solved"] += 1

        status = f"pass@{attempts_per_theorem}" if verified_count > 0 else "FAIL"
        print(f"  {tid}: {status} ({verified_count}/{attempts_per_theorem}) | {best_proof[:50] if best_proof else 'no proof'}...")

        results.append({
            "id": tid,
            "difficulty": diff,
            "attempts": attempts_per_theorem,
            "verified": verified_count,
            "solved": verified_count > 0,
            "best_proof": best_proof,
        })

    # Summary
    solved = sum(1 for r in results if r["solved"])
    print(f"\n=== Results ===")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  pass@1 (approx): {total_verified}/{total_attempts} = {total_verified/max(total_attempts,1):.1%}")
    print(f"  pass@{attempts_per_theorem}: {solved}/{len(all_theorems)} = {solved/len(all_theorems):.1%}")
    print(f"\n  By tier:")
    for d in sorted(by_tier.keys()):
        t = by_tier[d]
        print(f"    Tier {d}: {t['solved']}/{t['total']} = {t['solved']/max(t['total'],1):.1%}")

    # Save
    output_path = Path(__file__).parent.parent / "data" / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": OLLAMA_MODEL,
            "attempts_per_theorem": attempts_per_theorem,
            "total_theorems": len(all_theorems),
            "solved": solved,
            "pass_at_k": solved / len(all_theorems),
            "pass_at_1_approx": total_verified / max(total_attempts, 1),
            "by_tier": by_tier,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
