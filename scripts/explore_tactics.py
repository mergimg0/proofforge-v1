#!/usr/bin/env python3
"""
Tactic Exploration: expand the proof corpus by trying alternative tactics.

For each verified proof in the proof memory, try replacing each tactic
with alternatives and check if the result still compiles. This generates
new valid proofs, expanding the training corpus.

This is a simplified version of the LeanNavigator approach
(arXiv 2503.04772) that doesn't require the full LeanREPL infrastructure.
Instead, it works at the text level: parse tactics, substitute, re-check.

Usage:
  python3 scripts/explore_tactics.py [--max-proofs 20] [--max-alts 10]
"""

import json
import subprocess
import tempfile
import os
import sys
import itertools
from pathlib import Path
from typing import Optional

# Standard Lean 4 tactics to try as alternatives
TACTIC_ALTERNATIVES = {
    "simp": ["simp only", "simp_all", "simp [*]", "aesop", "omega", "decide", "norm_num"],
    "omega": ["simp", "decide", "norm_num", "linarith"],
    "decide": ["simp", "omega", "norm_num", "trivial", "rfl"],
    "rfl": ["trivial", "simp", "decide"],
    "trivial": ["simp", "rfl", "decide", "exact trivial"],
    "norm_num": ["simp", "omega", "decide"],
    "ring": ["simp", "omega", "norm_num"],
    "exact": [],  # too context-dependent
    "intro": [],  # structural, don't replace
    "induction": [],  # structural
    "cases": [],  # structural
    "constructor": [],  # structural
}


def check_lean(source: str, timeout: int = 30) -> bool:
    """Check if a Lean 4 source compiles."""
    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(source)
        tmpfile = f.name
    try:
        result = subprocess.run(
            ["lean", tmpfile],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(tmpfile)


def parse_top_tactic(proof: str) -> Optional[str]:
    """Extract the top-level tactic from a proof string."""
    proof = proof.strip()
    # Handle multi-line proofs: get the first tactic
    first_line = proof.split("\n")[0].strip()
    # Handle tactic combinators
    for tactic in TACTIC_ALTERNATIVES:
        if first_line.startswith(tactic):
            return tactic
    return None


def generate_alternatives(statement: str, proof: str, max_alts: int = 10) -> list[str]:
    """Generate alternative proofs by tactic substitution."""
    alternatives = []

    top_tactic = parse_top_tactic(proof)
    if top_tactic is None or top_tactic not in TACTIC_ALTERNATIVES:
        return alternatives

    for alt_tactic in TACTIC_ALTERNATIVES[top_tactic][:max_alts]:
        # Simple substitution: replace the tactic
        alt_proof = proof.replace(top_tactic, alt_tactic, 1)
        if alt_proof != proof:  # Only if actually different
            source = f"{statement}\n  {alt_proof}\n"
            if check_lean(source):
                alternatives.append(alt_proof)

    # Also try wrapping in tactic combinators
    simple_tactics = ["simp", "omega", "decide", "trivial", "rfl", "norm_num"]
    for tactic in simple_tactics:
        if tactic != proof.strip():
            source = f"{statement}\n  {tactic}\n"
            if check_lean(source):
                alternatives.append(tactic)

    return alternatives


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tactic exploration for proof corpus expansion")
    parser.add_argument("--max-proofs", type=int, default=20,
                       help="Max proofs to explore")
    parser.add_argument("--max-alts", type=int, default=8,
                       help="Max alternative tactics per proof")
    parser.add_argument("--input", type=str,
                       default=str(Path(__file__).parent.parent / "data" / "theorems.json"),
                       help="Input theorem dataset")
    parser.add_argument("--output", type=str,
                       default=str(Path(__file__).parent.parent / "data" / "expanded_proofs.json"),
                       help="Output expanded proofs")
    args = parser.parse_args()

    # Load theorem dataset
    with open(args.input) as f:
        dataset = json.load(f)

    all_theorems = []
    for tier in dataset["tiers"]:
        for t in tier["theorems"]:
            all_theorems.append(t)

    print(f"=== Tactic Exploration ===")
    print(f"Input: {len(all_theorems)} theorems")
    print(f"Max proofs to explore: {args.max_proofs}")
    print(f"Max alternatives per proof: {args.max_alts}")
    print()

    expanded = []
    total_new = 0

    for theorem in all_theorems[:args.max_proofs]:
        stmt = theorem["statement"]
        proof = theorem["known_proof"]

        print(f"  Exploring {theorem['id']}: {proof[:40]}...", end=" ", flush=True)

        alts = generate_alternatives(stmt, proof, args.max_alts)
        total_new += len(alts)

        expanded.append({
            "id": theorem["id"],
            "statement": stmt,
            "original_proof": proof,
            "alternative_proofs": alts,
            "num_alternatives": len(alts),
        })

        print(f"+{len(alts)} alternatives")

    # Save results
    result = {
        "description": "Expanded proof corpus via tactic exploration",
        "total_theorems": len(expanded),
        "total_new_proofs": total_new,
        "total_proofs": total_new + len(expanded),
        "theorems": expanded,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Results ===")
    print(f"  Original proofs: {len(expanded)}")
    print(f"  New alternatives: {total_new}")
    print(f"  Total proofs: {total_new + len(expanded)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
