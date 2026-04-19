"""Analysis 9: Tactic composition detection."""

import re

import numpy as np

from ._shared import TACTIC_KEYWORDS, TACTIC_SET


def detect_composition(proof_text: str) -> dict:
    text = proof_text.strip()
    if not text:
        return {"has_semicolon": False, "has_newline_tactics": False, "n_tactics": 0, "tactics": [], "tactic_lines": 0}

    has_semi = ";" in text

    tactics_found = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("--"):
            continue
        for tac in TACTIC_KEYWORDS:
            if re.search(r'\b' + re.escape(tac) + r'\b', stripped):
                tactics_found.append(tac)

    unique_tactics = list(dict.fromkeys(tactics_found))

    tactic_lines = 0
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped and not stripped.startswith("--") and not stripped.startswith("#"):
            first_token = re.split(r"[\s(;{]+", stripped)[0].rstrip(":")
            if first_token in TACTIC_SET:
                tactic_lines += 1

    return {
        "has_semicolon": has_semi,
        "has_newline_tactics": tactic_lines >= 2,
        "n_tactics": len(unique_tactics),
        "tactics": unique_tactics,
        "tactic_lines": tactic_lines,
    }


def analysis_tactic_composition(checkpoints, all_theorems):
    print("\n" + "=" * 72)
    print("ANALYSIS 9: Tactic composition detection")
    print("=" * 72)

    print(f"\n{'Checkpoint':<12}  {'N_pass':<8}  {'Has_semi':<10}  {'Multi_line':<12}  {'Multi_tactic':<14}  {'Avg_tactics':<12}")
    print("-" * 76)

    for step, records_by_thm in checkpoints:
        successes = []
        for tid in all_theorems:
            for rec in records_by_thm.get(tid, []):
                if rec["success"]:
                    successes.append(rec)

        n_pass = len(successes)
        n_semi = 0
        n_multi_line = 0
        n_multi_tactic = 0
        tactic_counts = []

        for rec in successes:
            comp = detect_composition(rec["proof_text"])
            if comp["has_semicolon"]:
                n_semi += 1
            if comp["has_newline_tactics"]:
                n_multi_line += 1
            if comp["n_tactics"] >= 2:
                n_multi_tactic += 1
            tactic_counts.append(comp["n_tactics"])

        avg_tac = np.mean(tactic_counts) if tactic_counts else 0

        print(f"C{step:04d}        {n_pass:<8d}  {n_semi:<10d}  {n_multi_line:<12d}  {n_multi_tactic:<14d}  {avg_tac:<12.2f}")

    print(f"\nComposition examples from C0150 (multi-tactic successes):")
    for step, records_by_thm in checkpoints:
        if step != 150:
            continue
        count = 0
        for tid in sorted(all_theorems):
            for rec in records_by_thm.get(tid, []):
                if rec["success"]:
                    comp = detect_composition(rec["proof_text"])
                    if comp["n_tactics"] >= 2:
                        tid_disp = tid[:50]
                        tacs = " -> ".join(comp["tactics"][:5])
                        print(f"  {tid_disp:<52}  tactics: {tacs}")
                        count += 1
                        break
            if count >= 15:
                break

    print(f"\nSummary: Is tactic composition emerging naturally?")
    final_step = checkpoints[-1]
    successes = []
    for tid in all_theorems:
        for rec in final_step[1].get(tid, []):
            if rec["success"]:
                successes.append(rec)
    n_pass = len(successes)
    n_comp = sum(1 for rec in successes if detect_composition(rec["proof_text"])["n_tactics"] >= 2)
    pct = 100 * n_comp / n_pass if n_pass > 0 else 0
    print(f"  At final checkpoint: {n_comp}/{n_pass} ({pct:.1f}%) successful proofs use 2+ tactics")
    if pct > 30:
        print(f"  -> Model is learning composition naturally. Level 1-2 curriculum may ACCELERATE but is not strictly required.")
    else:
        print(f"  -> Model relies primarily on single tactics. Level 1-2 curriculum is ESSENTIAL for Tier 4.")
