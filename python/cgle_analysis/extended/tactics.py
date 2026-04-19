"""Analysis 6+11: Tactic extraction and infrastructure usage."""

import re
from collections import defaultdict
from typing import Optional

from ._shared import TACTIC_KEYWORDS, TACTIC_SET


def extract_first_tactic(proof_text: str) -> Optional[str]:
    text = proof_text.strip()
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("--") or stripped.startswith("#"):
            continue
        if stripped.startswith("theorem") or stripped.startswith("lemma"):
            continue
        first_token = re.split(r"[\s(;{]+", stripped)[0].rstrip(":")
        if first_token in TACTIC_SET:
            return first_token
        for tac in TACTIC_KEYWORDS:
            if stripped.startswith(tac):
                return tac
        break
    return None


def analysis_tactic_distribution(checkpoints, all_theorems):
    print("\n" + "=" * 72)
    print("ANALYSIS 6+11: Tactic extraction and infrastructure usage")
    print("=" * 72)

    tactic_matrix = {}
    all_tactics_seen: set[str] = set()

    for step, records_by_thm in checkpoints:
        tactic_counts: dict[str, int] = defaultdict(int)
        for tid in all_theorems:
            for rec in records_by_thm.get(tid, []):
                if rec["success"]:
                    tac = extract_first_tactic(rec["proof_text"])
                    if tac:
                        tactic_counts[tac] += 1
                        all_tactics_seen.add(tac)
                    else:
                        tactic_counts["<unknown>"] += 1
        tactic_matrix[step] = dict(tactic_counts)

    sorted_tactics = sorted(all_tactics_seen)
    col_width = max(8, max((len(t) for t in sorted_tactics), default=8))

    print(f"\nTactic x Checkpoint matrix (count of successful proofs using each first tactic):")
    header = f"{'Checkpoint':<12}  " + "  ".join(f"{t:<{col_width}}" for t in sorted_tactics) + f"  {'<unknown>':<{col_width}}"
    print(header)
    print("-" * len(header))

    for step, records_by_thm in checkpoints:
        counts = tactic_matrix[step]
        row = "  ".join(f"{counts.get(t, 0):<{col_width}d}" for t in sorted_tactics)
        unk = counts.get("<unknown>", 0)
        print(f"C{step:04d}        {row}  {unk:<{col_width}d}")

    print(f"\nDominant tactic per checkpoint:")
    for step, records_by_thm in checkpoints:
        counts = tactic_matrix[step]
        if counts:
            dominant = max(counts.items(), key=lambda x: x[1])
            total = sum(counts.values())
            pct = 100 * dominant[1] / total if total > 0 else 0
            print(f"  C{step:04d}: {dominant[0]} ({dominant[1]}/{total}, {pct:.1f}%)")

    return tactic_matrix
