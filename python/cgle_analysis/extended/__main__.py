"""CLI entry point: python3 -m cgle_analysis.extended --data-dir X --output-dir Y"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ._shared import find_checkpoints, load_checkpoint, theorem_id
from .temperature import analysis_temperature_breadth
from .tactics import analysis_tactic_distribution
from .length_ratio import analysis_length_ratio
from .composition import analysis_tactic_composition
from .regression import analysis_regression
from .sigmoid import analysis_sigmoid_fit


def main():
    parser = argparse.ArgumentParser(
        description="Extended analyses on GRPO Lean-reward training data"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_entries = find_checkpoints(args.data_dir)
    print(f"Found {len(checkpoint_entries)} checkpoints")

    checkpoints = []
    all_theorems_set: set[str] = set()
    for step, chk_dir in checkpoint_entries:
        records = load_checkpoint(chk_dir)
        records_by_thm: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            tid = theorem_id(rec)
            records_by_thm[tid].append(rec)
            all_theorems_set.add(tid)
        n_pass = sum(1 for r in records if r["success"])
        print(f"  C{step:04d}: {len(records)} trajectories, {n_pass} passing")
        checkpoints.append((step, dict(records_by_thm)))

    all_theorems = sorted(all_theorems_set)
    print(f"Total unique theorems: {len(all_theorems)}")

    breadth, depth_ratios = analysis_temperature_breadth(checkpoints, all_theorems)
    tactic_matrix = analysis_tactic_distribution(checkpoints, all_theorems)
    analysis_length_ratio(checkpoints, all_theorems)
    analysis_tactic_composition(checkpoints, all_theorems)
    analysis_regression(checkpoints, all_theorems)
    analysis_sigmoid_fit(checkpoints)

    summary = {
        "depth_ratios": depth_ratios,
        "tactic_matrix": {str(k): v for k, v in tactic_matrix.items()},
    }
    out_path = args.output_dir / "extended_analyses_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nJSON summary saved: {out_path}")
    print("\nAll extended analyses complete.")


if __name__ == "__main__":
    main()
