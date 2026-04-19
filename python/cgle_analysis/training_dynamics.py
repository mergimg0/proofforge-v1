#!/usr/bin/env python3
"""
training_dynamics.py — GRPO Training Checkpoint Analysis

Analyses how model capabilities evolve across GRPO training checkpoints.
Operates on checkpoint directories containing .npz trajectory files.

Analyses performed:
  1. Per-theorem binary matrix (capability acquisition order)
  2. Temperature migration analysis (exploration -> internalized strategy)
  3. Failure mode taxonomy across checkpoints
  4. Per-theorem period tracking (oscillation period before/after learning)
  5. Trajectory length of successful proofs across checkpoints
  6. Pass rate curve with per-tier breakdown

Usage:
  python3 -m cgle_analysis.training_dynamics \
      --data-dir /path/to/grpo_lean_run \
      --theorems-path /path/to/theorems.json \
      --output-dir /path/to/results
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from .extended._shared import (
    find_checkpoints,
    load_checkpoint,
    theorem_id,
    TACTIC_SET,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse GRPO training dynamics across checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to grpo_lean_run directory containing checkpoint_NNN subdirectories",
    )
    parser.add_argument(
        "--theorems-path",
        type=Path,
        default=None,
        help="Path to theorems.json (optional; enables tier-grouped output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON summary is written",
    )
    return parser.parse_args()





# ---------------------------------------------------------------------------
# Failure mode classification
# ---------------------------------------------------------------------------


# TACTIC_KEYWORDS and TACTIC_SET imported from extended._shared


def classify_failure_mode(proof_text: str) -> str:
    """
    Classify a trajectory's proof text into a failure mode category.

    Categories (in priority order):
      REPETITION      -- restates the theorem/lemma header
      COMMENTARY      -- leads with a comment
      TACTIC_ATTEMPT  -- leads with a known tactic keyword
      DEGENERATE      -- very short (<= 3 meaningful tokens)
      OTHER           -- everything else
    """
    text = proof_text.strip()
    if not text:
        return "DEGENERATE"

    lines = text.splitlines()
    first_line = ""
    for ln in lines:
        stripped = ln.strip()
        if stripped:
            first_line = stripped.lower()
            break

    if first_line.startswith("theorem") or first_line.startswith("lemma"):
        return "REPETITION"

    if first_line.startswith("--") or first_line.startswith("#"):
        return "COMMENTARY"

    first_token = re.split(r"[\s(]+", first_line)[0].rstrip(":")
    if first_token in TACTIC_SET:
        return "TACTIC_ATTEMPT"

    if len(text.split()) <= 3:
        return "DEGENERATE"

    return "OTHER"


# ---------------------------------------------------------------------------
# Period computation via PCA autocorrelation
# ---------------------------------------------------------------------------

def compute_period(hidden_states: np.ndarray) -> Optional[float]:
    """
    Estimate the oscillation period of a trajectory via autocorrelation of
    PCA mode 0.

    Steps:
      1. Center the trajectory.
      2. Project onto first PCA component.
      3. Compute normalised autocorrelation.
      4. Find the first peak at lag >= 3 with height > 0.1.

    Returns the lag (period) or None if no qualifying peak found.
    """
    T, D = hidden_states.shape
    if T < 10 or D < 1:
        return None

    centered = hidden_states - hidden_states.mean(axis=0, keepdims=True)
    k = min(1, T, D)
    try:
        pca = PCA(n_components=k)
        mode0 = pca.fit_transform(centered)[:, 0]  # (T,)
    except Exception:
        return None

    mode0 = mode0 - mode0.mean()
    norm = np.dot(mode0, mode0)
    if norm < 1e-12:
        return None

    max_lag = T // 2
    acf = np.array([
        np.dot(mode0[:T - lag], mode0[lag:]) / norm
        for lag in range(max_lag + 1)
    ])

    for lag in range(3, max_lag - 1):
        if acf[lag] > 0.1 and acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1]:
            return float(lag)

    return None


# ---------------------------------------------------------------------------
# Theorems metadata
# ---------------------------------------------------------------------------

def load_theorems(theorems_path: Optional[Path]) -> dict[str, str]:
    """
    Load theorems.json and return a mapping from theorem ID to difficulty tier.

    Expects one of:
      {"theorem_name": {"difficulty": "easy", ...}, ...}
      [{"name": "...", "tier": "...", ...}, ...]
      {"theorem_name": "tier_string", ...}

    Returns {} if path is None or the file cannot be parsed.
    """
    if theorems_path is None:
        return {}
    try:
        with open(theorems_path) as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[warn] Could not load theorems.json: {exc}")
        return {}

    tier_map: dict[str, str] = {}

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("id") or entry.get("statement", "")
                tier = (
                    entry.get("tier")
                    or entry.get("difficulty")
                    or entry.get("level")
                    or "unknown"
                )
                if name:
                    tier_map[str(name).strip()] = str(tier)

    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str):
                tier_map[key.strip()] = val
            elif isinstance(val, dict):
                tier = (
                    val.get("tier")
                    or val.get("difficulty")
                    or val.get("level")
                    or "unknown"
                )
                tier_map[key.strip()] = str(tier)

    return tier_map


# ---------------------------------------------------------------------------
# Core data aggregation
# ---------------------------------------------------------------------------

class CheckpointData:
    """All trajectories for one checkpoint, indexed by theorem ID."""

    def __init__(self, step: int, records: list[dict]) -> None:
        self.step = step
        self.records = records
        self.by_theorem: dict[str, list[dict]] = {}
        for rec in records:
            tid = theorem_id(rec)
            self.by_theorem.setdefault(tid, []).append(rec)

    def solved(self, tid: str) -> bool:
        """True if any temperature/attempt solved this theorem."""
        return any(r["success"] for r in self.by_theorem.get(tid, []))

    def solving_temperatures(self, tid: str) -> list[float]:
        """Temperatures at which this theorem was solved."""
        return [
            r["temperature"]
            for r in self.by_theorem.get(tid, [])
            if r["success"]
        ]

    def mean_success_length(self) -> Optional[float]:
        """Mean token length of passing proofs."""
        lengths = [r["n_tokens"] for r in self.records if r["success"]]
        return float(np.mean(lengths)) if lengths else None

    def pass_rate(self) -> float:
        """Overall fraction of trajectories that passed."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r["success"]) / len(self.records)

    def failure_mode_counts(self) -> dict[str, int]:
        """Count failure mode categories (failing trajectories only)."""
        counts: dict[str, int] = {
            "TACTIC_ATTEMPT": 0,
            "REPETITION": 0,
            "COMMENTARY": 0,
            "OTHER": 0,
            "DEGENERATE": 0,
        }
        for rec in self.records:
            if not rec["success"]:
                cat = classify_failure_mode(rec["proof_text"])
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def mean_period_for_theorem(self, tid: str) -> Optional[float]:
        """Mean oscillation period across all trajectories for this theorem."""
        periods = []
        for rec in self.by_theorem.get(tid, []):
            hs = rec.get("hidden_states")
            if hs is not None:
                p = compute_period(hs)
                if p is not None:
                    periods.append(p)
        return float(np.mean(periods)) if periods else None


# ---------------------------------------------------------------------------
# Analysis 1: Per-theorem binary matrix
# ---------------------------------------------------------------------------

def analysis_binary_matrix(
    checkpoints: list[CheckpointData],
    all_theorems: list[str],
    tier_map: dict[str, str],
) -> None:
    """
    Print a visual matrix: theorems x checkpoints, filled block = solved, dot = unsolved.
    Group rows by difficulty tier.
    """
    steps = [c.step for c in checkpoints]

    n_theorems = len(all_theorems)
    n_chk = len(checkpoints)
    matrix = np.zeros((n_theorems, n_chk), dtype=int)
    for j, chk in enumerate(checkpoints):
        for i, tid in enumerate(all_theorems):
            matrix[i, j] = 1 if chk.solved(tid) else 0

    tier_to_indices: dict[str, list[int]] = {}
    for i, tid in enumerate(all_theorems):
        tier = tier_map.get(tid, "unknown")
        tier_to_indices.setdefault(tier, []).append(i)

    first_solved = np.full(n_theorems, -1, dtype=int)
    for i in range(n_theorems):
        for j in range(n_chk):
            if matrix[i, j] == 1:
                first_solved[i] = j
                break

    print("\n" + "=" * 72)
    print("ANALYSIS 1: Per-theorem capability acquisition matrix")
    print("=" * 72)

    step_labels = [f"C{s:04d}" for s in steps]
    tid_width = min(40, max((len(t) for t in all_theorems), default=10))
    header = f"{'Theorem':<{tid_width}}  {'Tier':<10}  " + "  ".join(step_labels)
    print(header)
    print("-" * len(header))

    tiers_sorted = sorted(tier_to_indices.keys())
    never_solved = []
    for tier in tiers_sorted:
        print(f"\n  [Tier: {tier}]")
        indices = tier_to_indices[tier]
        solved_indices = [(first_solved[i], i) for i in indices if first_solved[i] >= 0]
        unsolved_indices = [i for i in indices if first_solved[i] < 0]
        solved_indices.sort()

        for _, i in solved_indices:
            tid = all_theorems[i]
            row_bits = "  ".join("\u2588" if matrix[i, j] else "\u00b7" for j in range(n_chk))
            tid_display = (tid[:tid_width - 2] + "..") if len(tid) > tid_width else tid
            fs_label = f"(first@C{steps[first_solved[i]]:04d})"
            print(f"  {tid_display:<{tid_width}}  {tier:<10}  {row_bits}  {fs_label}")

        for i in unsolved_indices:
            tid = all_theorems[i]
            row_bits = "  ".join("\u00b7" for _ in range(n_chk))
            tid_display = (tid[:tid_width - 2] + "..") if len(tid) > tid_width else tid
            print(f"  {tid_display:<{tid_width}}  {tier:<10}  {row_bits}  (never)")
            never_solved.append(tid)

    print()
    n_ever = sum(1 for i in range(n_theorems) if first_solved[i] >= 0)
    print(f"Summary: {n_ever}/{n_theorems} theorems solved at some checkpoint")
    print(f"         {len(never_solved)} never solved")

    lost = []
    for i in range(n_theorems):
        row = matrix[i, :]
        if row.max() == 1 and row[-1] == 0:
            lost.append(all_theorems[i])
    if lost:
        print(f"\nCapability regression (solved then lost): {len(lost)} theorems")
        for tid in lost:
            print(f"  - {tid[:60]}")


# ---------------------------------------------------------------------------
# Analysis 2: Temperature migration
# ---------------------------------------------------------------------------

def analysis_temperature_migration(
    checkpoints: list[CheckpointData],
    all_theorems: list[str],
) -> None:
    """
    For each theorem that gets solved, track which temperatures succeed at
    each checkpoint. Identify whether solutions migrate from high-T to low-T.
    """
    print("\n" + "=" * 72)
    print("ANALYSIS 2: Temperature migration analysis")
    print("=" * 72)

    all_temps: set[float] = set()
    for chk in checkpoints:
        for rec in chk.records:
            t = rec["temperature"]
            if t == t:  # not nan
                all_temps.add(round(t, 3))
    sorted_temps = sorted(all_temps)

    if not sorted_temps:
        print("No temperature data found.")
        return

    print(f"Temperatures observed: {sorted_temps}")
    print()

    print(f"{'Checkpoint':<12}  " + "  ".join(f"T={t:.2f}" for t in sorted_temps))
    print("-" * (12 + 8 * len(sorted_temps) + 10))

    for chk in checkpoints:
        counts = {t: 0 for t in sorted_temps}
        for rec in chk.records:
            if rec["success"]:
                t_key = round(rec["temperature"], 3)
                if t_key in counts:
                    counts[t_key] += 1
        row = "  ".join(f"{counts.get(t, 0):6d}" for t in sorted_temps)
        print(f"C{chk.step:04d}        {row}")

    print("\nPer-theorem temperature of first solution:")
    solved_set: set[str] = set()
    for chk in checkpoints:
        for tid in all_theorems:
            if tid not in solved_set and chk.solved(tid):
                temps = chk.solving_temperatures(tid)
                min_t = min(temps) if temps else float("nan")
                max_t = max(temps) if temps else float("nan")
                tid_disp = tid[:50]
                print(f"  C{chk.step:04d}  {tid_disp:<52}  T_min={min_t:.2f}  T_max={max_t:.2f}")
                solved_set.add(tid)

    if len(checkpoints) >= 2:
        early_chk = checkpoints[0]
        late_chk = checkpoints[-1]
        early_temps_success = [
            rec["temperature"] for rec in early_chk.records
            if rec["success"] and rec["temperature"] == rec["temperature"]
        ]
        late_temps_success = [
            rec["temperature"] for rec in late_chk.records
            if rec["success"] and rec["temperature"] == rec["temperature"]
        ]
        if early_temps_success and late_temps_success:
            mu_early = float(np.mean(early_temps_success))
            mu_late = float(np.mean(late_temps_success))
            direction = "down (internalised)" if mu_late < mu_early else "up (still exploring)"
            print(f"\nMean solving temperature: early={mu_early:.3f}, late={mu_late:.3f}  -> {direction}")
        else:
            print("\n(Insufficient success data for temperature migration signal)")


# ---------------------------------------------------------------------------
# Analysis 3: Failure mode taxonomy
# ---------------------------------------------------------------------------

def analysis_failure_modes(checkpoints: list[CheckpointData]) -> None:
    """
    Track proportions of failure mode categories at each checkpoint.
    """
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Failure mode taxonomy across checkpoints")
    print("=" * 72)

    categories = ["TACTIC_ATTEMPT", "REPETITION", "COMMENTARY", "DEGENERATE", "OTHER"]

    header = f"{'Checkpoint':<12}  {'Failing':<8}  " + "  ".join(f"{c:<14}" for c in categories)
    print(header)
    print("-" * len(header))

    for chk in checkpoints:
        counts = chk.failure_mode_counts()
        n_fail = sum(counts.values())
        if n_fail == 0:
            row_parts = ["  0%          " for _ in categories]
        else:
            row_parts = [
                f"{counts.get(c, 0):5d} ({100 * counts.get(c, 0) / n_fail:4.1f}%)"
                for c in categories
            ]
        row = "  ".join(row_parts)
        print(f"C{chk.step:04d}        {n_fail:<8d}  {row}")

    print("\nTrend (first -> last checkpoint):")
    if len(checkpoints) >= 2:
        first_counts = checkpoints[0].failure_mode_counts()
        last_counts = checkpoints[-1].failure_mode_counts()
        n_first = max(sum(first_counts.values()), 1)
        n_last = max(sum(last_counts.values()), 1)
        for cat in categories:
            pct_first = 100 * first_counts.get(cat, 0) / n_first
            pct_last = 100 * last_counts.get(cat, 0) / n_last
            delta = pct_last - pct_first
            arrow = "^" if delta > 1 else ("v" if delta < -1 else "~")
            print(f"  {cat:<16}  {pct_first:5.1f}% -> {pct_last:5.1f}%  {arrow} ({delta:+.1f}pp)")


# ---------------------------------------------------------------------------
# Analysis 4: Per-theorem period tracking
# ---------------------------------------------------------------------------

def analysis_period_tracking(
    checkpoints: list[CheckpointData],
    all_theorems: list[str],
) -> None:
    """
    For theorems that are newly learned at a checkpoint, compare the mean
    oscillation period before and after the learning event.
    """
    print("\n" + "=" * 72)
    print("ANALYSIS 4: Per-theorem period tracking around learning events")
    print("=" * 72)

    first_solved_idx: dict[str, int] = {}
    for j, chk in enumerate(checkpoints):
        for tid in all_theorems:
            if tid not in first_solved_idx and chk.solved(tid):
                first_solved_idx[tid] = j

    if not first_solved_idx:
        print("No theorems solved -- no period transitions to analyse.")
        return

    has_period_data = False
    print(f"{'Theorem':<50}  {'LearntAt':<10}  {'Period_before':<15}  {'Period_after':<15}  {'Change'}")
    print("-" * 110)

    for tid, j in sorted(first_solved_idx.items(), key=lambda x: x[1]):
        before_periods = []
        for k in range(j):
            p = checkpoints[k].mean_period_for_theorem(tid)
            if p is not None:
                before_periods.append(p)

        after_periods = []
        for k in range(j, len(checkpoints)):
            p = checkpoints[k].mean_period_for_theorem(tid)
            if p is not None:
                after_periods.append(p)

        mu_before = float(np.mean(before_periods)) if before_periods else None
        mu_after = float(np.mean(after_periods)) if after_periods else None

        tid_disp = tid[:48]
        step_label = f"C{checkpoints[j].step:04d}"

        if mu_before is not None and mu_after is not None:
            delta = mu_after - mu_before
            change_str = f"{delta:+.2f} ({'shorter' if delta < 0 else 'longer'})"
            print(f"  {tid_disp:<50}  {step_label:<10}  {mu_before:13.2f}  {mu_after:13.2f}  {change_str}")
            has_period_data = True
        else:
            before_str = f"{mu_before:.2f}" if mu_before is not None else "N/A"
            after_str = f"{mu_after:.2f}" if mu_after is not None else "N/A"
            print(f"  {tid_disp:<50}  {step_label:<10}  {before_str:>13}  {after_str:>13}  (insufficient data)")

    if not has_period_data:
        print("\n(No hidden_states data available for period computation)")


# ---------------------------------------------------------------------------
# Analysis 5: Trajectory length of successful proofs
# ---------------------------------------------------------------------------

def analysis_proof_length(checkpoints: list[CheckpointData]) -> None:
    """
    Track mean trajectory length of passing proofs at each checkpoint.
    """
    print("\n" + "=" * 72)
    print("ANALYSIS 5: Trajectory length of successful proofs")
    print("=" * 72)

    print(f"{'Checkpoint':<12}  {'N_pass':<8}  {'Mean_len':<12}  {'Median_len':<12}  {'Min':<8}  {'Max':<8}")
    print("-" * 70)

    lengths_over_time: list[Optional[float]] = []
    for chk in checkpoints:
        pass_lengths = [r["n_tokens"] for r in chk.records if r["success"]]
        if pass_lengths:
            mu = float(np.mean(pass_lengths))
            med = float(np.median(pass_lengths))
            mn = int(min(pass_lengths))
            mx = int(max(pass_lengths))
            print(f"C{chk.step:04d}        {len(pass_lengths):<8d}  {mu:<12.1f}  {med:<12.1f}  {mn:<8d}  {mx:<8d}")
            lengths_over_time.append(mu)
        else:
            print(f"C{chk.step:04d}        {'0':<8}  {'--':<12}  {'--':<12}  {'--':<8}  {'--':<8}")
            lengths_over_time.append(None)

    valid = [(i, v) for i, v in enumerate(lengths_over_time) if v is not None]
    if len(valid) >= 2:
        idxs = np.array([i for i, _ in valid], dtype=float)
        vals = np.array([v for _, v in valid])
        coeffs = np.polyfit(idxs, vals, 1)
        slope = float(coeffs[0])
        direction = "shorter (more efficient)" if slope < 0 else "longer (less efficient)"
        print(f"\nTrend: slope={slope:.2f} tokens/checkpoint -> {direction}")
    else:
        print("\n(Insufficient data for trend analysis)")


# ---------------------------------------------------------------------------
# Analysis 6: Pass rate curve with per-tier breakdown
# ---------------------------------------------------------------------------

def analysis_pass_rate(
    checkpoints: list[CheckpointData],
    all_theorems: list[str],
    tier_map: dict[str, str],
) -> None:
    """
    Overall pass rate at each checkpoint with per-tier breakdown.
    """
    print("\n" + "=" * 72)
    print("ANALYSIS 6: Pass rate curve with per-tier breakdown")
    print("=" * 72)

    tiers_present: set[str] = set()
    for tid in all_theorems:
        tiers_present.add(tier_map.get(tid, "unknown"))
    sorted_tiers = sorted(tiers_present)

    tier_cols = "  ".join(f"T:{t:<8}" for t in sorted_tiers)
    print(f"{'Checkpoint':<12}  {'Overall':<10}  {tier_cols}")
    print("-" * (12 + 12 + 12 * len(sorted_tiers) + 10))

    overall_rates: list[float] = []
    for chk in checkpoints:
        overall = chk.pass_rate()
        overall_rates.append(overall)

        tier_rates: dict[str, str] = {}
        for tier in sorted_tiers:
            tier_theorems = [t for t in all_theorems if tier_map.get(t, "unknown") == tier]
            if not tier_theorems:
                tier_rates[tier] = "  N/A    "
                continue
            n_solved = sum(1 for t in tier_theorems if chk.solved(t))
            pct = 100.0 * n_solved / len(tier_theorems)
            tier_rates[tier] = f"{pct:5.1f}%   "

        tier_row = "  ".join(tier_rates.get(t, "  N/A    ") for t in sorted_tiers)
        print(f"C{chk.step:04d}        {overall * 100:7.2f}%   {tier_row}")

    if len(overall_rates) >= 2:
        print("\nSignificant pass-rate jumps (>= 5pp):")
        any_jump = False
        for i in range(1, len(overall_rates)):
            delta = 100 * (overall_rates[i] - overall_rates[i - 1])
            if abs(delta) >= 5.0:
                direction = "^" if delta > 0 else "v"
                print(
                    f"  C{checkpoints[i-1].step:04d} -> C{checkpoints[i].step:04d}: "
                    f"{direction} {abs(delta):.1f}pp  "
                    f"({100*overall_rates[i-1]:.1f}% -> {100*overall_rates[i]:.1f}%)"
                )
                any_jump = True
        if not any_jump:
            print("  None detected (all changes < 5pp)")


# ---------------------------------------------------------------------------
# JSON summary serialisation
# ---------------------------------------------------------------------------

def build_json_summary(
    checkpoints: list[CheckpointData],
    all_theorems: list[str],
    tier_map: dict[str, str],
) -> dict:
    """Build a JSON-serialisable summary of all checkpoint analyses."""
    summary: dict = {
        "n_checkpoints": len(checkpoints),
        "n_theorems": len(all_theorems),
        "checkpoint_steps": [c.step for c in checkpoints],
        "checkpoints": [],
    }

    for chk in checkpoints:
        entry: dict = {
            "step": chk.step,
            "n_records": len(chk.records),
            "pass_rate": chk.pass_rate(),
            "mean_success_length": chk.mean_success_length(),
            "failure_mode_counts": chk.failure_mode_counts(),
            "solved_theorems": [t for t in all_theorems if chk.solved(t)],
            "tier_pass_rates": {},
        }

        tiers: set[str] = {tier_map.get(t, "unknown") for t in all_theorems}
        for tier in sorted(tiers):
            tier_theorems = [t for t in all_theorems if tier_map.get(t, "unknown") == tier]
            if tier_theorems:
                n_solved = sum(1 for t in tier_theorems if chk.solved(t))
                entry["tier_pass_rates"][tier] = n_solved / len(tier_theorems)

        summary["checkpoints"].append(entry)

    n_t = len(all_theorems)
    n_c = len(checkpoints)
    matrix = [
        [1 if checkpoints[j].solved(all_theorems[i]) else 0 for j in range(n_c)]
        for i in range(n_t)
    ]
    summary["binary_matrix"] = {
        "theorems": all_theorems,
        "steps": [c.step for c in checkpoints],
        "matrix": matrix,
    }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: --data-dir does not exist: {args.data_dir}")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tier_map = load_theorems(args.theorems_path)
    print(f"Loaded {len(tier_map)} theorem tier mappings")

    checkpoint_entries = find_checkpoints(args.data_dir)
    if not checkpoint_entries:
        print(f"ERROR: No checkpoint_NNN directories found in {args.data_dir}")
        raise SystemExit(1)

    print(f"\nFound {len(checkpoint_entries)} checkpoints:")
    for step, path in checkpoint_entries:
        n_npz = len(list(path.glob("*.npz")))
        print(f"  checkpoint_{step:04d}  ({n_npz} .npz files)")

    print("\nLoading trajectories...")
    checkpoints: list[CheckpointData] = []
    for step, chk_dir in checkpoint_entries:
        records = load_checkpoint(chk_dir)
        checkpoints.append(CheckpointData(step, records))
        n_pass = sum(1 for r in records if r["success"])
        print(f"  C{step:04d}: {len(records)} trajectories, {n_pass} passing")

    if not checkpoints:
        print("No data loaded. Exiting.")
        raise SystemExit(1)

    all_theorem_ids_set: set[str] = set()
    for chk in checkpoints:
        all_theorem_ids_set.update(chk.by_theorem.keys())
    all_theorems = sorted(all_theorem_ids_set)
    print(f"\nTotal unique theorems: {len(all_theorems)}")

    for tid in all_theorems:
        if tid not in tier_map:
            tier_map[tid] = "unknown"

    # Run all analyses
    analysis_binary_matrix(checkpoints, all_theorems, tier_map)
    analysis_temperature_migration(checkpoints, all_theorems)
    analysis_failure_modes(checkpoints)
    analysis_period_tracking(checkpoints, all_theorems)
    analysis_proof_length(checkpoints)
    analysis_pass_rate(checkpoints, all_theorems, tier_map)

    # Save JSON summary
    summary = build_json_summary(checkpoints, all_theorems, tier_map)
    out_path = args.output_dir / "training_dynamics_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nJSON summary saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
