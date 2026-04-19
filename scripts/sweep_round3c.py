#!/usr/bin/env python3
"""ProofForge Round 3c Hyperparameter Sweep via Protein GP Optimization.

Loads sweep config from an INI file, runs the Protein sweep loop:
  1. protein.suggest() -> hyperparameter dict
  2. Launch grpo_lean_reward.py with those hyperparameters
  3. Read run_summary.json for final pass_rate and GPU time
  4. protein.observe(hypers, score=pass_rate, cost=gpu_seconds)
  5. protein.early_stop() to kill bad runs
  6. Repeat until max_runs exhausted

Save Pareto front to sweep_results.json.

Usage:
  python3 scripts/sweep_round3c.py --config config/sweep_round3c.ini
  python3 scripts/sweep_round3c.py --config config/sweep_round3c.ini --max-runs 50 --dry-run
"""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import time
from pathlib import Path


def load_sweep_config(config_path: str) -> dict:
    """Load INI sweep config into a flat dict matching Protein's expected format.

    Converts INI sections like [sweep.learning_rate] into nested dicts:
      {"learning_rate": {"distribution": ..., "min": ..., "max": ..., "scale": ...}}
    Top-level [sweep] keys become top-level dict entries.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    config: dict = {}

    # Top-level sweep metadata
    if "sweep" in parser:
        for key, val in parser["sweep"].items():
            config[key] = _auto_cast(val)

    # Per-parameter sections: [sweep.param_name]
    for section in parser.sections():
        if section.startswith("sweep.") and section != "sweep":
            param_name = section[len("sweep."):]
            param_config: dict = {}
            for key, val in parser[section].items():
                param_config[key] = _auto_cast(val)
            config[param_name] = param_config

    return config


def _auto_cast(val: str):
    """Cast INI string values to appropriate Python types."""
    # Booleans
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    # Try int
    try:
        return int(val)
    except ValueError:
        pass
    # Try float (handles scientific notation like 1e-7)
    try:
        return float(val)
    except ValueError:
        pass
    return val


def run_training(hypers: dict, run_idx: int, output_base: str, script_path: str,
                 extra_args: list[str]) -> dict | None:
    """Launch a single training run with the given hyperparameters.

    Returns the parsed run_summary.json or None on failure.
    """
    run_dir = Path(output_base) / f"sweep_run_{run_idx:04d}"

    cmd = [
        sys.executable, script_path,
        "--output-dir", str(run_dir),
        "--enable-efficiency",
        "--enable-controller",
    ]

    # Map sweep hyperparameters to CLI flags
    flag_map = {
        "learning_rate": "--learning-rate",
        "group_size": "--group-size",
        "efficiency_phase_in_step": "--efficiency-phase-in-step",
        "efficiency_phase_in_rate": "--efficiency-phase-in-rate",
        "controller_bias_correction": "--controller-bias-correction",
        "max_new_tokens": "--max-new-tokens",
    }

    for param_name, cli_flag in flag_map.items():
        if param_name in hypers:
            cmd.extend([cli_flag, str(hypers[param_name])])

    cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Run {run_idx}: {' '.join(cmd[:8])}...")
    print(f"  Hyperparameters: { {k: round(v, 6) if isinstance(v, float) else v for k, v in hypers.items()} }")
    print(f"{'='*60}")

    t_start = time.time()
    try:
        result = subprocess.run(cmd, timeout=7200, capture_output=True, text=True)
        elapsed = time.time() - t_start

        if result.returncode != 0:
            print(f"  Run {run_idx} FAILED (exit code {result.returncode})")
            stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
            print(f"  stderr: {stderr_tail}")
            return None

        # Read run_summary.json
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            print(f"  Run {run_idx}: no run_summary.json found")
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        summary["_elapsed_seconds"] = elapsed
        return summary

    except subprocess.TimeoutExpired:
        print(f"  Run {run_idx} TIMED OUT after 7200s")
        return None
    except Exception as e:
        print(f"  Run {run_idx} ERROR: {e}")
        return None


def extract_score(summary: dict) -> float:
    """Extract final pass_rate from run summary."""
    checkpoints = summary.get("checkpoints", [])
    if not checkpoints:
        return 0.0
    return checkpoints[-1].get("pass_rate", 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="ProofForge Round 3c Hyperparameter Sweep"
    )
    parser.add_argument(
        "--config", type=str, default="config/sweep_round3c.ini",
        help="Path to sweep config INI file",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Override max_runs from config",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/workspace/sweep_round3c",
        help="Base output directory for sweep runs",
    )
    parser.add_argument(
        "--training-script", type=str, default="scripts/grpo_lean_reward.py",
        help="Path to the GRPO training script",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print suggested hyperparameters without launching runs",
    )
    parser.add_argument(
        "extra_args", nargs="*",
        help="Extra arguments passed through to the training script",
    )

    args = parser.parse_args()

    # Load sweep configuration
    sweep_config = load_sweep_config(args.config)
    max_runs = args.max_runs or sweep_config.get("max_runs", 100)

    print(f"Sweep config loaded from {args.config}")
    print(f"  Metric: {sweep_config.get('metric', 'pass_rate')}")
    print(f"  Goal: {sweep_config.get('goal', 'maximize')}")
    print(f"  Max runs: {max_runs}")
    print(f"  Max suggestion cost: {sweep_config.get('max_suggestion_cost', 7200)}s")
    print(f"  Parameters: {[k for k in sweep_config if isinstance(sweep_config[k], dict)]}")

    # Create Protein sweep instance
    from proofforge.sweep.protein_sweep import Protein, pareto_points

    protein = Protein(sweep_config)

    # Main sweep loop
    results = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, max_runs + 1):
        # 1. Suggest hyperparameters
        hypers, info = protein.suggest()
        print(f"\nSuggestion {run_idx}/{max_runs}: {hypers}")
        if info:
            print(f"  GP predicted score={info.get('score', '?'):.4f}, "
                  f"cost={info.get('cost', '?'):.1f}")

        if args.dry_run:
            print("  [DRY RUN] Skipping actual training")
            continue

        # 2. Run training
        summary = run_training(
            hypers=hypers,
            run_idx=run_idx,
            output_base=args.output_dir,
            script_path=args.training_script,
            extra_args=args.extra_args,
        )

        # 3. Observe results
        if summary is None:
            protein.observe(hypers, score=0.0, cost=7200, is_failure=True)
            results.append({"run": run_idx, "hypers": hypers, "score": 0.0,
                            "cost": 7200, "failure": True})
        else:
            score = extract_score(summary)
            cost = summary.get("_elapsed_seconds", 7200)
            protein.observe(hypers, score=score, cost=cost)
            results.append({"run": run_idx, "hypers": hypers, "score": score,
                            "cost": cost, "failure": False})
            print(f"  Result: pass_rate={score:.4f}, cost={cost:.1f}s")

        # 4. Save intermediate results
        pareto, _ = pareto_points(protein.success_observations)
        sweep_state = {
            "completed_runs": len(results),
            "max_runs": max_runs,
            "results": results,
            "pareto_front": [
                {"score": p["output"], "cost": p["cost"]}
                for p in pareto
            ],
            "config_path": args.config,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        results_path = Path(args.output_dir) / "sweep_results.json"
        with open(results_path, "w") as f:
            json.dump(sweep_state, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(results)} runs")
    if protein.success_observations:
        pareto, _ = pareto_points(protein.success_observations)
        print(f"Pareto front ({len(pareto)} points):")
        for p in pareto:
            print(f"  score={p['output']:.4f}  cost={p['cost']:.1f}s")
    print(f"Results saved to {Path(args.output_dir) / 'sweep_results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
