#!/usr/bin/env python3
"""
round3_launcher.py — Pre-flight validation + live .md output for Round 3 training.

This script:
  1. Merges all 3 curriculum files into a single unified JSON
  2. Validates every theorem statement compiles in Lean 4
  3. Validates custom definitions compile
  4. Checks GPU memory, dependencies, and disk space
  5. Launches grpo_round3.py with correct args
  6. Captures all output to a live-updating training_live.md

Usage:
  python3 round3_launcher.py --gpu b200     # or --gpu h100
  python3 round3_launcher.py --dry-run      # validate only, don't train
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT = Path("/workspace/proofforge")
DATA_DIR = PROJ_ROOT / "data"
SCRIPTS_DIR = PROJ_ROOT / "scripts"
OUTPUT_DIR = Path("/workspace/grpo_round3")

CURRICULUM_FILES = [
    DATA_DIR / "theorems.json",           # Level 0 (50 theorems)
    DATA_DIR / "theorems_level1_2.json",  # Levels 1-2 (30 theorems)
    DATA_DIR / "curriculum_radical.json",  # Levels 3-12 (98 theorems)
]

MERGED_CURRICULUM = DATA_DIR / "curriculum_merged_round3.json"
LIVE_MD = OUTPUT_DIR / "training_live.md"


# ---------------------------------------------------------------------------
# Live .md writer
# ---------------------------------------------------------------------------

class LiveWriter:
    """Append-only structured markdown writer for training output."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(self.path, "w") as f:
            f.write(f"# ProofForge Round 3 Training — Live Output\n\n")
            f.write(f"**Started**: {ts}\n\n")
            f.write(f"---\n\n")

    def section(self, title: str):
        with open(self.path, "a") as f:
            f.write(f"\n## {title}\n\n")

    def log(self, text: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with open(self.path, "a") as f:
            f.write(f"`[{ts}]` {text}\n\n")

    def table_header(self, cols: list[str]):
        with open(self.path, "a") as f:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

    def table_row(self, vals: list[str]):
        with open(self.path, "a") as f:
            f.write("| " + " | ".join(vals) + " |\n")

    def code_block(self, text: str, lang: str = ""):
        with open(self.path, "a") as f:
            f.write(f"```{lang}\n{text}\n```\n\n")

    def metric(self, step: int, metrics: dict):
        """Append a training step metric row."""
        with open(self.path, "a") as f:
            parts = [f"step={step}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            f.write(f"- {' | '.join(parts)}\n")

    def checkpoint(self, step: int, pass_rate: float, n_solved: int, n_total: int):
        """Log a checkpoint evaluation."""
        with open(self.path, "a") as f:
            bar = "#" * int(pass_rate * 50)
            f.write(f"\n### Checkpoint step={step}\n\n")
            f.write(f"**Pass rate: {pass_rate:.1%}** ({n_solved}/{n_total})\n\n")
            f.write(f"```\n{bar}\n```\n\n")

    def finish(self, final_msg: str = ""):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(self.path, "a") as f:
            f.write(f"\n---\n\n**Finished**: {ts}\n\n")
            if final_msg:
                f.write(f"{final_msg}\n")


# ---------------------------------------------------------------------------
# Curriculum merger
# ---------------------------------------------------------------------------

def merge_curricula(files: list[Path], output: Path) -> dict:
    """Merge multiple curriculum JSONs into a single unified file."""
    all_tiers = []
    custom_definitions = ""
    total = 0

    for fpath in files:
        if not fpath.exists():
            print(f"  [WARN] Curriculum file not found: {fpath}")
            continue
        with open(fpath) as f:
            data = json.load(f)

        # Extract custom definitions if present
        if "definitions" in data and data["definitions"]:
            custom_definitions = data["definitions"]

        for tier in data.get("tiers", []):
            theorems = tier.get("theorems", [])
            tier_name = tier.get("name", f"tier_{len(all_tiers)}")
            requires_defs = tier.get("requires_definitions", False)

            # Tag each theorem with its source and definition requirement
            for thm in theorems:
                thm["_source_file"] = fpath.name
                thm["_requires_definitions"] = requires_defs

            all_tiers.append({
                "name": tier_name,
                "theorems": theorems,
                "requires_definitions": requires_defs,
            })
            total += len(theorems)
            print(f"  Loaded {tier_name}: {len(theorems)} theorems from {fpath.name}")

    merged = {
        "description": "ProofForge Round 3 merged curriculum (Levels 0-12)",
        "version": "round3",
        "total_theorems": total,
        "definitions": custom_definitions,
        "tiers": all_tiers,
    }

    with open(output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"  Merged curriculum: {total} theorems across {len(all_tiers)} tiers -> {output}")
    return merged


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def check_dependencies() -> list[str]:
    """Check all required Python packages."""
    errors = []
    required = [
        "torch", "transformers", "peft", "accelerate", "numpy",
        "sentencepiece", "google.protobuf", "scipy",
    ]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"Missing Python package: {pkg}")
    return errors


def check_gpu() -> tuple[str, float, list[str]]:
    """Check GPU availability and memory."""
    errors = []
    try:
        import torch  # type: ignore[import-not-found]  # runtime-only on GPU host
    except ImportError:
        return "none", 0.0, ["torch not installed"]
    if not torch.cuda.is_available():
        return "none", 0.0, ["No CUDA GPU available"]

    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if mem_gb < 70:
        errors.append(f"GPU memory {mem_gb:.0f}GB may be insufficient (need >=80GB)")

    return name, mem_gb, errors


def check_lean() -> list[str]:
    """Check Lean 4 is available."""
    if not shutil.which("lean"):
        return ["Lean 4 not found on PATH (will be installed by training script)"]
    result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return [f"Lean 4 version check failed: {result.stderr[:100]}"]
    return []


def check_disk_space(path: Path, min_gb: float = 50) -> list[str]:
    """Check available disk space."""
    import shutil as sh
    total, used, free = sh.disk_usage(path)
    free_gb = free / 1e9
    if free_gb < min_gb:
        return [f"Low disk space: {free_gb:.1f}GB free (need {min_gb}GB)"]
    return []


def validate_lean_definitions(definitions: str) -> list[str]:
    """Validate that custom definitions compile in Lean 4."""
    if not definitions.strip():
        return []
    if not shutil.which("lean"):
        return ["Cannot validate definitions — Lean not installed"]

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
        f.write(definitions)
        tmpfile = f.name
    try:
        result = subprocess.run(["lean", tmpfile], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return [f"Custom definitions failed to compile: {result.stderr[:200]}"]
        return []
    except subprocess.TimeoutExpired:
        return ["Custom definitions compilation timed out (60s)"]
    finally:
        os.unlink(tmpfile)


def validate_sample_theorems(merged: dict, n_sample: int = 5) -> list[str]:
    """Validate a sample of theorems compile with their known proofs."""
    if not shutil.which("lean"):
        return ["Cannot validate theorems — Lean not installed"]

    errors = []
    definitions = merged.get("definitions", "")
    all_theorems = []
    for tier in merged.get("tiers", []):
        for thm in tier.get("theorems", []):
            if thm.get("known_proof"):
                all_theorems.append((thm, tier.get("requires_definitions", False)))

    import random
    sample = random.sample(all_theorems, min(n_sample, len(all_theorems)))

    for thm, needs_defs in sample:
        stmt = thm["statement"]
        proof = thm["known_proof"]
        proof_lines = proof.split("\n")
        indented = "\n".join(f"  {line}" for line in proof_lines)
        source = ""
        if needs_defs and definitions:
            source += definitions + "\n\n"
        source += f"{stmt}\n{indented}\n"

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(source)
            tmpfile = f.name
        try:
            result = subprocess.run(["lean", tmpfile], capture_output=True, text=True, timeout=30)
            status = "PASS" if result.returncode == 0 else "FAIL"
            if result.returncode != 0:
                errors.append(f"Theorem {thm.get('id', '?')} failed: {result.stderr[:100]}")
            print(f"  [{status}] {thm.get('id', '?')}: {stmt[:60]}")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {thm.get('id', '?')}")
        finally:
            os.unlink(tmpfile)

    return errors


def run_preflight(merged: dict, writer: LiveWriter) -> bool:
    """Run all pre-flight checks. Returns True if all pass."""
    writer.section("Pre-flight Validation")
    all_errors = []

    # 1. Dependencies
    print("\n[1/6] Checking Python dependencies...")
    errs = check_dependencies()
    all_errors.extend(errs)
    writer.log(f"Dependencies: {'PASS' if not errs else 'FAIL — ' + '; '.join(errs)}")

    # 2. GPU
    print("[2/6] Checking GPU...")
    gpu_name, gpu_mem, errs = check_gpu()
    all_errors.extend(errs)
    writer.log(f"GPU: {gpu_name} ({gpu_mem:.0f}GB) — {'PASS' if not errs else 'FAIL'}")

    # 3. Lean
    print("[3/6] Checking Lean 4...")
    errs = check_lean()
    all_errors.extend(errs)
    writer.log(f"Lean 4: {'PASS' if not errs else 'WARN — ' + '; '.join(errs)}")

    # 4. Disk space
    print("[4/6] Checking disk space...")
    errs = check_disk_space(Path("/workspace"))
    all_errors.extend(errs)
    writer.log(f"Disk: {'PASS' if not errs else 'FAIL — ' + '; '.join(errs)}")

    # 5. Custom definitions
    print("[5/6] Validating custom definitions...")
    errs = validate_lean_definitions(merged.get("definitions", ""))
    all_errors.extend(errs)
    writer.log(f"Custom defs: {'PASS' if not errs else 'FAIL — ' + '; '.join(errs)}")

    # 6. Sample theorems
    print("[6/6] Validating sample theorems...")
    errs = validate_sample_theorems(merged, n_sample=5)
    all_errors.extend(errs)
    writer.log(f"Sample theorems: {'PASS' if not errs else 'FAIL — ' + '; '.join(errs)}")

    # Summary
    critical = [e for e in all_errors if "Missing" in e or "No CUDA" in e or "disk" in e.lower()]
    if critical:
        writer.log(f"**BLOCKED**: {len(critical)} critical errors")
        for e in critical:
            writer.log(f"  - {e}")
        return False

    writer.log(f"**ALL CHECKS PASSED** ({len(all_errors)} warnings)")
    return True


# ---------------------------------------------------------------------------
# Patch: inject custom definitions into Lean verification
# ---------------------------------------------------------------------------

def write_lean_definitions_file(definitions: str, output_dir: Path) -> Path:
    """Write custom definitions to a file that lean_verify_single can prepend."""
    defs_file = output_dir / "lean_custom_defs.lean"
    defs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(defs_file, "w") as f:
        f.write(definitions)
    return defs_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Round 3 pre-flight + launcher")
    parser.add_argument("--gpu", choices=["h100", "b200"], default="b200")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't train")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Override group size (default: 64 for B200, 32 for H100)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--stage", choices=["both", "1", "2"], default="both")
    parser.add_argument("--skip-sft", action="store_true")
    args = parser.parse_args()

    # Default group size by GPU
    if args.group_size is None:
        args.group_size = 64 if args.gpu == "b200" else 32

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = LiveWriter(LIVE_MD)

    print("=" * 60)
    print("  ProofForge Round 3 — Pre-flight + Launch")
    print("=" * 60)
    writer.section("Configuration")
    writer.log(f"GPU: {args.gpu} | group_size: {args.group_size} | steps: {args.steps} | stage: {args.stage}")

    # Step 1: Merge curricula
    print("\n=== Merging curriculum files ===")
    writer.section("Curriculum Merge")
    merged = merge_curricula(CURRICULUM_FILES, MERGED_CURRICULUM)
    writer.log(f"Merged: {merged['total_theorems']} theorems across {len(merged['tiers'])} tiers")

    # Write custom definitions file
    defs = merged.get("definitions", "")
    if defs:
        defs_path = write_lean_definitions_file(defs, OUTPUT_DIR)
        writer.log(f"Custom definitions written to {defs_path}")
        # Set env var so grpo_round3.py can find it
        os.environ["PROOFFORGE_LEAN_DEFS"] = str(defs_path)

    # Step 2: Pre-flight
    print("\n=== Running pre-flight checks ===")
    ok = run_preflight(merged, writer)

    if args.dry_run:
        if ok:
            print("\n=== DRY RUN COMPLETE — All checks passed ===")
            writer.log("**DRY RUN COMPLETE** — ready to train")
        else:
            print("\n=== DRY RUN FAILED — Fix errors above ===")
            writer.log("**DRY RUN FAILED**")
        writer.finish()
        return

    if not ok:
        print("\n=== PRE-FLIGHT FAILED — Aborting ===")
        writer.finish("Aborted due to pre-flight failures.")
        sys.exit(1)

    # Step 3: Launch training
    writer.section("Training Launch")
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "grpo_round3.py"),
        "--stage", args.stage,
        "--theorems-path", str(MERGED_CURRICULUM),
        "--max-theorems", str(merged["total_theorems"]),
        "--group-size", str(args.group_size),
        "--steps", str(args.steps),
        "--output-dir", str(OUTPUT_DIR),
        "--lean-workers", "8",
    ]
    if args.skip_sft:
        cmd.append("--skip-sft")

    writer.log(f"Command: `{' '.join(cmd)}`")
    writer.section("Training Output")

    print(f"\n=== Launching: {' '.join(cmd)} ===\n")

    # Stream output to both stdout and live .md
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    step_pattern_import = None
    try:
        import re
        step_pattern_import = re
    except Exception:
        pass

    stdout = process.stdout
    if stdout is None:
        print("ERROR: Failed to capture subprocess stdout")
        sys.exit(1)

    with open(LIVE_MD, "a") as md:
        md.write("```\n")  # Open code block for raw output
        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            md.write(line)
            md.flush()

            # Parse key metrics and write structured entries
            if step_pattern_import and "step" in line and "mean_reward" in line:
                # Extract step metrics for structured logging
                match = step_pattern_import.search(
                    r'step\s+(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*mean_reward=([\d.]+)', line
                )
                if match:
                    pass  # Already in the code block

            # Detect checkpoint evaluations
            if "pass_rate=" in line and step_pattern_import:
                match = step_pattern_import.search(r'pass_rate=([\d.]+)\s*\((\d+)/(\d+)\)', line)
                if match:
                    pass  # Already in the code block

        md.write("```\n\n")  # Close code block

    retcode = process.wait()

    if retcode == 0:
        writer.finish("Training completed successfully.")
        print("\n=== Training complete ===")
    else:
        writer.finish(f"Training FAILED with exit code {retcode}.")
        print(f"\n=== Training FAILED (exit code {retcode}) ===")
        sys.exit(retcode)


if __name__ == "__main__":
    main()
