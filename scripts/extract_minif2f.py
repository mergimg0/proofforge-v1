#!/usr/bin/env python3
"""Extract miniF2F theorem statements into ProofForge dataset format."""

import json
import re
from pathlib import Path

MINIF2F_DIR = Path.home() / "projects" / "miniF2F-lean4" / "MiniF2F"
OUTPUT = Path.home() / "projects" / "proofforge" / "data" / "minif2f_theorems.json"


def extract_theorem(filepath: Path) -> dict | None:
    """Extract the theorem statement from a miniF2F .lean file."""
    content = filepath.read_text()

    # Find the theorem line(s) — they end with `:= by sorry`
    # The theorem may span multiple lines
    match = re.search(
        r'(theorem\s+\S+.*?):=\s*by\s*sorry',
        content, re.DOTALL
    )
    if not match:
        return None

    statement = match.group(1).strip()
    # Clean up whitespace
    statement = re.sub(r'\s+', ' ', statement)

    # Extract theorem name
    name_match = re.match(r'theorem\s+(\S+)', statement)
    name = name_match.group(1) if name_match else filepath.stem

    # Extract imports and opens (needed for proof checking)
    imports = [line.strip() for line in content.split('\n')
               if line.strip().startswith('import ') or line.strip().startswith('open ')]
    set_options = [line.strip() for line in content.split('\n')
                   if line.strip().startswith('set_option')]

    # Determine difficulty from source
    difficulty = 2  # default
    if 'aime' in name.lower() or 'imo' in name.lower():
        difficulty = 4
    elif 'amc' in name.lower():
        difficulty = 3
    elif 'algebra' in name.lower() or 'number_theory' in name.lower():
        difficulty = 2

    return {
        "id": name,
        "statement": statement + " := by",
        "imports": imports,
        "set_options": set_options,
        "difficulty": difficulty,
        "source_file": str(filepath.relative_to(MINIF2F_DIR.parent)),
        "split": "valid" if "/Valid/" in str(filepath) else "test",
    }


def main():
    valid_dir = MINIF2F_DIR / "Valid"
    test_dir = MINIF2F_DIR / "Test"

    valid_theorems = []
    test_theorems = []

    for f in sorted(valid_dir.glob("*.lean")):
        t = extract_theorem(f)
        if t:
            valid_theorems.append(t)

    for f in sorted(test_dir.glob("*.lean")):
        t = extract_theorem(f)
        if t:
            test_theorems.append(t)

    dataset = {
        "description": "miniF2F-lean4 theorem statements for ProofForge evaluation",
        "version": "0.1.0",
        "source": "https://github.com/yangky11/miniF2F-lean4",
        "note": "These theorems require Mathlib imports. Use Kimina Lean Server for checking.",
        "stats": {
            "validation": len(valid_theorems),
            "test": len(test_theorems),
            "total": len(valid_theorems) + len(test_theorems),
        },
        "validation": valid_theorems,
        "test": test_theorems,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Extracted {len(valid_theorems)} validation + {len(test_theorems)} test theorems")
    print(f"Output: {OUTPUT}")

    # Show difficulty distribution
    for split_name, theorems in [("validation", valid_theorems), ("test", test_theorems)]:
        by_diff = {}
        for t in theorems:
            by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1
        print(f"  {split_name}: {dict(sorted(by_diff.items()))}")


if __name__ == "__main__":
    main()
