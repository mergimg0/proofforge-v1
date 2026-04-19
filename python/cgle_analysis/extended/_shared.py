"""Shared infrastructure for extended analyses."""

import re
from pathlib import Path
from typing import Optional

import numpy as np

MAX_RECORDS_PER_CHECKPOINT = 50_000

TACTIC_KEYWORDS = [
    "simp", "rfl", "decide", "omega", "intro", "exact", "constructor",
    "trivial", "tauto", "aesop", "norm_num", "ring", "cases", "induction",
    "apply", "have", "unfold", "rw", "linarith", "by_cases", "left", "right",
    "contradiction", "absurd", "funext", "ext", "congr", "calc", "suffices",
    "obtain", "rcases", "rintro", "push_neg", "by_contra", "exfalso",
]

TACTIC_SET = set(TACTIC_KEYWORDS)


def decode_proof_text(raw: str) -> str:
    return raw.replace("\u010a", "\n").replace("\u0120", " ")


def find_checkpoints(data_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^checkpoint_(\d+)$")
    found = []
    for entry in data_dir.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                found.append((int(m.group(1)), entry))
    found.sort(key=lambda x: x[0])
    return found


def load_npz_file(fpath: Path) -> Optional[dict]:
    try:
        data = np.load(fpath, allow_pickle=True)  # own trajectory files
    except Exception:
        return None
    rec: dict = {"_path": fpath}
    if "success" in data:
        val = data["success"]
        rec["success"] = bool(val.item() if hasattr(val, "item") else val)
    else:
        rec["success"] = False
    if "temperature" in data:
        val = data["temperature"]
        rec["temperature"] = float(val.item() if hasattr(val, "item") else val)
    else:
        rec["temperature"] = float("nan")
    if "statement" in data:
        val = data["statement"]
        rec["statement"] = str(val.item() if hasattr(val, "item") else val)
    else:
        rec["statement"] = fpath.stem
    if "proof_text" in data:
        val = data["proof_text"]
        rec["proof_text"] = decode_proof_text(str(val.item() if hasattr(val, "item") else val))
    else:
        rec["proof_text"] = ""
    if "tokens" in data:
        rec["n_tokens"] = int(len(data["tokens"]))
    elif "hidden_states" in data and data["hidden_states"].ndim == 2:
        rec["n_tokens"] = int(data["hidden_states"].shape[0])
    else:
        rec["n_tokens"] = len(rec["proof_text"].split())
    return rec


def load_checkpoint(chk_dir: Path) -> list[dict]:
    records = []
    for fpath in sorted(chk_dir.glob("*.npz")):
        if len(records) >= MAX_RECORDS_PER_CHECKPOINT:
            print(f"  [warn] Capped at {MAX_RECORDS_PER_CHECKPOINT} records for {chk_dir.name}")
            break
        rec = load_npz_file(fpath)
        if rec is not None:
            records.append(rec)
    return records


def theorem_id(record: dict) -> str:
    stmt = record.get("statement", "")
    if stmt:
        return stmt.strip()
    return record["_path"].stem
