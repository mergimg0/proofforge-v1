"""
analyze.py — CGLE Analysis Pipeline for Transformer Hidden State Trajectories

Runs a multi-step analysis on hidden state trajectories collected from a
language model generating Lean 4 proofs. Steps:

  0. Summarize dataset
  1. Verify proofs (Lean or heuristic)
  2. Cluster separation (success vs failure)
  3. Intrinsic dimensionality estimation
  4. Per-trajectory TDA (ripser)
  5. Population-level TDA stability across PCA reduction levels
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import ripser


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze transformer hidden state trajectories for proof generation."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing .npz trajectory files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where plots and JSON results are written.",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip Lean proof verification; use heuristic labeling instead.",
    )
    p.add_argument(
        "--steps",
        type=str,
        default="0,2,3,4,5",
        help="Comma-separated list of analysis steps to run (e.g. '0,2,3,4,5').",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to load (None = all).",
    )
    p.add_argument(
        "--lean-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each Lean verification call.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def safe_load_npz(path: str) -> dict:
    """Load a .npz file produced by the collection pipeline.

    Uses allow_pickle=True because our files may contain object arrays
    (e.g. variable-length token lists).  These are our own files so
    pickle trust is acceptable.
    """
    data = np.load(path, allow_pickle=True)
    return dict(data)


def load_trajectories(data_dir: str, max_n: int | None) -> list[dict]:
    """Load all .npz trajectory files from *data_dir*.

    Each returned dict has keys:
        file          : str   — source filename
        hidden_states : ndarray (T, D)  — hidden states over T generation steps
        tokens        : list[str] | ndarray — generated tokens
        temperature   : float — sampling temperature used
        proof_text    : str   — raw proof text
        theorem_id    : str   — unique identifier for the theorem
        statement     : str   — natural-language theorem statement
        success       : bool  — whether the proof was accepted (initially from metadata)
        n_steps       : int   — number of generation steps  T
        d_hidden      : int   — hidden dimension D
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))

    if max_n is not None:
        npz_files = npz_files[:max_n]

    trajectories = []
    for fp in npz_files:
        raw = safe_load_npz(str(fp))

        hs = np.array(raw.get("hidden_states", np.empty((0, 1))), dtype=np.float32)
        if hs.ndim == 1:
            # single-step edge case — reshape to (1, D)
            hs = hs.reshape(1, -1)

        tokens = raw.get("tokens", np.array([])).tolist()
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        temperature = float(raw.get("temperature", np.nan))
        proof_text = str(raw.get("proof_text", ""))
        theorem_id = str(raw.get("theorem_id", fp.stem))
        statement = str(raw.get("statement", ""))
        success_raw = raw.get("success", False)
        # handle numpy scalar
        success = bool(success_raw.item() if hasattr(success_raw, "item") else success_raw)

        n_steps, d_hidden = hs.shape

        trajectories.append(
            {
                "file": str(fp),
                "hidden_states": hs,
                "tokens": tokens,
                "temperature": temperature,
                "proof_text": proof_text,
                "theorem_id": theorem_id,
                "statement": statement,
                "success": success,
                "n_steps": n_steps,
                "d_hidden": d_hidden,
            }
        )

    print(f"[load] Loaded {len(trajectories)} trajectories from {data_dir}")
    return trajectories


# ---------------------------------------------------------------------------
# Step 0 — Summary statistics
# ---------------------------------------------------------------------------

def step0_summarize(trajectories: list[dict], output_dir: str) -> dict:
    """Print and save summary statistics for the loaded trajectories."""
    n = len(trajectories)
    n_success = sum(t["success"] for t in trajectories)
    n_fail = n - n_success

    d_hiddens = [t["d_hidden"] for t in trajectories]
    n_steps_list = [t["n_steps"] for t in trajectories]
    temps = [t["temperature"] for t in trajectories if not np.isnan(t["temperature"])]

    summary = {
        "n_trajectories": n,
        "n_success": n_success,
        "n_fail": n_fail,
        "d_hidden_min": int(min(d_hiddens)) if d_hiddens else None,
        "d_hidden_max": int(max(d_hiddens)) if d_hiddens else None,
        "n_steps_mean": float(np.mean(n_steps_list)) if n_steps_list else None,
        "n_steps_min": int(min(n_steps_list)) if n_steps_list else None,
        "n_steps_max": int(max(n_steps_list)) if n_steps_list else None,
        "temperature_mean": float(np.mean(temps)) if temps else None,
    }

    print("\n=== Step 0: Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = Path(output_dir) / "step0_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Step 1 — Proof verification
# ---------------------------------------------------------------------------

def _verify_with_lean(proof_text: str, theorem_id: str, timeout: int) -> bool | None:
    """Attempt to verify a single proof by calling the Lean 4 executable.

    Returns True (verified), False (rejected), or None (error/timeout).
    """
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".lean", mode="w", delete=False
        ) as tmp:
            tmp.write(proof_text)
            tmp_path = tmp.name

        result = subprocess.run(
            ["lean", "--no-info", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            return True
        # Check for error keywords in output
        combined = (result.stdout + result.stderr).lower()
        if "error" in combined or "sorry" in combined:
            return False
        return True  # no errors → treat as success
    except FileNotFoundError:
        # Lean not installed
        return None
    except subprocess.TimeoutExpired:
        print(f"    [lean] Timeout for {theorem_id}")
        return None
    except Exception as e:
        print(f"    [lean] Error for {theorem_id}: {e}")
        return None


def _heuristic_label(proof_text: str) -> bool:
    """Simple heuristic: flag proofs containing 'sorry' as failures."""
    text_lower = proof_text.lower()
    if "sorry" in text_lower:
        return False
    # A proof that ends with #check or is very short is suspicious
    if len(proof_text.strip()) < 20:
        return False
    return True


def step1_verify(
    trajectories: list[dict],
    output_dir: str,
    skip_verify: bool,
    lean_timeout: int,
) -> dict:
    """Verify (or heuristically label) each proof and update trajectory dicts."""
    print("\n=== Step 1: Proof Verification ===")
    results = []
    lean_available = None  # determined on first call

    for t in trajectories:
        if skip_verify:
            verified = _heuristic_label(t["proof_text"])
            method = "heuristic"
        else:
            lean_result = _verify_with_lean(
                t["proof_text"], t["theorem_id"], lean_timeout
            )
            if lean_result is None:
                if lean_available is None:
                    lean_available = False
                    print("  [lean] Lean not available — falling back to heuristic labeling.")
                verified = _heuristic_label(t["proof_text"])
                method = "heuristic_fallback"
            else:
                lean_available = True
                verified = lean_result
                method = "lean"

        t["success_verified"] = verified
        results.append(
            {
                "theorem_id": t["theorem_id"],
                "original_label": t["success"],
                "verified": verified,
                "method": method,
            }
        )

    n_verified_success = sum(r["verified"] for r in results)
    print(
        f"  Verified successes: {n_verified_success}/{len(results)}"
    )

    out_path = Path(output_dir) / "step1_verification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return {"n_verified": len(results), "n_success": n_verified_success}


# ---------------------------------------------------------------------------
# Step 2 — Cluster separation (success vs failure)
# ---------------------------------------------------------------------------

def step2_cluster_separation(trajectories: list[dict], output_dir: str) -> dict:
    """Check whether successful and failed proofs are geometrically separable.

    Uses the mean hidden state as a compact representation per trajectory,
    then evaluates logistic regression cross-validation accuracy and
    produces a PCA 3D scatter plot.
    """
    print("\n=== Step 2: Cluster Separation ===")

    # Use verified label if available, otherwise original
    labels = []
    mean_hs = []
    for t in trajectories:
        label = t.get("success_verified", t["success"])
        labels.append(int(label))
        mean_hs.append(t["hidden_states"].mean(axis=0))

    labels = np.array(labels)
    X = np.array(mean_hs, dtype=np.float32)

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    results: dict = {
        "n_success": n_pos,
        "n_fail": n_neg,
        "separable": False,
        "cv_accuracy": None,
        "cv_std": None,
        "pca_variance_explained_3": None,
    }

    if n_pos < 3 or n_neg < 3:
        print(
            f"  Skipping: need >= 3 of each class (have {n_pos} success, {n_neg} fail)."
        )
        out_path = Path(output_dir) / "step2_cluster_separation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression CV
    clf = LogisticRegression(max_iter=1000, random_state=42)
    n_folds = min(5, n_pos, n_neg)
    cv_scores = cross_val_score(clf, X_scaled, labels, cv=n_folds, scoring="accuracy")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    results["cv_accuracy"] = cv_mean
    results["cv_std"] = cv_std
    results["separable"] = cv_mean > 0.7

    print(f"  Logistic regression CV accuracy: {cv_mean:.3f} ± {cv_std:.3f} (n_folds={n_folds})")
    print(f"  Geometrically separable: {results['separable']}")

    # PCA 3D scatter
    n_components = min(3, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = float(pca.explained_variance_ratio_.sum())
    results["pca_variance_explained_3"] = var_exp

    fig = plt.figure(figsize=(8, 6))
    if n_components >= 3:
        ax = fig.add_subplot(111, projection="3d")
        for lab, color, marker, lbl in [
            (1, "steelblue", "o", "success"),
            (0, "tomato", "x", "fail"),
        ]:
            mask = labels == lab
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                X_pca[mask, 2],
                c=color,
                marker=marker,
                label=lbl,
                alpha=0.7,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        ax = fig.add_subplot(111)
        for lab, color, marker, lbl in [
            (1, "steelblue", "o", "success"),
            (0, "tomato", "x", "fail"),
        ]:
            mask = labels == lab
            if n_components == 2:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker, label=lbl, alpha=0.7)
            else:
                ax.scatter(X_pca[mask, 0], np.zeros(mask.sum()), c=color, marker=marker, label=lbl, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if n_components >= 2 else "")

    ax.legend()
    ax.set_title(
        f"PCA of mean hidden states\n(CV acc={cv_mean:.2f}, var_exp={var_exp:.2f})"
    )
    plot_path = Path(output_dir) / "step2_pca_scatter.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    out_path = Path(output_dir) / "step2_cluster_separation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Step 3 — Intrinsic dimensionality
# ---------------------------------------------------------------------------

def _two_nn_estimator(X: np.ndarray) -> float:
    """Estimate intrinsic dimension using the 2NN estimator (Facco et al 2017).

    For each point, compute r1 = distance to nearest neighbour and
    r2 = distance to second nearest neighbour.  Then:

        mu_i = r2_i / r1_i
        d = N / sum_i(log(mu_i))   (MLE)

    Returns the estimated intrinsic dimension.
    """
    n = X.shape[0]
    if n < 3:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is distance to self (= 0), skip it
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Avoid division by zero
    valid = r1 > 0
    if valid.sum() < 2:
        return float("nan")

    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    d_hat = valid.sum() / log_mu.sum()
    return float(d_hat)


def step3_intrinsic_dim(trajectories: list[dict], output_dir: str) -> dict:
    """Estimate the intrinsic dimensionality of the hidden state manifold.

    Uses:
    - Per-trajectory 2NN estimator on the hidden state sequence
    - PCA cumulative variance on pooled mean hidden states to find 90/95/99% thresholds
    - Eigenvalue spectrum plot
    """
    print("\n=== Step 3: Intrinsic Dimensionality ===")

    # ---- Per-trajectory 2NN ----
    d_int_per_traj = []
    for t in trajectories:
        hs = t["hidden_states"]
        if hs.shape[0] < 3:
            continue
        d_est = _two_nn_estimator(hs)
        d_int_per_traj.append(d_est)

    d_int_mean = float(np.nanmean(d_int_per_traj)) if d_int_per_traj else float("nan")
    d_int_median = float(np.nanmedian(d_int_per_traj)) if d_int_per_traj else float("nan")
    print(f"  2NN intrinsic dim (per-trajectory): mean={d_int_mean:.2f}, median={d_int_median:.2f}")

    # ---- PCA on pooled mean hidden states ----
    mean_hs = np.array([t["hidden_states"].mean(axis=0) for t in trajectories], dtype=np.float32)
    n_samples, d_hidden = mean_hs.shape

    n_components_pca = min(n_samples, d_hidden, 200)
    pca = PCA(n_components=n_components_pca)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mean_hs)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    def dims_for_variance(threshold: float) -> int:
        idx = np.searchsorted(cumvar, threshold)
        return int(min(idx + 1, n_components_pca))

    pca_90 = dims_for_variance(0.90)
    pca_95 = dims_for_variance(0.95)
    pca_99 = dims_for_variance(0.99)

    print(f"  PCA dims for 90% variance: {pca_90}")
    print(f"  PCA dims for 95% variance: {pca_95}")
    print(f"  PCA dims for 99% variance: {pca_99}")

    # ---- Eigenvalue spectrum plot ----
    n_show = min(50, n_components_pca)
    eigenvalues = pca.explained_variance_[:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(np.arange(1, n_show + 1), eigenvalues, "b.-")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (log scale)")
    axes[0].set_title("Eigenvalue Spectrum")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, len(cumvar) + 1), cumvar, "r-")
    axes[1].axhline(0.90, color="gray", linestyle="--", alpha=0.7, label="90%")
    axes[1].axhline(0.95, color="gray", linestyle="-.", alpha=0.7, label="95%")
    axes[1].axhline(0.99, color="gray", linestyle=":", alpha=0.7, label="99%")
    axes[1].axvline(pca_95, color="steelblue", linestyle="--", alpha=0.7, label=f"d={pca_95} for 95%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance Explained")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Intrinsic Dim: 2NN≈{d_int_median:.1f}, PCA-95%={pca_95}",
        fontsize=12,
    )
    plot_path = Path(output_dir) / "step3_intrinsic_dim.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    results = {
        "d_int_2nn_mean": d_int_mean,
        "d_int_2nn_median": d_int_median,
        "n_trajectories_used": len(d_int_per_traj),
        "pca_dims_90pct": pca_90,
        "pca_dims_95pct": pca_95,
        "pca_dims_99pct": pca_99,
        "d_hidden": int(d_hidden),
    }
    out_path = Path(output_dir) / "step3_intrinsic_dim.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Step 4 — Per-trajectory TDA
# ---------------------------------------------------------------------------

def _max_h1_lifetime(diagrams: list[np.ndarray]) -> float:
    """Return the maximum H1 persistence lifetime from ripser diagrams."""
    if len(diagrams) < 2:
        return 0.0
    h1 = diagrams[1]  # H1 diagram: shape (n, 2)
    if h1.shape[0] == 0:
        return 0.0
    lifetimes = h1[:, 1] - h1[:, 0]
    # Remove infinite bars
    finite_mask = np.isfinite(lifetimes)
    if finite_mask.sum() == 0:
        return 0.0
    return float(lifetimes[finite_mask].max())


def _null_random_walk(hs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a null trajectory via random walk with same step norms."""
    steps = np.diff(hs, axis=0)  # (T-1, D)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)  # (T-1, 1)
    random_directions = rng.standard_normal(steps.shape)
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True) + 1e-12
    random_steps = random_directions * norms
    null_hs = np.vstack([hs[0:1], hs[0:1] + np.cumsum(random_steps, axis=0)])
    return null_hs.astype(np.float32)


def _null_shuffled(hs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a null trajectory by shuffling the time ordering."""
    idx = rng.permutation(hs.shape[0])
    return hs[idx]


def step4_individual_tda(
    trajectories: list[dict],
    output_dir: str,
    n_samples: int = 10,
) -> dict:
    """Run TDA (ripser, maxdim=1) on individual trajectories.

    Compares maximum H1 lifetime of real trajectory against two null models:
    1. Random walk with the same step norms
    2. Shuffled time ordering

    Plots lifetime distributions.
    """
    print(f"\n=== Step 4: Individual TDA (n_samples={n_samples}) ===")

    rng = np.random.default_rng(seed=42)
    sample_indices = list(range(min(n_samples, len(trajectories))))

    real_lifetimes = []
    null_rw_lifetimes = []
    null_shuffle_lifetimes = []
    per_traj_results = []

    for idx in sample_indices:
        t = trajectories[idx]
        hs = t["hidden_states"]

        if hs.shape[0] < 4:
            print(f"  Skipping traj {idx}: too few steps ({hs.shape[0]})")
            continue

        # Standardise each trajectory independently to make distances comparable
        scaler = StandardScaler()
        hs_scaled = scaler.fit_transform(hs)

        # Real trajectory
        rips_real = ripser.ripser(hs_scaled, maxdim=1)
        real_lt = _max_h1_lifetime(rips_real["dgms"])

        # Null 1: random walk
        null_rw = _null_random_walk(hs_scaled, rng)
        rips_rw = ripser.ripser(null_rw, maxdim=1)
        rw_lt = _max_h1_lifetime(rips_rw["dgms"])

        # Null 2: shuffled
        null_sh = _null_shuffled(hs_scaled, rng)
        rips_sh = ripser.ripser(null_sh, maxdim=1)
        sh_lt = _max_h1_lifetime(rips_sh["dgms"])

        real_lifetimes.append(real_lt)
        null_rw_lifetimes.append(rw_lt)
        null_shuffle_lifetimes.append(sh_lt)

        per_traj_results.append(
            {
                "trajectory_idx": idx,
                "theorem_id": t["theorem_id"],
                "n_steps": t["n_steps"],
                "real_h1_max_lifetime": real_lt,
                "null_rw_h1_max_lifetime": rw_lt,
                "null_shuffle_h1_max_lifetime": sh_lt,
            }
        )

        print(
            f"  traj {idx}: real={real_lt:.4f}, null_rw={rw_lt:.4f}, null_shuffle={sh_lt:.4f}"
        )

    # Plot lifetime distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        0,
        max(
            max(real_lifetimes) if real_lifetimes else 1,
            max(null_rw_lifetimes) if null_rw_lifetimes else 1,
            max(null_shuffle_lifetimes) if null_shuffle_lifetimes else 1,
        )
        * 1.1,
        20,
    )
    ax.hist(real_lifetimes, bins=bins, alpha=0.6, label="Real trajectory", color="steelblue")
    ax.hist(null_rw_lifetimes, bins=bins, alpha=0.6, label="Null: random walk", color="tomato")
    ax.hist(null_shuffle_lifetimes, bins=bins, alpha=0.6, label="Null: shuffled", color="goldenrod")
    ax.set_xlabel("Max H1 Lifetime")
    ax.set_ylabel("Count")
    ax.set_title("H1 Lifetime: Real vs Null Models (per-trajectory TDA)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = Path(output_dir) / "step4_tda_lifetimes.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    aggregate = {
        "n_sampled": len(per_traj_results),
        "real_h1_mean": float(np.mean(real_lifetimes)) if real_lifetimes else None,
        "null_rw_h1_mean": float(np.mean(null_rw_lifetimes)) if null_rw_lifetimes else None,
        "null_shuffle_h1_mean": float(np.mean(null_shuffle_lifetimes)) if null_shuffle_lifetimes else None,
        "per_trajectory": per_traj_results,
    }

    out_path = Path(output_dir) / "step4_individual_tda.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Saved: {out_path}")

    return aggregate


# ---------------------------------------------------------------------------
# Step 5 — Population TDA
# ---------------------------------------------------------------------------

def _count_significant_features(diagrams: list[np.ndarray], dim: int, threshold_factor: float = 2.0) -> int:
    """Count H_dim features with lifetime > threshold_factor * median lifetime."""
    if len(diagrams) <= dim:
        return 0
    dgm = diagrams[dim]
    if dgm.shape[0] == 0:
        return 0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    finite = lifetimes[np.isfinite(lifetimes)]
    if len(finite) == 0:
        return 0
    med = np.median(finite)
    if med == 0:
        return int((finite > 0).sum())
    return int((finite > threshold_factor * med).sum())


def step5_population_tda(
    trajectories: list[dict],
    output_dir: str,
    d_int_estimate: float | None = None,
) -> dict:
    """Run population-level TDA.

    Computes ripser(maxdim=2) on the cloud of mean hidden states,
    reduced to multiple PCA dimensions [10, 20, 50, 100] (capped at the
    data dimensionality).  Checks stability of H1/H2 features across
    reduction levels.

    Also provides a topological interpretation.
    """
    print("\n=== Step 5: Population TDA ===")

    # Build the point cloud: one point per trajectory (mean hidden state)
    mean_hs = np.array([t["hidden_states"].mean(axis=0) for t in trajectories], dtype=np.float32)
    n_points, d_hidden = mean_hs.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mean_hs)

    pca_dims = [d for d in [10, 20, 50, 100] if d < min(n_points, d_hidden)]
    if not pca_dims:
        pca_dims = [min(n_points - 1, d_hidden - 1, 5)]
    # Also add the raw (full) dimension if small enough
    if d_hidden <= 100 and d_hidden not in pca_dims:
        pca_dims.append(d_hidden)
    pca_dims = sorted(set(pca_dims))

    print(f"  PCA reduction levels: {pca_dims}")

    per_dim_results = []
    h1_counts = []
    h2_counts = []

    for d_pca in pca_dims:
        pca = PCA(n_components=d_pca)
        X_pca = pca.fit_transform(X_scaled)

        rips = ripser.ripser(X_pca, maxdim=2)
        dgms = rips["dgms"]

        n_h1 = _count_significant_features(dgms, dim=1)
        n_h2 = _count_significant_features(dgms, dim=2)

        h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
        n_components = int((h0[:, 1] == np.inf).sum()) if h0.shape[0] > 0 else 0

        h1_counts.append(n_h1)
        h2_counts.append(n_h2)

        per_dim_results.append(
            {
                "pca_dim": d_pca,
                "n_h0_components": n_components,
                "n_h1_significant": n_h1,
                "n_h2_significant": n_h2,
            }
        )

        print(
            f"  pca_dim={d_pca}: H0 components={n_components}, "
            f"significant H1={n_h1}, significant H2={n_h2}"
        )

    # Stability check: coefficient of variation of H1 counts across levels
    h1_arr = np.array(h1_counts, dtype=float)
    h2_arr = np.array(h2_counts, dtype=float)
    h1_cv = float(h1_arr.std() / (h1_arr.mean() + 1e-9))
    h2_cv = float(h2_arr.std() / (h2_arr.mean() + 1e-9))
    h1_stable = h1_cv < 0.5
    h2_stable = h2_cv < 0.5

    print(f"  H1 stability (CV={h1_cv:.2f}): {'stable' if h1_stable else 'unstable'}")
    print(f"  H2 stability (CV={h2_cv:.2f}): {'stable' if h2_stable else 'unstable'}")

    # Topological interpretation
    h1_mean_count = float(h1_arr.mean())
    interpretation_parts = []
    if h1_mean_count < 1:
        interpretation_parts.append("No persistent H1 loops — point cloud is contractible or sparse.")
    elif h1_mean_count < 3:
        interpretation_parts.append(
            f"~{h1_mean_count:.0f} persistent H1 loop(s) detected — possible circular or cyclic structure."
        )
    else:
        interpretation_parts.append(
            f"{h1_mean_count:.0f} significant H1 loops on average — complex topology with multiple cycles."
        )
    if not h1_stable:
        interpretation_parts.append("H1 features are NOT stable across PCA levels — may be noise.")
    if d_int_estimate is not None and not np.isnan(d_int_estimate):
        interpretation_parts.append(
            f"Estimated intrinsic dim ≈ {d_int_estimate:.1f} from Step 3."
        )
    interpretation = " ".join(interpretation_parts)
    print(f"  Interpretation: {interpretation}")

    # Stability plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_labels = [str(d) for d in pca_dims]
    x_pos = np.arange(len(pca_dims))

    axes[0].bar(x_pos, h1_counts, color="steelblue", alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_xlabel("PCA Dimension")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Significant H1 Features\n(CV={h1_cv:.2f}, {'stable' if h1_stable else 'unstable'})")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x_pos, h2_counts, color="goldenrod", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel("PCA Dimension")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Significant H2 Features\n(CV={h2_cv:.2f}, {'stable' if h2_stable else 'unstable'})")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Population TDA Stability Across PCA Reduction Levels", fontsize=12)
    plot_path = Path(output_dir) / "step5_population_tda.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    results = {
        "per_pca_dim": per_dim_results,
        "h1_cv": h1_cv,
        "h1_stable": h1_stable,
        "h2_cv": h2_cv,
        "h2_stable": h2_stable,
        "interpretation": interpretation,
    }
    out_path = Path(output_dir) / "step5_population_tda.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse requested steps
    steps_to_run = set(int(s.strip()) for s in args.steps.split(",") if s.strip())

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    trajectories = load_trajectories(args.data_dir, args.max_trajectories)

    if not trajectories:
        print("No trajectories found. Exiting.")
        return

    d_int_estimate: float | None = None

    # Step 0: Summarize
    if 0 in steps_to_run:
        step0_summarize(trajectories, args.output_dir)

    # Step 1: Verify
    if 1 in steps_to_run:
        step1_verify(trajectories, args.output_dir, args.skip_verify, args.lean_timeout)

    # Step 2: Cluster separation
    if 2 in steps_to_run:
        step2_cluster_separation(trajectories, args.output_dir)

    # Step 3: Intrinsic dimensionality
    if 3 in steps_to_run:
        step3_results = step3_intrinsic_dim(trajectories, args.output_dir)
        d_int_estimate = step3_results.get("d_int_2nn_median")

    # Step 4: Individual TDA
    if 4 in steps_to_run:
        step4_individual_tda(trajectories, args.output_dir, n_samples=10)

    # Step 5: Population TDA
    if 5 in steps_to_run:
        step5_population_tda(trajectories, args.output_dir, d_int_estimate=d_int_estimate)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
