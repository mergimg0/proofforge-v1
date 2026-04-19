"""
modes.py — Mode amplitude tracker for CGLE dynamics in transformer hidden states.

Analyzes how hidden-state trajectories decompose into PCA modes and whether
those modes exhibit CGLE-predicted behavior: linear decay of growth rates with
mode index, phase-locked vs turbulent oscillation, etc.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CGLE mode amplitude tracker for transformer hidden states"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing .npz trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots and JSON results are written",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="Number of PCA modes to extract per trajectory (default: 10)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=20,
        help="Maximum number of trajectories to process (default: 20)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=30,
        help="Minimum number of time steps required to include a trajectory (default: 30)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(
    data_dir: Path,
    min_steps: int,
    max_n: int,
) -> list[dict]:
    """
    Load .npz trajectory files from data_dir.

    Each file is expected to contain at least a 'hidden_states' array of shape
    (T, D) where T is the number of time steps and D is the hidden dimension.
    Files with fewer than min_steps time steps are skipped.

    Returns a list (up to max_n entries) of dicts with keys:
        name        : stem of the .npz file
        hidden_states : np.ndarray of shape (T, D)
        n_steps     : T
        n_dims      : D
    """
    trajectories = []
    npz_files = sorted(data_dir.glob("*.npz"))

    for fpath in npz_files:
        if len(trajectories) >= max_n:
            break
        data = np.load(fpath, allow_pickle=True)
        if "hidden_states" not in data:
            print(f"  [skip] {fpath.name}: no 'hidden_states' key")
            continue
        hs = data["hidden_states"]
        if hs.ndim != 2:
            print(f"  [skip] {fpath.name}: hidden_states must be 2-D, got shape {hs.shape}")
            continue
        T, D = hs.shape
        if T < min_steps:
            print(f"  [skip] {fpath.name}: only {T} steps (need {min_steps})")
            continue
        trajectories.append(
            {
                "name": fpath.stem,
                "hidden_states": hs.astype(np.float64),
                "n_steps": T,
                "n_dims": D,
            }
        )
        print(f"  [load] {fpath.name}: T={T}, D={D}")

    print(f"Loaded {len(trajectories)} trajectories from {data_dir}")
    return trajectories


# ---------------------------------------------------------------------------
# PCA decomposition
# ---------------------------------------------------------------------------

def decompose_trajectory(
    hidden_states: np.ndarray,
    n_modes: int,
) -> dict:
    """
    Perform PCA on a centered trajectory.

    Parameters
    ----------
    hidden_states : (T, D) array
    n_modes       : number of principal components to retain

    Returns
    -------
    dict with keys:
        mode_amplitudes   : (T, n_modes) — projection onto each PC
        eigenvalues       : (n_modes,)   — explained variance per PC
        explained_ratio   : (n_modes,)   — fraction of variance per PC
        cumulative_variance : (n_modes,) — cumulative explained variance
    """
    T, D = hidden_states.shape
    k = min(n_modes, T, D)

    # Center
    centered = hidden_states - hidden_states.mean(axis=0, keepdims=True)

    pca = PCA(n_components=k)
    mode_amplitudes = pca.fit_transform(centered)  # (T, k)

    eigenvalues = pca.explained_variance_          # (k,)
    explained_ratio = pca.explained_variance_ratio_  # (k,)
    cumulative_variance = np.cumsum(explained_ratio)  # (k,)

    return {
        "mode_amplitudes": mode_amplitudes,
        "eigenvalues": eigenvalues,
        "explained_ratio": explained_ratio,
        "cumulative_variance": cumulative_variance,
    }


# ---------------------------------------------------------------------------
# Analytic signal analysis
# ---------------------------------------------------------------------------

def compute_analytic_signal(mode_amplitudes: np.ndarray) -> dict:
    """
    Apply the Hilbert transform to each mode to extract instantaneous amplitude,
    phase, and frequency.

    Parameters
    ----------
    mode_amplitudes : (T, n_modes) array

    Returns
    -------
    dict with keys:
        amplitudes  : (T, n_modes) — instantaneous envelope
        phases      : (T, n_modes) — instantaneous phase (radians)
        frequencies : (T, n_modes) — instantaneous frequency (cycles / sample)
    """
    T, n_modes = mode_amplitudes.shape
    amplitudes = np.zeros_like(mode_amplitudes)
    phases = np.zeros_like(mode_amplitudes)
    frequencies = np.zeros_like(mode_amplitudes)

    for k in range(n_modes):
        analytic = hilbert(mode_amplitudes[:, k])
        amplitudes[:, k] = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        phases[:, k] = phase
        # Instantaneous frequency: derivative of unwrapped phase / 2π
        freq = np.gradient(phase) / (2.0 * np.pi)
        frequencies[:, k] = freq

    return {
        "amplitudes": amplitudes,
        "phases": phases,
        "frequencies": frequencies,
    }


# ---------------------------------------------------------------------------
# Growth rate fitting
# ---------------------------------------------------------------------------

def fit_growth_rate(amplitude: np.ndarray) -> tuple[float, float, float]:
    """
    Fit an exponential growth model A(t) = a0 * exp(sigma * t) to the
    instantaneous amplitude by performing a log-linear regression.

    Parameters
    ----------
    amplitude : (T,) array of non-negative values

    Returns
    -------
    (sigma, a0, r_squared)
        sigma      : exponential growth rate (positive → growing, negative → decaying)
        a0         : initial amplitude
        r_squared  : coefficient of determination of the log-linear fit
    """
    eps = 1e-12
    amp = np.clip(amplitude, eps, None)
    log_amp = np.log(amp)
    T = len(log_amp)
    t = np.arange(T, dtype=np.float64)

    # Linear fit: log_amp = sigma * t + log(a0)
    def log_linear(t_arr, sigma, log_a0):
        return sigma * t_arr + log_a0

    try:
        popt, _ = curve_fit(log_linear, t, log_amp, p0=[0.0, log_amp[0]], maxfev=2000)
        sigma, log_a0 = popt
        a0 = np.exp(log_a0)
    except RuntimeError:
        # Fallback: numpy polyfit
        coeffs = np.polyfit(t, log_amp, 1)
        sigma, log_a0 = coeffs
        a0 = np.exp(log_a0)

    # R²
    fitted = sigma * t + log_a0
    ss_res = np.sum((log_amp - fitted) ** 2)
    ss_tot = np.sum((log_amp - log_amp.mean()) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + eps)

    return float(sigma), float(a0), float(r_squared)


# ---------------------------------------------------------------------------
# Frequency stability
# ---------------------------------------------------------------------------

def analyze_frequency_stability(
    frequencies: np.ndarray,
    window: int = 10,
) -> float:
    """
    Compute a stability score for an instantaneous frequency time series using
    a sliding-window variance ratio.

    A perfectly locked phase oscillator has constant frequency → variance = 0.
    A chaotic/turbulent oscillator has rapidly varying frequency → high variance.

    The score is 1 − (mean windowed variance / total variance), clipped to [0, 1].

    Parameters
    ----------
    frequencies : (T,) array of instantaneous frequencies
    window      : sliding window size

    Returns
    -------
    stability_score in [0, 1]
        1 = perfectly locked
        0 = maximally turbulent
    """
    T = len(frequencies)
    eps = 1e-12

    if T < window:
        return 0.0

    # Sliding window variances
    windowed_vars = []
    for i in range(T - window + 1):
        seg = frequencies[i : i + window]
        windowed_vars.append(np.var(seg))

    mean_windowed_var = np.mean(windowed_vars)
    total_var = np.var(frequencies) + eps

    stability = 1.0 - mean_windowed_var / total_var
    return float(np.clip(stability, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------

def classify_mode(
    sigma: float,
    r_squared: float,
    amp_cv: float,
    freq_stability: float,
) -> str:
    """
    Classify a single PCA mode based on its dynamical properties.

    Rules (in priority order):
        r_squared < 0.1              → NOISE           (fit too poor to trust)
        sigma < -0.01                → DECAYING         (exponentially damped)
        freq_stability > 0.5         → ACTIVE_LOCKED    (oscillating, phase-locked)
        else                         → ACTIVE_TURBULENT (oscillating, phase-turbulent)

    Parameters
    ----------
    sigma          : exponential growth rate
    r_squared      : quality of log-linear fit
    amp_cv         : coefficient of variation of amplitude (std / mean)
    freq_stability : frequency stability score in [0, 1]

    Returns
    -------
    One of: 'NOISE', 'DECAYING', 'ACTIVE_LOCKED', 'ACTIVE_TURBULENT'
    """
    if r_squared < 0.1:
        return "NOISE"
    if sigma < -0.01:
        return "DECAYING"
    if freq_stability > 0.5:
        return "ACTIVE_LOCKED"
    return "ACTIVE_TURBULENT"


# ---------------------------------------------------------------------------
# Single-trajectory analysis
# ---------------------------------------------------------------------------

def analyze_single_trajectory(traj: dict, n_modes: int) -> dict:
    """
    Run the full analysis pipeline on one trajectory.

    Returns a dict with keys:
        name            : trajectory name
        n_steps         : number of time steps
        n_dims          : hidden dimension
        decomposition   : output of decompose_trajectory (without mode_amplitudes array)
        analytic_signal : output of compute_analytic_signal (arrays omitted in JSON)
        mode_results    : list of per-mode result dicts
        category_counts : dict with counts for each classification label
        _arrays         : internal arrays needed for plotting (not serialised)
    """
    hs = traj["hidden_states"]
    decomp = decompose_trajectory(hs, n_modes)
    mode_amps = decomp["mode_amplitudes"]           # (T, k)
    analytic = compute_analytic_signal(mode_amps)   # dicts of (T, k) arrays

    k = mode_amps.shape[1]
    mode_results = []
    category_counts = {
        "ACTIVE_LOCKED": 0,
        "ACTIVE_TURBULENT": 0,
        "DECAYING": 0,
        "NOISE": 0,
    }

    for i in range(k):
        amp_series = analytic["amplitudes"][:, i]
        freq_series = analytic["frequencies"][:, i]

        sigma, a0, r_sq = fit_growth_rate(amp_series)
        freq_stab = analyze_frequency_stability(freq_series)

        amp_mean = float(np.mean(amp_series))
        amp_std = float(np.std(amp_series))
        amp_cv = amp_std / (amp_mean + 1e-12)

        category = classify_mode(sigma, r_sq, amp_cv, freq_stab)
        category_counts[category] += 1

        mode_results.append(
            {
                "mode_index": i,
                "sigma": sigma,
                "a0": a0,
                "r_squared": r_sq,
                "amp_mean": amp_mean,
                "amp_std": amp_std,
                "amp_cv": float(amp_cv),
                "freq_stability": freq_stab,
                "mean_frequency": float(np.mean(freq_series)),
                "eigenvalue": float(decomp["eigenvalues"][i]),
                "explained_ratio": float(decomp["explained_ratio"][i]),
                "category": category,
            }
        )

    return {
        "name": traj["name"],
        "n_steps": traj["n_steps"],
        "n_dims": traj["n_dims"],
        "cumulative_variance": decomp["cumulative_variance"].tolist(),
        "mode_results": mode_results,
        "category_counts": category_counts,
        # Internal arrays for plotting only
        "_arrays": {
            "mode_amplitudes": mode_amps,
            "amplitudes": analytic["amplitudes"],
            "frequencies": analytic["frequencies"],
        },
    }


# ---------------------------------------------------------------------------
# Per-trajectory plot
# ---------------------------------------------------------------------------

_CATEGORY_COLORS = {
    "ACTIVE_LOCKED": "green",
    "ACTIVE_TURBULENT": "orange",
    "DECAYING": "red",
    "NOISE": "gray",
}


def plot_trajectory_modes(
    traj: dict,
    analysis: dict,
    output_path: Path,
) -> None:
    """
    2×2 subplot figure for a single trajectory:
        (0,0) Raw PCA projections (first 4 modes)
        (0,1) Instantaneous amplitude envelopes (all modes)
        (1,0) Growth rates bar chart (coloured by category)
        (1,1) Frequency stability bar chart (coloured by category)
    """
    arrays = analysis["_arrays"]
    mode_amps = arrays["mode_amplitudes"]   # (T, k)
    amplitudes = arrays["amplitudes"]       # (T, k)
    results = analysis["mode_results"]
    k = len(results)
    T = mode_amps.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Mode Analysis: {analysis['name']}", fontsize=13, fontweight="bold")

    # --- (0,0) Raw projections ---
    ax = axes[0, 0]
    n_show = min(4, k)
    for i in range(n_show):
        ax.plot(t, mode_amps[:, i], label=f"Mode {i}", alpha=0.8, linewidth=0.9)
    ax.set_title("Raw PCA Projections (first 4 modes)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Amplitude envelopes ---
    ax = axes[0, 1]
    for i in range(k):
        color = _CATEGORY_COLORS[results[i]["category"]]
        ax.plot(t, amplitudes[:, i], color=color, alpha=0.6, linewidth=0.8)
    ax.set_title("Instantaneous Amplitude Envelopes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Envelope |A(t)|")
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=lbl)
        for lbl, c in _CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Growth rates ---
    ax = axes[1, 0]
    mode_indices = [r["mode_index"] for r in results]
    sigmas = [r["sigma"] for r in results]
    colors = [_CATEGORY_COLORS[r["category"]] for r in results]
    ax.bar(mode_indices, sigmas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Growth Rates by Mode")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("σ (growth rate)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Frequency stability ---
    ax = axes[1, 1]
    stabilities = [r["freq_stability"] for r in results]
    ax.bar(mode_indices, stabilities, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="purple", linewidth=0.8, linestyle="--", label="Lock threshold (0.5)")
    ax.set_title("Frequency Stability by Mode")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Stability score [0,1]")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate spectrum plot
# ---------------------------------------------------------------------------

def plot_growth_rate_spectrum(
    all_analyses: list[dict],
    output_path: Path,
) -> None:
    """
    3-panel figure summarising results across all trajectories:
        Panel 1: Mean growth rate vs mode index with linear fit
                 (CGLE prediction: sigma_k decreases linearly with k)
        Panel 2: Category distribution bar chart (stacked)
        Panel 3: Growth rate vs frequency stability scatter
    """
    if not all_analyses:
        return

    # Gather per-mode statistics across trajectories
    # Find common mode count
    k = min(len(a["mode_results"]) for a in all_analyses)

    sigma_matrix = np.array(
        [[r["sigma"] for r in a["mode_results"][:k]] for a in all_analyses]
    )  # (n_traj, k)
    stability_matrix = np.array(
        [[r["freq_stability"] for r in a["mode_results"][:k]] for a in all_analyses]
    )

    mean_sigma = sigma_matrix.mean(axis=0)   # (k,)
    std_sigma = sigma_matrix.std(axis=0)     # (k,)
    mode_indices = np.arange(k)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"CGLE Growth Rate Spectrum  (n={len(all_analyses)} trajectories)",
        fontsize=13,
        fontweight="bold",
    )

    # --- Panel 1: Mean sigma vs mode index ---
    ax = axes[0]
    ax.errorbar(
        mode_indices,
        mean_sigma,
        yerr=std_sigma,
        fmt="o-",
        color="steelblue",
        alpha=0.8,
        capsize=4,
        label="Mean ± std",
    )

    # Linear fit (CGLE prediction)
    if k >= 2:
        coeffs = np.polyfit(mode_indices, mean_sigma, 1)
        fit_line = np.polyval(coeffs, mode_indices)
        slope, intercept = coeffs
        ax.plot(
            mode_indices,
            fit_line,
            "r--",
            linewidth=1.5,
            label=f"Linear fit: slope={slope:.4f}",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Growth Rate Spectrum\n(CGLE: linear decay with mode index)")
    ax.set_xlabel("Mode Index k")
    ax.set_ylabel("Mean Growth Rate σ_k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Category distribution ---
    ax = axes[1]
    category_totals: dict[str, int] = {
        "ACTIVE_LOCKED": 0,
        "ACTIVE_TURBULENT": 0,
        "DECAYING": 0,
        "NOISE": 0,
    }
    for analysis in all_analyses:
        for cat, cnt in analysis["category_counts"].items():
            category_totals[cat] += cnt

    categories = list(category_totals.keys())
    counts = [category_totals[c] for c in categories]
    bar_colors = [_CATEGORY_COLORS[c] for c in categories]
    ax.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, (cat, cnt) in enumerate(zip(categories, counts)):
        ax.text(i, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("Mode Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count (across all trajectories)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: sigma vs freq_stability scatter ---
    ax = axes[2]
    all_sigma = []
    all_stability = []
    all_colors = []
    for analysis in all_analyses:
        for r in analysis["mode_results"]:
            all_sigma.append(r["sigma"])
            all_stability.append(r["freq_stability"])
            all_colors.append(_CATEGORY_COLORS[r["category"]])

    ax.scatter(
        all_stability,
        all_sigma,
        c=all_colors,
        alpha=0.5,
        s=20,
        edgecolors="none",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0.5, color="purple", linewidth=0.8, linestyle="--", label="Lock threshold")
    ax.set_title("Growth Rate vs Frequency Stability")
    ax.set_xlabel("Frequency Stability [0,1]")
    ax.set_ylabel("Growth Rate σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes_dir = args.output_dir / "modes"
    modes_dir.mkdir(exist_ok=True)

    # Load trajectories
    print(f"\nLoading trajectories from: {args.data_dir}")
    trajectories = load_trajectories(
        data_dir=args.data_dir,
        min_steps=args.min_steps,
        max_n=args.n_trajectories,
    )

    if not trajectories:
        print("No valid trajectories found. Exiting.")
        return

    # Analyse each trajectory
    all_analyses = []
    print(f"\nAnalysing {len(trajectories)} trajectories with {args.n_modes} modes each...")

    for traj in trajectories:
        print(f"  Processing: {traj['name']}")
        analysis = analyze_single_trajectory(traj, n_modes=args.n_modes)
        all_analyses.append(analysis)

        # Per-trajectory plot
        plot_path = modes_dir / f"{traj['name']}_modes.png"
        plot_trajectory_modes(traj, analysis, plot_path)

    # Summary statistics
    total_modes = sum(len(a["mode_results"]) for a in all_analyses)
    total_counts: dict[str, int] = {
        "ACTIVE_LOCKED": 0,
        "ACTIVE_TURBULENT": 0,
        "DECAYING": 0,
        "NOISE": 0,
    }
    for analysis in all_analyses:
        for cat, cnt in analysis["category_counts"].items():
            total_counts[cat] += cnt

    print("\n=== Summary ===")
    print(f"  Trajectories analysed : {len(all_analyses)}")
    print(f"  Total modes classified: {total_modes}")
    for cat, cnt in total_counts.items():
        pct = 100.0 * cnt / total_modes if total_modes > 0 else 0.0
        print(f"    {cat:<20s}: {cnt:4d}  ({pct:.1f}%)")

    # Check CGLE linear-decay prediction
    if all_analyses:
        k = min(len(a["mode_results"]) for a in all_analyses)
        sigma_matrix = np.array(
            [[r["sigma"] for r in a["mode_results"][:k]] for a in all_analyses]
        )
        mean_sigma = sigma_matrix.mean(axis=0)
        if k >= 2:
            coeffs = np.polyfit(np.arange(k), mean_sigma, 1)
            slope = coeffs[0]
            print(f"\n  CGLE linear-decay fit slope: {slope:.5f}")
            if slope < 0:
                print("  → Consistent with CGLE prediction (higher modes more damped)")
            else:
                print("  → Inconsistent with CGLE prediction (slope should be negative)")

    # Save aggregate spectrum plot
    spectrum_path = args.output_dir / "modes_growth_spectrum.png"
    plot_growth_rate_spectrum(all_analyses, spectrum_path)
    print(f"\nSpectrum plot saved: {spectrum_path}")

    # Serialise results to JSON (strip internal arrays)
    json_analyses = []
    for analysis in all_analyses:
        entry = {k: v for k, v in analysis.items() if k != "_arrays"}
        json_analyses.append(entry)

    results_path = args.output_dir / "modes_analysis.json"
    with open(results_path, "w") as fh:
        json.dump(
            {
                "n_trajectories": len(json_analyses),
                "n_modes": args.n_modes,
                "category_totals": total_counts,
                "trajectories": json_analyses,
            },
            fh,
            indent=2,
        )
    print(f"Results JSON saved: {results_path}")


if __name__ == "__main__":
    main()
