# ProofForge Training: Open Analysis Threads
## Source: Session 0320e841, outstanding items as of 2026-03-30
## Status: ALL THREADS COMPLETE (2026-04-05)

### 1. Full training_dynamics.py on ALL 9 Lean-reward checkpoints
**Status: COMPLETE (2026-04-04)**
All 9 checkpoints analyzed (1,350 trajectories). Results: 44/50 theorems solved, 1 permanent regression,
failure modes shift from REPETITION (40%) to TACTIC_ATTEMPT (43%), temperature migration reverses
(exploration→internalization). Output: `cgle_analysis_results/training_dynamics_full/`

### 2. Manifold Comparison: Checkpoint 30 vs Checkpoint 75
**Status: COMPLETE (2026-04-04/05)**
Full analyze.py (steps 0-5) + modes.py on both checkpoints.
- d_int: 6.90 → 5.77 (compressed)
- Separation: 86.7% YES → 61.3% NO (inverted)
- H1 topology: 13 → 12 (preserved)
- Phase locking: 0.785 max → 0.578 max (dissolving)
- CGLE spectrum: negative slope at both (consistent)
Output: `cgle_analysis_results/manifold_ckpt030/`, `manifold_ckpt075/`, `modes_ckpt030/`, `modes_ckpt075/`

### 3. Full 5-Question Analysis (analyze.py) on Lean-reward Checkpoints
**Status: COMPLETE (2026-04-05)**
All 9 checkpoints have full analyze.py (steps 0,2,3,4,5) + modes.py (10 modes, 50 trajectories).
Complete evolution table in FULL_ANALYSIS_SYNTHESIS.md.
Key findings: d_int 7.01→5.20, H1 stable (8-13), phase locking dissolves (0.785→0.450),
CGLE spectrum consistent at all checkpoints.
Output: `cgle_analysis_results/manifold_ckpt{000-150}/`, `modes_ckpt{000-150}/`

### 4. Learning Depth Metric (Temperature Breadth) — Systematic
**Status: COMPLETE (2026-04-04)**
26 fully-learned (3T), 13 mostly (2T), 4 fragile (1T) at C150. Depth ratio 0→0.605.
26/50 theorems show monotonic depth increase.
Output: `cgle_analysis_results/extended/`

### 5. Validate ALL Registered Predictions vs Actual Data
**Status: COMPLETE (2026-04-05)**
12 predictions validated: 9 confirmed, 1 partially confirmed, 2 wrong.
Wrong: pass rate magnitude (2.5x underestimate), stability direction (decreased, not increased).
Output: `cgle_analysis_results/prediction_validation.md`

### 6. Tactic Analysis Across Checkpoints
**Status: COMPLETE (2026-04-04)**
simp dominates at 92% by C150. Full tactic × checkpoint matrix computed.
Output: `cgle_analysis_results/extended/`

### 7. Trajectory Length Ratio (low-T / high-T) Across Checkpoints
**Status: COMPLETE (2026-04-04)**
9.6x ratio at C150 (low-T 256 tokens, high-T 145 tokens). Strategies not yet promoted.
Output: `cgle_analysis_results/extended/`

### 8. Conceptual Cluster Tracking — Systematic
**Status: COMPLETE (2026-04-04)**
6 clusters identified. Nat_arithmetic first (100% by C100), logic clusters crack at C075 breakout.
Hierarchical development confirms infrastructure learning.

---

### All Data Locations
- Full synthesis: `cgle_analysis_results/FULL_ANALYSIS_SYNTHESIS.md`
- Prediction validation: `cgle_analysis_results/prediction_validation.md`
- Training dynamics: `cgle_analysis_results/training_dynamics_full/`
- Extended analyses: `cgle_analysis_results/extended/`
- Manifold per-checkpoint: `cgle_analysis_results/manifold_ckpt{000,005,010,020,030,050,075,100,150}/`
- Modes per-checkpoint: `cgle_analysis_results/modes_ckpt{000,005,010,020,030,050,075,100,150}/`
