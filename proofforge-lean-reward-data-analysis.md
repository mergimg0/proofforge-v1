  CRITICAL — Analysis on Downloaded Lean-Reward Data (tools exist, data exists, just need to run)

  1. Full training_dynamics.py on ALL 9 Lean-reward checkpoints — We only ran it on checkpoints 0-20 (the first 4). We now have 9 checkpoints (0, 5, 10, 20, 30, 50, 75, 100, 150) with 1,350
  trajectories. The per-theorem binary matrix, temperature migration, failure taxonomy, period tracking, trajectory length analysis, and pass rate breakdown have NOT been computed on the full
  dataset.
  2. Manifold comparison: checkpoint 30 vs checkpoint 75 — Both reviewers identified this as THE most scientifically important analysis. d_int, TDA, and mode structure before vs after the breakout.
   The data is downloaded. The analysis tools (analyze.py) exist. NOT RUN.
  3. Full 5-question analysis (analyze.py) on Lean-reward checkpoints — d_int evolution, phase locking evolution, population TDA evolution across 9 checkpoints. NOT RUN on the Lean-reward data.
  4. Learning depth metric (temperature breadth) systematically across all checkpoints — We computed this manually from log data for ckpts 50 and 75. Need systematic computation: for each theorem ×
   checkpoint, count temperatures at which it passes. Track fully-learned / mostly-learned / fragile ratios. NOT DONE.
  5. Validate ALL registered predictions against actual data — We registered predictions for ckpts 50, 75, 100, 150, 200. We checked some on-the-fly from logs. Need systematic comparison: predicted
   vs actual for d_int, pass rate, learning depth, Tier 4 cracks, lost Tier 1 recovery, regression rate. NOT DONE.
  6. Tactic analysis: what tactics are used in successful proofs at each checkpoint — The specialization finding (simp vs decide) was identified manually at ckpt 50. Need systematic extraction: for
   every passing trajectory at every checkpoint, what tactic was used? Track tactic distribution over training. NOT DONE.
  7. Trajectory length ratio (low-T / high-T) across checkpoints — Observed manually for t1_06 (ckpts 50 vs 75). Need systematic computation for every theorem that passes at multiple temperatures.
  Track whether low-T proofs get shorter. NOT DONE.
  8. Conceptual cluster tracking — The Nat cluster, computational cluster, and propositional logic cluster were identified from ckpt 50/75 logs. Need systematic analysis: group theorems by shared
  proof infrastructure, track cluster development across checkpoints. NOT DONE.
  9. Per-theorem period tracking across Lean-reward checkpoints — We ran this on heuristic-reward ckpts 0-20. NOT on Lean-reward data where actual learning occurred. The reviewer's prediction:
  theorems show period shifts when first learned. NOT VALIDATED.
  10. Geometric separation with real Lean labels across checkpoints — Failed at ckpt 0 with real labels. But at ckpt 100 the model has 61% pass rate — much more balanced classes. Does geometric
  separation emerge now? NOT TESTED.

  IMPORTANT — New Analyses to Compute

  11. Infrastructure usage analysis — For each successful proof: extract the first tactic used. Build a tactic × checkpoint matrix. Track which tactics the model relies on and how that changes.
  Validates the infrastructure learning mechanism.
  12. Tactic composition detection — Do successful proofs at ckpt 75-150 contain multi-tactic chains (semicolons, newlines between tactics)? If yes, the model learned composition on its own. If no,
   it still relies on single tactics and the Level 1-2 curriculum is essential.
  13. Regression analysis across all checkpoints — At each checkpoint: how many previously-solved theorems are still solved? Compute regression rate per checkpoint. Track whether it decreases
  (consolidation) or stays high (volatility). Full matrix not just ckpts 0-20.
  14. Compare the two runs — Heuristic-reward (ckpts 0-30, 746 files) vs Lean-reward (ckpts 0-150, 1350 files). Same base model, different rewards. Do the manifold properties differ? Does the
  heuristic run's flat 8-9% curve produce different hidden state geometry than the Lean run's 2-72% curve?
  15. Scaling law fit — Fit sigmoid to the pass rate curve. Extrapolate. The reviewer proposed this as Application 7. The data is complete. NOT DONE.

  DOCUMENTATION

  16. Update continuity document with final results — The doc says training is running. It's done. Need: final curve, OOM note, complete data inventory, analysis results.
  17. Store session learnings to memory — The memory entry exists but needs updating with final numbers and key insights.

  ---
  That's 17 outstanding items. Items 1-3 are the highest priority because they answer the questions we set out to ask. The data is downloaded. The tools are built.

  Want me to start executing? I'd begin with item 1 (full training_dynamics.py on all 9 checkpoints) — it covers the per-theorem matrix, temperature migration, failure taxonomy, and period tracking
   in one run.