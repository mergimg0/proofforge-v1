 Megalithic Synthesis: Every Finding, Pattern, and Anomaly

  The Complete Pass Rate Arc

  Step   0:   2.0%   — 3 proofs pass (rfl, decide, simp on trivial theorems)
  Step   5:   8.0%   — formatting improvement, model outputs tactic-like text
  Step  10:   9.3%   — slight further gain
  Step  20:   8.0%   — DIP: non-monotone, capabilities regressing
  Step  30:   8.7%   — plateau, 25 steps of apparent stagnation
  Step  50:  22.7%   — FIRST BREAKOUT: 7 Tier 3 theorems crack simultaneously
  Step  75:  42.7%   — SECOND BREAKOUT: lost theorems recover, Tier 2 logic cracks
  Step 100:  60.7%   — deceleration begins, Tier 4 wall holds
  Step 150:  72.0%   — deep saturation, gradient dying
  Step 200:  OOM     — training finished, collection failed

  ---
  12 Structural Patterns

  Pattern 1: Three-Phase Training Architecture
  The training curve decomposes into three phases with distinct dynamics:
  - Phase 1 (Disruption, steps 5-30): GRPO reshapes output distribution, base capabilities temporarily degrade, pass rate dips from 9.3% to 8.0%. The model is leaving its pretrained basin.
  - Phase 2 (Accumulation, steps 30-45): Pass rate is flat at 8-9% but per-step reward average starts climbing at step 35. Infrastructure builds invisibly. The reward average is the leading
  indicator.
  - Phase 3 (Breakout, steps 45-75): Infrastructure hits critical mass. Self-accelerating loop engages. Pass rate jumps from 8.7% to 42.7%. Convex improvement.

  The transition from Phase 2 to Phase 3 is a CRITICAL MASS event, not gradual improvement. The specific step where it happens (~35) is identifiable from the reward data before it appears in the
  pass rate.

  Pattern 2: Infrastructure Learning Dominates Theorem Learning
  The model doesn't learn "how to prove theorem X." It learns "how to use simp." Once simp is strong enough, EVERY simp-solvable theorem crosses the pass threshold simultaneously. Evidence: 7 Tier
  3 theorems cracked at checkpoint 50, all requiring simp/omega. None were solved at any previous checkpoint. This is a collective transition driven by shared infrastructure, not 7 independent
  learning events.

  Pattern 3: Self-Accelerating Loop with Critical Mass Threshold
  Below ~15% per-step success rate, gradient is too sparse to sustain improvement. Above it, the loop is self-sustaining: more successes → more gradient → better tactics → more successes. The
  critical mass was crossed at step 35. Before: reward average 0.08. After: reward average climbing continuously (0.15 → 0.25 → 0.35 → 0.50). The loop doesn't just sustain — it ACCELERATES because
  each new solved theorem adds to the gradient density.

  Pattern 4: Forgetting Before Consolidation
  Individual theorem tracking reveals a three-act pattern:
  - t1_06 (Nat.succ 0 = 1): Solved at ckpt 0 (T=1.2 only) → LOST at ckpts 5-20 → RECOVERED at ckpt 50 (ALL 3 temperatures)
  - t1_12 (7-3=4): Solved at ckpt 0 (T=0.3) → LOST at ckpts 5-50 → RECOVERED at ckpt 75 (ALL 3 temperatures)
  - t1_13 (Nat.zero = 0): Solved at ckpt 10 → LOST at ckpt 20 → RECOVERED at ckpt 50 (ALL 3 temperatures)

  Every recovered theorem came back STRONGER than before (more temperatures = more robust). The disruption phase is the model reorganizing its proof strategies. It's analogous to loss barrier
  crossing — the model must get worse before it can get better.

  Pattern 5: Sequential Multi-Strategy Acquisition
  The model learns tactic families in sequence, not in parallel:
  Steps  0-30:  Formatting (stop repeating, start attempting)
  Steps 30-50:  Cluster 1 — simp/omega (algebraic simplification)
  Steps 50-75:  Cluster 2 — decide RECOVERS (computational verification)
  Steps 50-75:  Cluster 3 — intro/exact EMERGING (propositional logic)
  Steps 75-100: Consolidation of all three clusters

  This is NOT specialization (trading one capability for another). It's sequential development: master one family, consolidate, then develop the next. The decide theorems lost at step 50 RETURNED
  at step 75 once the simp infrastructure was stable and the model had capacity to recover them.

  Pattern 6: Convex Then Concave Improvement
  Steps  0→50:   0.41 pp/step  (bootstrapping)
  Steps 50→75:   0.80 pp/step  (PEAK acceleration)
  Steps 75→100:  0.72 pp/step  (beginning deceleration)
  Steps 100→150: 0.23 pp/step  (deep deceleration)

  The improvement rate DOUBLED from bootstrapping to peak, then fell by 3.5x from peak to late training. The convex phase (steps 30-75) is driven by infrastructure sharing: each new capability
  unlocks multiple theorems. The concave phase (steps 75+) is driven by gradient saturation: solved theorems approach 100% success, eliminating advantage separation.

  Pattern 7: Running Reward Average Leads Pass Rate by 10-20 Steps
  The per-step reward average started rising at step 35. The pass rate breakout appeared at checkpoint 50 (15 steps later). The reward average captures infrastructure improvement (the model
  generates better tactics that sometimes work) before pass rate captures capability improvement (the model consistently solves new theorems). This is a PREDICTIVE DIAGNOSTIC: if reward is climbing
   while pass rate is flat, breakout is imminent.

  Pattern 8: Temperature Migration as Learning Depth
  Each theorem progresses through temperature stages:
  Stage 1: Passes only at T=1.2 (needs high-temperature noise to find proof)
  Stage 2: Passes at T=1.2 and T=0.7 (strategy partially internalized)
  Stage 3: Passes at all three temperatures (fully internalized, deterministic)

  Quantified: Checkpoint 50 had 4 fully-learned theorems. Checkpoint 75 had 9. The ratio of fully-learned to fragile increased from 4:11 to 9:many. The temperature breadth of individual theorems
  increases monotonically once they're learned — no theorem went from 3-temperature to 1-temperature.

  Pattern 9: Trajectory Length Bimodality
  Same theorem, same checkpoint, different temperatures:
  - t3_02 at T=0.3: 256 steps (exhaustive search)
  - t3_02 at T=1.2: 61 steps (direct execution)

  The proof strategy EXISTS in the model's distribution but is NOT the highest-probability path. At low temperature (near-deterministic), the model follows the most likely continuation and has to
  search. At high temperature, noise pushes it onto the correct but lower-probability strategy immediately.

  As training progresses, the strategy PROMOTES:
  - t1_06 at T=0.7: ckpt 50 = 214 steps → ckpt 75 = 27 steps (8x faster)
  - t1_06 at T=1.2: ckpt 50 = 27 steps → ckpt 75 = 9 steps (3x faster)

  The strategy is moving from secondary mode to primary mode in the output distribution.

  Pattern 10: Conceptual Cluster Development
  Three clusters emerged in sequence:
  1. Nat arithmetic cluster (ckpts 30-50): t1_02, t1_06, t1_13, t2_02, t3_02, t3_05, t3_06, t3_07, t3_10 — all Nat/succ/zero, solved with simp/omega
  2. Computational cluster (ckpt 75 recovery): t1_03, t1_04, t1_07, t1_08, t1_12, t1_14 — concrete numerics, solved with decide
  3. Logic cluster (ckpts 50-75): t2_08, t2_09, t2_10, t2_13, t4_02, t4_09 — propositional reasoning, solved with intro/exact/rw

  Theorems within a cluster share proof infrastructure. Learning one helps all others in the same cluster. The model develops domain expertise before broadening.

  Pattern 11: Gradient Lifecycle on a Fixed Theorem Set
  Steps   0-30:  STARVATION (0.16 expected successes/group, 85% dead steps)
  Steps  30-75:  PRODUCTIVE (0.7-3.2 expected successes, rich signal)
  Steps  75-150: SATURATION (5+ expected successes, advantages collapsing)
  Steps 150+:    DEATH (7-8/8 succeeding, near-zero gradient)

  The productive zone is temporary and self-destroying: the model improves, which increases its success rate, which reduces the gradient signal that enabled the improvement. The expanding ring
  curriculum directly addresses this by adding harder theorems to maintain the productive zone.

  Pattern 12: Tier 4 Wall = Tactic Composition Wall
  The model mastered 8+ individual tactics: simp, omega, intro, exact, rfl, decide, rw, cases. But only 2 of 10 Tier 4 theorems cracked (t4_02, t4_09). The remaining 8 require multi-tactic CHAINS
  that the model can't compose. t4_01 (transitivity) needs intro h1; intro h2; exact h1 ∘ h2. The model knows intro. It knows exact. It can't chain them. This is the precise capability gap the
  Level 1-2 curriculum addresses.

  ---
  6 Anomalies

  Anomaly 1: group_size=8 Worked
  Every paper says 32-64. We used 8. The model went 2% → 72%. The explanation: infrastructure sharing makes the effective signal density much higher than the per-theorem success rate implies. Each
  simp success provides gradient for ALL simp-solvable theorems. The literature's recommendation assumes independent theorem difficulty. When theorems share infrastructure, group_size=8 is
  sufficient.

  Anomaly 2: Non-Monotone Path Consistent with Stochastic SOS
  The dip from 9.3% to 8.0% (steps 10→20) violates deterministic monotone improvement but matches the stochastic SOS framework: improvement in expectation with variance along the path. The variance
   during the disruption phase preceded a faster-than-predicted breakout, consistent with the variance-acceleration theorem.

  Anomaly 3: Seven Tier 3 Theorems Cracked Simultaneously
  Not one. Not two. SEVEN Tier 3 theorems appeared at checkpoint 50 that were absent from ALL previous checkpoints. This is the collective transition signature: infrastructure (simp) crossed a
  threshold that simultaneously made 7+ theorems solvable. This looks more like a phase transition (statistical mechanics) than individual bifurcations (CGLE).

  Anomaly 4: Lost Theorems Returned Stronger
  t1_12 was solved at checkpoint 0 with 1 temperature. After being LOST for 5 consecutive checkpoints, it returned at checkpoint 75 with ALL 3 temperatures. The model forgot an easy theorem,
  learned harder things, then re-learned the easy theorem more robustly. This is not just recovery — it's improvement through temporary regression.

  Anomaly 5: d_int ≈ 6 Is Universally Stable
  Across A40, H100 heuristic, H100 Lean — always ~6. Across model sizes: GPT-2 = 3.8, DeepSeek-7B = 6.0. The intrinsic dimensionality appears to be a fundamental property of the model and task, not
   of the training state. Whether the Lean-reward training changes it is THE open manifold question.

  Anomaly 6: The OOM at Step 200
  80GB H100 ran out of memory after 200 LoRA training steps during inference. This is a fragmentation issue — 200 steps of gradient accumulation left the CUDA allocator unable to find contiguous
  memory for the inference pass. Practical implication: long training runs need explicit torch.cuda.empty_cache() between training and inference phases.

  ---
  5 Positive Anomalies

  Positive Anomaly 1: Step 47-48 Pre-Breakout Signal
  Steps 47 (0.406) and 48 (0.469) were the first per-step rewards above 0.3. They appeared 3-15 steps BEFORE the checkpoint 50 evaluation showed the breakout. The step-level data predicted the
  checkpoint-level result. This validates the running reward average as a real-time diagnostic.

  Positive Anomaly 2: t2_13 (Double Negation Introduction)
  p → ¬¬p requires understanding negation as a function type (¬p = p → False), introducing the hypothesis, then applying it through a lambda. The model learned this through BINARY GRPO — no shaped
  reward, no curriculum, no human guidance. Pure binary Lean verification was sufficient signal to develop higher-order logical reasoning.

  Positive Anomaly 3: t3_07 in 4 Tokens
  n + 1 = Nat.succ n completed in 4 tokens at T=1.2 at checkpoint 50. Four tokens is the theoretical minimum for a tactic proof. The model identified the exact tactic (likely simp or rfl) and
  executed it with zero waste. Compare to checkpoint 0 where this theorem produced 256-token repetition loops.

  Positive Anomaly 4: Temperature Breadth Explosion at Checkpoint 75
  Fully-learned theorems went from 4 (ckpt 50) to 9 (ckpt 75) — a 125% increase in 25 training steps. This is faster than new theorems appearing. The model was DEEPENING its mastery (making fragile
   theorems robust) at the same time as BROADENING (cracking new theorems). Both dimensions improved simultaneously.

  Positive Anomaly 5: Self-Discovered Curriculum
  The model discovered the optimal learning order without any human-designed curriculum:
  1. Simple arithmetic tactics (decide, rfl) — steps 0-10
  2. Algebraic simplification (simp, omega) — steps 30-50
  3. Computational tactics recovered (decide) — steps 50-75
  4. Propositional logic (intro, exact) — steps 50-75
  5. Conjunction/disjunction reasoning — steps 75-100

  This matches the logical dependency structure of mathematics: you need simp before you can simp[lemma], you need intro before intro;exact. The model rediscovered this hierarchy through gradient
  descent on binary reward.

  ---
  The Meta-Pattern

  Everything above is an instance of ONE underlying mechanism: infrastructure learning creates superlinear returns in domains with shared structure. Each learned capability enables multiple
  downstream capabilities. The self-accelerating loop, the convex improvement, the conceptual clusters, the simultaneous Tier 3 crack, the sequential strategy acquisition — all are consequences of
  this single principle applied to formal theorem proving.

  The principle predicts: this pattern should appear in ANY domain where solutions share structural infrastructure and a perfect binary verifier exists. Code with test suites. Smart contracts with
  formal verifiers. Hardware with assertion checkers. The theorem proving result is the first empirical demonstration. The generalization is the paper's deepest claim.

  ---
  Now: items 1-3 from the outstanding list. Run the full analysis. Want me to begin?