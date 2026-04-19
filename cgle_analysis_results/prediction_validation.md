# Pre-Registered Prediction Validation — COMPLETE

## Original Predictions (from session_handoff_2026_03_29.md)

| Metric | Ckpt 0 (actual) | Ckpt 50 (prediction) | Ckpt 200 (prediction) |
|--------|-----------------|---------------------|----------------------|
| d_int | 6.1 | 5.5 +/- 1.0 | 5.0 +/- 1.5 |
| Pass rate | 2.0% | 10 +/- 5% | 12 +/- 8% |
| Max stability | 0.71 | 0.72 +/- 0.1 | 0.75 +/- 0.15 |
| Any mode crosses stable threshold | Yes | No new crossings | 25% chance |
| Pass rate discontinuity | N/A | 20% chance | 30% chance |

---

## Actual Results

### 1. d_int Evolution
| Checkpoint | Predicted | Actual (2NN mean) | Actual (2NN median) | Within CI? |
|---|---|---|---|---|
| C000 | 6.1 (baseline) | 7.01 | 6.30 | Higher than assumed baseline |
| C030 | - | 6.90 | 5.92 | - |
| C050 | 5.5 +/- 1.0 | **6.06** | **5.66** | YES (6.06 within [4.5, 6.5]) |
| C075 | - | 5.77 | 5.25 | - |
| C100 | - | **5.20** | **4.81** | - |
| C150 | - | **5.35** | **5.28** | - |
| C200 | 5.0 +/- 1.5 | OOM (no data) | - | N/A |

Prediction: 5.5 +/- 1.0 at C050. Actual: 6.06. Within 1-sigma CI.
Full trajectory: 7.01 → 6.90 → 6.06 → 5.77 → 5.20 → 5.35 (monotonic compression with slight rebound at saturation).
Verdict: **CONFIRMED** — d_int decreases monotonically through training, consistent with manifold compression.

### 2. Pass Rate
| Checkpoint | Predicted | Actual | Within CI? |
|---|---|---|---|
| C000 | 2.0% | 2.0% | baseline |
| C050 | 10 +/- 5% | **22.7%** | **NO** — 2.5x underestimate |
| C100 | - | 60.7% | - |
| C150 | - | 72.0% | - |
| C200 | 12 +/- 8% | OOM | N/A |

Verdict: **MASSIVELY UNDERESTIMATED**. Predicted 10+/-5% at step 50, got 22.7%.
Sigmoid asymptote of 72.95% was not predicted. This is prediction anchoring failure.

### 3. Pass Rate Discontinuity
Predicted: 20% chance at ckpt 50
Actual: **YES** — 14pp jump (C030→C050), 20pp jump (C050→C075)
Verdict: **CONFIRMED** — discontinuity occurred and was more dramatic than estimated.

### 4. Max Stability / Phase Locking
Predicted: 0.72 +/- 0.1 at C050, 0.75 +/- 0.15 at C200. "No new crossings" at C050.

Actual mode evolution (modes.py on all 9 checkpoints, 50 trajectories each, 10 PCA modes):
| Checkpoint | Max Stability | Locked% | Turbulent% | Noise% |
|---|---|---|---|---|
| C000 | 0.733 | 0.6% | 25.6% | 72.4% |
| C005 | 0.769 | 0.8% | 27.0% | 70.8% |
| C030 | **0.785** | 0.6% | 24.4% | 72.6% |
| C050 | 0.660 | 1.4% | 24.2% | 73.6% |
| C075 | 0.578 | 0.2% | 12.4% | 87.2% |
| C100 | 0.555 | 0.2% | 8.2% | 90.8% |
| C150 | **0.450** | **0.0%** | **3.8%** | **95.8%** |

Prediction at C050: 0.72 +/- 0.1. Actual: 0.660. **WITHIN CI** (barely, at lower edge).
Prediction at C200: 0.75 +/- 0.15 (increasing). Actual trend: **DECREASING** (0.785→0.450).
"No new crossings" at C050: **CORRECT** (no new stable modes appeared).

Verdict: **DIRECTION WRONG**. Stability was predicted to increase; it decreased by 43%.
Phase locking dissolves as the model masters the task — proofs become shorter and more
deterministic, eliminating the rich oscillatory structure that generated phase locking.
This is consistent with d_int compression: simpler dynamics on a lower-dimensional manifold.

CGLE linear-decay fit slope: negative at ALL checkpoints (range -0.00014 to -0.00049).
This is **consistent with CGLE prediction** (higher modes more damped), even as overall
phase locking decreases. The CGLE growth rate spectrum holds but the dynamical regime
transitions from turbulent to noise-dominated.

### 5. Tier 4 Cracks (implicit prediction: hard theorems remain unsolved)
From megalithic synthesis: "Tier 4 wall holds" — only 2/10 Tier 4 expected to crack.

Actual Tier 4 results from binary matrix:
| Theorem | C000-C030 | C050 | C075 | C100 | C150 | Status |
|---|---|---|---|---|---|---|
| t4_02 (p∧q → q∧p) | C005 | YES | YES | YES | YES | Cracked early |
| t4_09 (f a = f b) | - | C050 | YES | YES | YES | Cracked at breakout |
| t4_07 (¬(p∧¬p)) | - | - | C075 | YES | YES | Cracked post-breakout |
| t4_01 (transitivity) | - | - | - | C100 | YES | Late crack |
| t4_10 (de Morgan) | - | - | - | - | C150 | Very late crack |
| t4_03, t4_04, t4_05, t4_06, t4_08 | - | - | - | - | - | **NEVER** |

Prediction: 2/10 Tier 4 crack. Actual: **5/10** cracked (2.5x more than predicted).
The Tier 4 wall is real (5 never crack) but more porous than expected.
Verdict: **PARTIALLY CORRECT** — wall exists but is lower than predicted.

### 6. Lost Tier 1 Recovery (prediction: forgotten theorems should return stronger)
From megalithic synthesis Pattern 4: theorems lost during disruption should recover with wider temperature breadth.

Actual:
- t1_12 (7-3=4): Solved C000 → lost C005 → lost C010 → recovered C050 at 3T. **CONFIRMED stronger.**
- t1_06 (succ 0=1): Solved C010 → lost C020 → lost C030 → recovered C050 at 3T. **CONFIRMED stronger.**
- t2_01 (n=n): Solved C000 → lost C005 → recovered C020, then lost again, recovered C075 at 3T. **CONFIRMED.**
- t1_01 (True): Solved C005 → lost C010 → recovered C030, then lost, recovered C075 at 3T. **CONFIRMED.**

38 of 39 regression events eventually recovered. The one exception (t3_09: n*2=n+n) cracked at C075 then lost by C100.
Verdict: **STRONGLY CONFIRMED** — recovery rate 97.4% (38/39), and recovered theorems show wider temperature breadth.

### 7. Regression Rate Evolution
From early data: 16/27 theorems showed regression in ckpts 0-20.
Prediction: regression should decrease as model consolidates.

Actual consolidation trajectory:
| Transition | Retention | Regression Rate |
|---|---|---|
| C000→C005 | 0.0% | 1.000 (total disruption) |
| C005→C010 | 41.7% | 0.583 |
| C010→C020 | 23.1% | 0.769 |
| C020→C030 | 9.1% | 0.909 (worst) |
| C030→C050 | 50.0% | 0.500 |
| C050→C075 | **95.2%** | **0.048** |
| C075→C100 | **94.9%** | **0.051** |
| C100→C150 | **100%** | **0.000** |

Verdict: **CONFIRMED** — regression drops from 0.91 to 0.00. The transition is NOT gradual but a sharp phase transition at the breakout (C050→C075).

---

## Complete Prediction Scorecard

| # | Prediction | Verdict | Quantitative |
|---|---|---|---|
| 1 | d_int decreases with training | **CONFIRMED** | 7.01 → 5.20 (26% compression) |
| 2 | d_int ~5.5 at C050 | **WITHIN CI** | Actual 6.06 (within [4.5, 6.5]) |
| 3 | Pass rate 10% at C050 | **WRONG** (2.5x under) | Actual 22.7% |
| 4 | Pass rate discontinuity | **CONFIRMED** | 14pp + 20pp jumps |
| 5 | Non-monotone path (SOS) | **CONFIRMED** | 9.3% → 8.0% dip |
| 6 | d_int universally ~6 | **CONFIRMED** | Range 5.2-7.0, centered at ~6 |
| 7 | Tier 4 wall (2/10 crack) | **PARTIALLY** | 5/10 cracked (wall exists but lower) |
| 8 | Lost theorems recover stronger | **STRONGLY CONFIRMED** | 38/39 recovered (97.4%) |
| 9 | Regression decreases | **CONFIRMED** | 0.91 → 0.00 (phase transition) |
| 10 | Max stability increases | **WRONG** (direction) | 0.785→0.450, predicted increase |
| 11 | No new stable crossings (C050) | **CONFIRMED** | No new locked modes |
| 12 | CGLE linear-decay spectrum | **CONFIRMED** | Negative slope at all checkpoints |

**Overall: 9 confirmed, 1 partially confirmed, 2 wrong.**
Wrong predictions: pass rate magnitude (anchoring), stability direction (dissolution).

## Key Lessons
1. **Prediction anchoring is real**: Pass rate predictions anchored to the plateau were 2.5x too low.
2. **Phase transitions invalidate extrapolation**: The sigmoid from ckpts 0-20 would predict ~10% at C050. The breakout was a phase transition, not a continuation.
3. **Qualitative predictions outperform quantitative**: "d_int decreases" was correct; "d_int = 5.5" was approximately correct. "Pass rate = 10%" was wrong because it extrapolated within a regime.
4. **SOS framework predictions held**: Non-monotone path, eventual convergence, and the variance-acceleration connection all confirmed.
