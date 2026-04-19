# Idea Reactor Proposals — Evaluation Against Computed Findings

Evaluated: 2026-04-05, after completing all 17 analysis items + mode evolution.

---

## IR-1: Categorical Proof Curriculum (`ir-cat-curriculum-001`)

**Proposal**: Replace proof-length-as-difficulty with categorical-depth-as-difficulty.
Levels mapped to: coproducts → initial algebras → exponentials → composition → universal properties → natural transformations.

**Evaluation against findings**:

| Finding | Implication for this proposal |
|---|---|
| simp monoculture (92%) | SUPPORTS — current Level 1-2 curriculum is simp-centric. Categorical framing would force tactic diversity. |
| 49% tactic composition at C150 | Model CAN learn composition. Categorical levels would systematize what's already emerging. |
| 6 unsolved Tier 4 theorems | 3 of 6 require cases/induction (= coproduct/initial algebra concepts). Direct alignment. |
| Conceptual clusters develop hierarchically | CONFIRMS the categorical hierarchy matches natural learning order. |
| Sigmoid asymptote at 73% | New curriculum = new sigmoid. Categorical depth provides principled way to stratify. |

**Verdict**: HIGH VALUE. The categorical framing is more principled than proof-length heuristics. Specific mapping:
- Level 3 (coproducts) → `cases` on ∨ and ℕ — directly addresses t4_03, t4_06
- Level 4 (initial algebras) → `induction` — addresses t4_05, t4_08
- Level 5 (exponentials) → `by_contra`, `push_neg` — addresses t4_04

**Action**: When designing Level 3-5 curriculum, use categorical depth as the organizing principle instead of ad hoc difficulty. The existing `theorems_level1_2.json` already partially follows this (Level 1 = morphism arguments, Level 2 = composition). Extend the pattern.

---

## IR-2: Phase-Amplitude Training Decomposition (`ir-phase-amplitude-003`)

**Proposal**: Separate amplitude (mathematical knowledge base, slow) from phase (tactic selection policy, fast). simp monoculture = collapsed phase diversity. Fix by phase diversity preservation.

**Evaluation against findings**:

| Finding | Implication for this proposal |
|---|---|
| simp at 92% of proofs | DIRECTLY RELEVANT — this IS collapsed phase diversity |
| Phase locking dissolves (0.785→0.450) | The model's internal oscillatory dynamics simplify. Phase diversity loss is measurable. |
| Temperature breadth (depth ratio 0.605) | 26/50 theorems fully internalized — the amplitude is stable, only phase varies |
| Two-phase temperature dynamic | Early exploration = high phase diversity, late internalization = collapsed phase |
| d_int compression (7.01→5.20) | Manifold compression = both amplitude and phase compressing together |

**Verdict**: MEDIUM-HIGH VALUE. The amplitude/phase decomposition provides a clean theoretical frame for the simp monoculture problem. Concrete implications:
1. Level transitions in Round 3 should "reset phase" (temperature/exploration schedule) while keeping "amplitude" (LoRA weights)
2. Phase diversity can be monitored via the tactic distribution entropy — if entropy drops below threshold, inject exploration
3. The `ppl@1.4` cheap convergence signal is worth testing as a phase-collapse detector

**Action**: Add tactic distribution entropy monitoring to the Round 3 training loop. When entropy drops below `log(3)` (model using fewer than 3 tactics effectively), increase temperature or inject curriculum perturbation. This directly implements "phase diversity preservation."

---

## IR-3: Q-Analogue Curriculum Amplifier (`ir-q-amplifier-002`)

**Proposal**: Every solved theorem T at level L auto-generates T_q at level L+1 by replacing finite sets with F_q vector spaces. Four amplifiers compose fractally to generate 200+ theorems from a 50-theorem seed.

**Proposal**: Seductive but impractical for our context.

**Evaluation against findings**:

| Finding | Implication for this proposal |
|---|---|
| 44/50 theorems solved at C150 | Large seed set available for amplification |
| simp handles 92% | q-analogues would require tactics beyond simp, forcing diversity |
| Lean 4 formalization (5,471 lines) | The q-analogue theorems would need Lean 4 statements — significant manual effort |

**Verdict**: LOW VALUE for Round 3, HIGH VALUE for long-term.

Problems:
1. **Lean 4 statement generation**: q-analogues of Nat theorems require Gaussian binomial coefficients, which aren't in Lean's Mathlib as basic simp lemmas. Each theorem needs manual formalization.
2. **Verification complexity**: F_q vector spaces are graduate-level algebra. The DeepSeek-7B model won't be able to prove q-analogues without significant curriculum scaffolding.
3. **4 amplifiers composing fractally** sounds like it generates theorems the model can't solve, defeating the purpose.

The USEFUL kernel: systematic theorem generation from seed theorems. But through simpler amplifiers:
- Commutativity amplifier: every `a + b = b + a` generates `a * b = b * a`
- Universalization: every concrete proof generates a universally quantified version
- Negation: every `P → Q` generates `¬Q → ¬P`

**Action**: For Round 3, use SIMPLE amplifiers (commutativity, universalization, negation) on the existing 44 solved theorems to generate Level 3 candidates. Defer q-analogues to when the model can handle Mathlib-level algebra.

---

## Summary

| Proposal | Verdict | Round 3 Action |
|---|---|---|
| Categorical curriculum | HIGH | Use categorical depth for Level 3-5 design |
| Phase-amplitude decomposition | MEDIUM-HIGH | Add tactic entropy monitoring to training loop |
| Q-analogue amplifier | LOW (short-term) | Use simple amplifiers instead; defer q-analogues |
