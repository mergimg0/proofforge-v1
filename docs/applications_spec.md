# ProofForge Applications Specification

**Date**: 2026-03-30
**Status**: Deep spec for all 8 applications. Implementation deferred to dedicated sessions.
**Foundation**: 2% → 61% GRPO training result, infrastructure learning mechanism, three-phase dynamics.

---

## Application 6: Trajectory Length as Training Signal

**Priority**: HIGHEST — implement in round 3
**Effort**: Small (reward function modification)
**Dependencies**: None beyond current infrastructure

### Problem
Binary reward (correct/incorrect) provides no gradient once the model solves a theorem reliably. At 60%+ pass rate, most GRPO groups have 6-7/8 successes, collapsing advantage separation. The model can't improve further because it can't distinguish "good proof" from "better proof."

### Solution
Replace binary reward with efficiency-weighted reward for solved theorems:

```
reward(proof) = lean_verified(proof) * (1.0 - α * length / max_length)
```

Where α ∈ [0, 1) controls the efficiency pressure. At α=0, pure binary. At α=0.5, a 256-token proof gets reward 0.5 while a 10-token proof gets reward 0.98.

### Design Details

**Phase-in strategy**: Use binary reward for the first 50 steps (bootstrap phase). Switch to efficiency reward after the model reaches 20%+ pass rate. This prevents the efficiency signal from interfering with initial tactic learning.

**Per-theorem adaptive α**: Theorems the model has fully learned (3-temperature breadth) get high α (strong efficiency pressure). Theorems still fragile (1 temperature) keep α=0 (pure binary, don't risk destabilizing fragile capabilities). This creates a natural curriculum: master the proof first, then optimize it.

```python
def compute_reward(theorem_id, proof_text, lean_result, temp_breadth):
    if not lean_result:
        return 0.0
    if temp_breadth[theorem_id] >= 3:
        alpha = 0.5  # Strong efficiency pressure on mastered theorems
    elif temp_breadth[theorem_id] >= 2:
        alpha = 0.2  # Moderate
    else:
        alpha = 0.0  # No efficiency pressure on fragile theorems
    length_ratio = len(proof_tokens) / max_length
    return 1.0 - alpha * length_ratio
```

**Expected outcome**: Maintains gradient signal on mastered theorems (shorter proofs get higher reward even when all succeed). Pushes the model from exhaustive 256-token search toward concise, internalized proof strategies. The trajectory length ratio (low-T / high-T) should decrease toward 1.0 as proof strategies are promoted from secondary to primary modes.

**Metrics**: Track mean proof length of correct proofs per checkpoint. Track trajectory length ratio per theorem. If both decrease while pass rate holds, the efficiency signal is working.

**Risk**: The model might learn to produce SHORTER but LESS GENERALIZABLE proofs (e.g., `native_decide` which is fast but doesn't teach transferable reasoning). Mitigation: only apply efficiency pressure to theorems at temperature breadth ≥ 2, ensuring the base capability is robust before optimizing for speed.

---

## Application 1: Self-Bootstrapping Code Verification Pipeline

**Priority**: HIGH — natural extension of current work
**Effort**: Medium (new dataset + reward oracle)
**Dependencies**: Proven GRPO infrastructure learning mechanism

### Problem
Code generation models produce code that looks right but often doesn't compile, pass tests, or handle edge cases. Current training uses human feedback (expensive) or heuristic rewards (noisy).

### Solution
Apply the ProofForge pattern to code: model generates code, test suite verifies it, binary reward feeds back through GRPO. Infrastructure learning predicts the model will learn coding PATTERNS (error handling, iteration, API usage) that transfer across tasks.

### Design Details

**Reward oracle**: Replace Lean type-checker with test suite execution.
```python
def code_reward(generated_code, test_suite):
    try:
        exec_result = run_tests(generated_code, test_suite, timeout=30)
        return 1.0 if exec_result.all_passed else 0.0
    except (SyntaxError, TimeoutError, RuntimeError):
        return 0.0
```

**Dataset design**: The fractal curriculum principle applies. Design code tasks in levels:
- Level 0: Single-expression tasks (compute fibonacci(10), reverse a string)
- Level 1: Single-function tasks (implement binary search, parse CSV line)
- Level 2: Multi-function tasks (implement a class with 2-3 methods)
- Level 3: Module-level tasks (implement a small library with multiple interacting components)

Each level's solutions use infrastructure from the previous level. The model learns function composition, error handling, and API patterns through the same self-accelerating loop.

**Expected three-phase dynamics**:
- Phase 1 (disruption): Model output quality drops as GRPO reshapes the distribution
- Phase 2 (accumulation): Model learns code patterns (try/except, list comprehensions, type handling)
- Phase 3 (breakout): Pattern infrastructure reaches critical mass, multiple task categories crack simultaneously

**Temperature breadth metric**: "At how many temperatures does the model produce code that passes all tests?" Measures how deeply the model has internalized the coding pattern vs. getting lucky through sampling.

**Key difference from ProofForge**: Test suites are imperfect verifiers (partial coverage, flaky tests). Lean is a perfect verifier. The infrastructure learning mechanism depends on reward quality. Noisy rewards from partial test suites may degrade the signal. Mitigation: use comprehensive test suites with edge cases, or add static analysis as a secondary verifier.

**Implementation path**:
1. Curate 100-200 Python coding tasks with test suites (can use existing benchmarks like HumanEval, MBPP)
2. Deploy same GRPO pipeline with test execution as reward
3. Track pass rate curve — look for three-phase dynamics
4. Compare infrastructure learning rate with theorem proving

---

## Application 8: Adaptive Training Controller

**Priority**: HIGH — automates training decisions
**Effort**: Medium (monitoring + decision logic)
**Dependencies**: Three-phase model, running reward average as leading indicator

### Problem
Current GRPO training requires human monitoring to detect phase transitions, reward saturation, and capability plateaus. Intervention decisions (change learning rate, expand dataset, switch reward) are made manually and often too late.

### Solution
An automated controller that monitors training metrics in real time and makes intervention decisions based on the three-phase model.

### Design Details

**Architecture**: A monitoring daemon that polls the training log every N steps and computes diagnostic metrics.

```python
class TrainingController:
    def __init__(self, config):
        self.phase = "disruption"  # Current detected phase
        self.reward_window = deque(maxlen=20)  # Rolling reward average
        self.pass_rate_history = []
        self.intervention_log = []

    def update(self, step, reward, pass_rate=None):
        self.reward_window.append(reward)
        avg_reward = np.mean(self.reward_window)

        if pass_rate is not None:
            self.pass_rate_history.append((step, pass_rate))

        # Phase detection
        if self.phase == "disruption":
            if avg_reward > 0.15 and self._reward_trend() > 0:
                self.phase = "accumulation"
                self.log("Phase transition: disruption -> accumulation")

        elif self.phase == "accumulation":
            if avg_reward > 0.25:
                self.phase = "breakout"
                self.log("Phase transition: accumulation -> breakout")

        elif self.phase == "breakout":
            if avg_reward > 0.8:
                self.phase = "saturation"
                self.log("WARNING: Gradient saturation detected")
                self.recommend("Expand theorem set or switch to efficiency reward")

        return self.get_recommendations()
```

**Intervention rules**:

| Phase | Diagnostic | Intervention |
|-------|-----------|-------------|
| Disruption | Pass rate drops, reward near 0 | WAIT. Log "disruption phase, patience required." |
| Accumulation | Reward climbing, pass rate flat | Log "infrastructure building." Estimate breakout ETA from reward slope. |
| Breakout | Reward 0.2-0.8, pass rate rising | OPTIMAL ZONE. No intervention. Log progress. |
| Saturation | Reward > 0.8 consistently | Switch to efficiency reward OR expand dataset OR increase theorem difficulty |
| Stagnation | Reward flat < 0.15 for 50+ steps | Increase group_size OR add shaped reward OR reduce learning rate |

**Leading indicator**: The running reward average leads the pass rate by ~10-20 steps. Reward average started climbing at step 35; pass rate broke out at step 50. The controller uses this lead time to predict phase transitions before they happen.

**Integration with expanding ring**: When the controller detects saturation, it automatically activates the expanding ring curriculum — adding harder theorems from the Level 1-2 set to keep theorems in the frontier zone.

**Implementation path**:
1. Build as a separate Python script that tails the training log
2. Test on the current run's log file (replay historical data)
3. Validate that phase detection matches our manual observations
4. Integrate into round 3 as a monitoring daemon

---

## Application 2: Infrastructure-Aware GRPO

**Priority**: MEDIUM-HIGH — research contribution
**Effort**: Medium (tactic tracking + modified sampling)
**Dependencies**: Infrastructure learning model, tactic classification

### Problem
Standard GRPO samples theorems uniformly. But theorems that share proof infrastructure should be trained together because gradient from one transfers to the other. Uniform sampling wastes steps on theorems far from the model's current infrastructure frontier.

### Solution
Track which tactics appear in successful proofs. Select training theorems that require infrastructure the model has PARTIALLY learned — theorems where some required tactics are mastered and others are near the frontier.

### Design Details

**Tactic registry**: Maintain a running record of which tactics produce verified proofs.

```python
class TacticRegistry:
    def __init__(self):
        self.tactic_success = defaultdict(lambda: {'attempts': 0, 'successes': 0})

    def record(self, proof_text, lean_result):
        tactics = extract_tactics(proof_text)
        for tactic in tactics:
            self.tactic_success[tactic]['attempts'] += 1
            if lean_result:
                self.tactic_success[tactic]['successes'] += 1

    def mastery(self, tactic):
        s = self.tactic_success[tactic]
        if s['attempts'] == 0:
            return 0.0
        return s['successes'] / s['attempts']
```

**Infrastructure-aware sampling**: For each theorem, estimate which tactics it likely needs (from known proofs or from the model's failed attempts). Score theorems by partial mastery — high score when some required tactics are mastered and others aren't.

```python
def select_training_theorem(theorems, registry):
    scores = []
    for thm in theorems:
        required = estimate_required_tactics(thm)
        masteries = [registry.mastery(t) for t in required]
        # Peak score when mix of mastered (>0.5) and unmastered (<0.2)
        score = np.std(masteries) * np.mean(masteries)
        scores.append(score)
    # Sample proportional to score
    probs = np.array(scores) / sum(scores)
    return np.random.choice(theorems, p=probs)
```

**Expected benefit**: Accelerates the self-accelerating loop by deliberately presenting theorems at the infrastructure frontier. Instead of waiting for random sampling to hit frontier theorems, the system actively seeks them out.

**Validation**: Compare pass rate curves between uniform sampling (current) and infrastructure-aware sampling. The infrastructure-aware version should show faster breakout (shorter Phase 2) and steeper improvement (stronger Phase 3).

---

## Application 5: Three-Phase Meta-Protocol

**Priority**: MEDIUM — methodology paper
**Effort**: Low (framing + validation on existing data)
**Dependencies**: Training curve data, infrastructure learning model

### Problem
GRPO training with binary verification exhibits a specific three-phase pattern that practitioners don't expect. The disruption phase looks like training failure, leading to premature termination. The accumulation phase looks like a plateau, suggesting the approach doesn't work. Only patience through both reveals the breakout.

### Solution
Document the three-phase protocol as a transferable methodology with diagnostic criteria for each phase.

### Specification

**Title**: "Three-Phase GRPO Training: How Binary Verification Produces Convex Capability Improvement Through Infrastructure Learning"

**Content**:
1. The three phases: disruption, accumulation, breakout
2. Diagnostic metrics for each phase (running reward average, pass rate, failure mode taxonomy)
3. When to wait (disruption), when to expect breakthrough (accumulation with rising reward average), when to expand (saturation)
4. The infrastructure learning mechanism: why improvement is convex, not concave
5. The temperature breadth metric: measuring internalization depth
6. Case study: theorem proving (this project's data)
7. Predicted applicability: any domain with formal verification as reward

**Key claim**: The disruption phase is not failure. The accumulation plateau is not a ceiling. The running reward average is the leading indicator. If it's climbing while pass rate is flat, infrastructure is building and breakout is coming.

**Evidence**: The complete pass rate curve (2% → 8% plateau → 23% breakout → 43% → 61%) with per-step reward data showing the leading indicator pattern.

---

## Application 7: Scaling Law Prediction

**Priority**: MEDIUM — quantitative planning tool
**Effort**: Low-Medium (curve fitting + validation)
**Dependencies**: Complete training curve (10 checkpoints)

### Problem
How many GRPO training steps are needed to reach a target pass rate? Currently there's no way to predict this without running the full training.

### Solution
Fit a parametric model to the pass rate curve. Extrapolate. Validate on held-out checkpoints.

### Design Details

**Model**: The pass rate curve has three regimes:
1. Initial rapid rise (steps 0-5): formatting improvement
2. Plateau (steps 5-30): infrastructure accumulation
3. Sigmoid rise (steps 30+): infrastructure-driven improvement

A three-parameter sigmoid fits regime 3:
```
pass_rate(step) = L / (1 + exp(-k * (step - step_0)))
```
Where L is the ceiling, k is the growth rate, step_0 is the inflection point.

**Fitting procedure**:
1. Use checkpoints 50-100 to fit the sigmoid parameters
2. Extrapolate to steps 150-200
3. Compare against actual checkpoint 150 and 200 data
4. Report prediction error

**Transfer test**: Train on 50 theorems, measure scaling law. Evaluate on a different 50 theorems. Does the same sigmoid shape appear with shifted parameters? If yes, the scaling law transfers and can predict training time for new theorem sets.

**Practical output**: "For this model and theorem difficulty distribution, reaching X% pass rate requires approximately N GRPO training steps." This is a planning tool for anyone building a similar system.

---

## Application 4: Formal Verification as Universal Reward Oracle

**Priority**: HIGHEST LONG-TERM — paradigm shift
**Effort**: Large (domain-specific reward oracles)
**Dependencies**: Proven pattern from theorem proving, infrastructure learning mechanism

### Problem
RL training for code/specification generation currently relies on noisy rewards (human feedback, heuristic rules, learned reward models). Formal verifiers provide PERFECT binary reward but are seen as too sparse for training.

### Solution
Our result proves that binary formal verification IS sufficient for GRPO training, despite being sparse, IF the domain has shared infrastructure. The infrastructure learning mechanism converts sparse binary signal into effective dense learning.

### Domain Applications

**Smart Contracts (Solidity + Certora/Slither)**:
- Model generates Solidity code
- Formal verifier checks for reentrancy, overflow, access control violations
- Binary reward: no vulnerabilities found = 1, any vulnerability = 0
- Infrastructure: the model learns Solidity safety patterns (checks-effects-interactions, SafeMath, access modifiers)
- Expected: three-phase dynamics with breakout when safety pattern infrastructure reaches critical mass

**Protocol Specifications (TLA+ + TLC model checker)**:
- Model generates TLA+ specifications
- Model checker verifies safety/liveness properties
- Binary reward: all properties hold = 1, any violation = 0
- Infrastructure: the model learns specification patterns (invariants, fairness, stuttering)

**Hardware Design (SystemVerilog + formal verification)**:
- Model generates RTL code
- Formal verification tool checks assertions
- Binary reward: all assertions pass = 1, any failure = 0
- Infrastructure: timing patterns, FSM structure, pipeline hazard avoidance

**Database Queries (SQL + query analyzer)**:
- Model generates SQL queries
- Analyzer checks correctness against schema + test data
- Binary reward: correct results = 1, any mismatch = 0
- Infrastructure: JOIN patterns, aggregation, subquery structure

### Design Principle
Each domain requires:
1. A formal verifier that provides perfect binary reward
2. A task set where solutions share infrastructure (common patterns/idioms)
3. A difficulty gradient from easy to hard tasks
4. The three-phase training protocol (patience through disruption + accumulation)

### Key Prediction
The infrastructure learning mechanism should produce convex improvement curves in ANY domain where:
- Solutions share structural patterns (infrastructure)
- A perfect binary verifier exists
- The task set spans a difficulty gradient

If this prediction holds across 3+ domains, it establishes formal-verification-as-reward as a general training paradigm, not just a theorem-proving trick.

### Validation Plan
1. Validate on code generation (Application 1) — closest to theorem proving
2. Validate on SQL generation — different domain, same pattern
3. If both show three-phase dynamics + convex improvement, the paradigm is general

---

## Application 3: Claude Code Self-Improvement Loop

**Priority**: LOW (niche, requires additional infrastructure)
**Effort**: Large
**Dependencies**: Logging infrastructure, routing model

### Problem
Claude Code generates solutions of varying quality. There's no systematic way to learn from which approaches work and which don't for specific task types.

### Solution
Log Claude Code interactions with binary outcomes (solution worked / didn't). Train a lightweight routing model to predict task difficulty and recommend approach strategies (more context, decomposed subtasks, different prompting). The temperature breadth concept applies: tasks Claude Code solves reliably at all settings are deeply competent; tasks requiring specific conditions need flagging.

### Design Details

**Data collection**: Log each Claude Code task with:
- Task type (code generation, debugging, refactoring, explanation)
- Approach used (direct, decomposed, iterative)
- Outcome (succeeded, failed, partial)
- Temperature/sampling settings

**Routing model**: Small classifier (can be logistic regression or small neural net) that predicts P(success | task_features, approach) and recommends the approach with highest expected success.

**Not a training signal for Claude Code itself**: This doesn't improve the base model. It improves task ROUTING — which tasks to attempt directly, which to decompose, which to flag for human review.

**This is the weakest application** because it requires substantial logging infrastructure and the routing model's value depends on having enough historical data to learn patterns. Defer until Applications 1, 4, 6, 8 are proven.

---

## Cross-Application Dependencies

```
App 6 (efficiency reward) ──→ App 8 (adaptive controller uses efficiency switch)
                            ↗
App 2 (infra-aware GRPO) ──→ App 5 (three-phase protocol documents it)
                            ↘
App 1 (code verification) ──→ App 4 (universal formal verification)
                            ↗
App 7 (scaling law) ────────→ App 8 (controller uses scaling prediction)
```

**Critical path**: App 6 → Round 3 → App 8 → App 5 (paper) → App 1 → App 4

**The efficiency reward (App 6) unblocks everything** because it solves the gradient saturation problem that limits all subsequent training runs.
