# ForgeStream: Live Meeting Intelligence with SOS-Governed Autonomous Agent Swarm

**Date:** 2026-03-27
**Status:** Approved
**Repository:** Separate from ProofForge (new repo: `forgestream`)
**Language:** Python (primary), Claude Code CLI (agent execution)

---

## 1. Vision

ForgeStream is a live meeting intelligence system that uses the Gemini Live API to process audio/video from expert meetings in real-time, extracting knowledge into an append-only event log (ECEF pattern), and autonomously dispatching Claude Code agent swarms to research, scaffold, and build software during the meeting itself.

The system is governed by the SOS (Self-Optimizing Systems) convergence framework — the same framework proven in 4,229 lines of machine-verified Lean 4 in the companion ProofForge project. The three SOS axioms (Monotone Improvement, Bounded Step, Constraint Preservation) are enforced as runtime invariants, not just theoretical properties. This gives ForgeStream something no other autonomous agent system has: a formal convergence certificate.

The system improves across meetings. Each meeting adds to a persistent knowledge graph. The evaluator trajectory converges. The trust region widens as competence is demonstrated. The system earns autonomy through proven performance.

## 2. System Architecture

The core principle: **the append-only event log IS the system**. Every component is either a writer or a reader of events. There is no other coordination mechanism. This makes the architecture isomorphic to the ECEF SOS, which means the convergence theorems apply to the running system.

```
                        +-------------------------+
                        |   LIVE MEETING           |
                        |   (audio / video / text) |
                        +-----------+-------------+
                                    |
                                    v
                    +-------------------------------+
                    |   GEMINI LIVE API (WebSocket)  |
                    |   Processes audio/video/screen |
                    |   Extracts structured claims   |
                    +---------------+---------------+
                                    | writes
                                    v
    +---------------------------------------------------------------+
    |                                                               |
    |              THE ECEF EVENT LOG                                |
    |              (append-only, the single source of truth)        |
    |                                                               |
    |  SOS INVARIANTS (checked on every write):                     |
    |  - E(pi_{n+1}) >= E(pi_n)     -- Monotone Improvement        |
    |  - ||delta(pi)|| <= epsilon   -- Bounded Step                 |
    |  - C(pi_n) => C(pi_{n+1})    -- Constraint Preservation      |
    |                                                               |
    +---+--------+--------+--------+--------+---------------------+
        |        |        |        |        |
  reads |  reads |  reads |  reads |  reads |
        v        v        v        v        v
    +-------+ +------+ +------+ +-----+ +-----------+
    |RESEARCH| |SCAFF | |SYNTH | | TUI | | DASHBOARD |
    | SWARM  | |SWARM | |ENGINE| |     | |   (web)   |
    |        | |      | |      | |     | |           |
    | Claude | |Claude| |Class-| |tmux | | Knowledge |
    | Code   | |Code  | |ifier | |term | | graph viz |
    | CLIs   | |in git| |+prio | |     | | + queue   |
    |        | |work- | |queue | |     | |           |
    | writes:| |trees | |      | |     | |           |
    |verified| |      | |writes| |     | |           |
    |_finding| |writes| |sugg- | |     | |           |
    |        | |artif.| |estion| |     | |           |
    +--------+ +------+ +------+ +-----+ +-----------+
```

Data flow is unidirectional: Meeting -> Gemini -> Event Log -> Agents -> Event Log -> UI. Agents never talk to each other directly. They communicate exclusively through events.

### Language Split

- Event system, Gemini integration, synthesis engine, orchestrator: **Python** (Gemini SDK is Python-native, ADK is Python)
- Code scaffolding agents: **Claude Code CLI** (spawned as subprocesses in tmux/worktrees)
- Research agents: **Claude Code CLI** (with web search, MCP tools)
- Dashboard: **Python + FastAPI + htmx/D3.js**
- TUI: **Python + textual**

### Relationship to ProofForge

ProofForge's Rust crates remain in their own repository. ForgeStream connects to ProofForge via MCP tools and the `TrainingBridge` REST API (port 8420) when meeting content involves ProofForge-related work. The Rust crates are tools in the agent toolbox, not the orchestration layer.

### LLM Topology

- **Gemini 2.5 Flash**: Meeting intelligence (Live API audio/video processing). Fixed.
- **Claude via Claude Code CLI**: Code scaffolding and deep research. Not the Claude API — Claude Code subscription. Each agent is a Claude Code instance with full tool access.
- **Model-agnostic layer**: The agent spawner and prompt templates are model-agnostic. Claude Code is the default, but the architecture supports hot-swapping to other backends (DeepSeek for proof generation, local models for low-latency tasks) without structural changes.

## 3. The Event System

### Event Envelope

Every event has a common structure:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Globally unique |
| `session_id` | UUID | Groups events within a meeting |
| `timestamp` | ISO-8601 (microsecond) | When the event was created |
| `event_type` | enum | See event types below |
| `parent_id` | UUID or null | Causal parent event |
| `branch_id` | UUID | Conversation branch this belongs to |
| `author` | string | Who created this event (gemini, research_agent_3, user, etc.) |
| `evaluator` | float | E(pi) at time of write |
| `payload` | JSON | Type-specific data |
| `degradation_flag` | boolean | True if this event caused E to decrease |
| `trust_region_ok` | boolean | True if this event is within the trust region |

**IMMUTABLE AFTER WRITE. No updates. No deletes. Ever.**

### Event Types

| Type | Written by | Payload | Purpose |
|------|-----------|---------|---------|
| `claim` | Gemini | `{text, speaker, confidence, audio_timestamp, tone_markers}` | Raw knowledge extracted from meeting audio |
| `contradiction` | Synthesis Engine | `{claim_a_id, claim_b_id, explanation, resolution_hint}` | Two claims that conflict |
| `requirement` | Synthesis Engine | `{description, domain, complexity_estimate, linked_claims[]}` | Actionable thing to build |
| `verified_finding` | Research Swarm | `{query, finding, sources[], verification_chain, confidence}` | Research result with citation trail |
| `artifact` | Scaffold Swarm | `{worktree_path, branch_name, files_created[], compiles, tests_pass}` | Code scaffold output |
| `suggestion` | Synthesis Engine | `{text, priority, category, linked_events[], decay_rate}` | Surfaced to user in priority queue |
| `branch_point` | Synthesis Engine | `{parent_branch_id, trigger_event_id, potential_score, description}` | Conversation diverged |
| `seed` | Synthesis Engine | `{cluster_events[], novelty_score, domain_guess, description}` | New root topic detected |
| `evaluator_snapshot` | SOS Governor | `{E_micro, E_meso, E_macro, axiom_checks}` | Periodic SOS health check |
| `mode_switch` | User or System | `{from_mode, to_mode, reason}` | Session mode changed |
| `merge` | User or System | `{branch_id, into_branch_id, retained_events[], summary}` | Branch merged back |
| `meeting_summary` | Synthesis Engine | `{...comprehensive summary...}` | Post-meeting synthesis |

### PostgreSQL Schema

```sql
CREATE TABLE events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type  TEXT NOT NULL,
    parent_id   UUID REFERENCES events(id),
    branch_id   UUID NOT NULL,
    author      TEXT NOT NULL,
    evaluator   FLOAT NOT NULL,
    payload     JSONB NOT NULL,
    degradation_flag  BOOLEAN DEFAULT FALSE,
    trust_region_ok   BOOLEAN DEFAULT TRUE
);

-- Append-only enforcement
CREATE RULE no_updates AS ON UPDATE TO events DO INSTEAD NOTHING;
CREATE RULE no_deletes AS ON DELETE TO events DO INSTEAD NOTHING;

-- Indexes
CREATE INDEX idx_events_session ON events(session_id, timestamp);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_branch ON events(branch_id);
CREATE INDEX idx_events_parent ON events(parent_id);
```

Real-time subscriptions use PostgreSQL LISTEN/NOTIFY on channels per event_type.

### The Conversation Tree

The `branch_id` and `parent_id` fields create a tree structure:

```
main --*--*--*--*--*--*--*--*--*--*--*--
                  |              |
                  |              +-- branch-2 --*--*--*-- (merged back)
                  |
                  +-- branch-1 --*--*--*--*--*-- (still open)
                                       |
                                       +-- seed-1 --*--* (new root detected)
```

### SOS Mapping

**Axiom 1 (Monotone Improvement):** The evaluator E is computed over a sliding window, not per event. Individual events can decrease E, but the trajectory must be non-decreasing. Three consecutive declining windows trigger a DEGRADATION ALERT and trust region contraction.

**Axiom 2 (Bounded Step):** The trust region epsilon governs semantic drift (cosine distance of new claims from branch centroid), resource delta (change in active agents), and scaffold scope (files created). Events outside the trust region are still written (append-only) but tagged `trust_region_ok = false`.

**Axiom 3 (Constraint Preservation):** Verified claims are immutable (structural guarantee from append-only log). Knowledge graph type structure is monotone. Compilation invariants are preserved (new scaffolds in separate worktrees). Source chains are required for all verified findings.

### Evaluator Function (Pluggable)

```python
def compute_evaluator(event_log: EventLog, window: TimeWindow) -> float:
    """
    Pluggable evaluator. Replace this function to change what
    'improvement' means. The SOS convergence theorems hold for
    ANY evaluator that satisfies the three axioms.
    """
    return (
        w1 * knowledge_density(window)
        + w2 * verification_rate(window)
        + w3 * scaffold_success_rate(window)
        + w4 * suggestion_uptake(window)
    )
```

This is an explicit extension point. Future evaluator equations (including the autoresearch RL equation) plug in here.

## 4. Knowledge Graph

The knowledge graph is a **materialized view** over the event log. It has three entity types and four edge types.

### Entities

- **Concept**: Named knowledge extracted from claims. Fields: name, domain, confidence, verified (bool), source_events[].
- **Requirement**: Actionable thing to build, derived from claims. Fields: description, domain, complexity_estimate, status (detected/scaffolding/built/verified), linked_claims[].
- **Artifact**: Code scaffold output. Fields: path, branch, compiles, tests_pass, linked_requirements[].

### Edges

- `concept --relates_to--> concept` (weighted by co-occurrence)
- `concept --supports--> requirement` (this knowledge implies this need)
- `requirement --fulfilled_by--> artifact` (this code satisfies this need)
- `concept --contradicts--> concept` (flagged conflict)

### Event Sourcing

The graph is rebuilt from events on startup. No separate persistence. The event log IS the source of truth. If the graph corrupts, replay the log and reconstruct.

## 5. Gemini Live API Integration

### Architecture

The Gemini Live API client connects via WebSocket and streams audio (PCM 16kHz mono) from the meeting microphone. Optionally streams video frames from screen share.

Gemini 2.5 Flash (native audio model) processes the stream with a system instruction tuned for ECEF knowledge extraction. The system instruction varies by mode:

| Mode | Emphasis |
|------|----------|
| **A (Extract)** | Focus on what the expert needs built. Extract requirements, constraints, tech preferences. |
| **B (Collaborative)** | Design discussion. Extract architectural decisions, trade-offs, agreements and disagreements. |
| **C (Knowledge)** | Expertise capture. Extract domain knowledge, mental models, heuristics, tacit knowledge. |

### What Native Audio Captures

| Signal | Meaning | Effect |
|--------|---------|--------|
| Hesitation | Lower confidence | `confidence` reduced 0.1-0.3 |
| Emphasis / pitch change | Speaker believes this is important | `tone_markers: ["emphasis"]`, boosts suggestion priority |
| Backtracking | Previous claim may be wrong | Triggers contradiction check |
| Excitement | Breakthrough moment | `tone_markers: ["excitement"]`, flag as potential seed |
| Long pause after question | Expert thinking deeply | System holds suggestions |
| Crosstalk / interruption | Disagreement | Flag for synthesis to analyze both positions |

### Context Injection

Every 10 minutes, the system injects a summary of the current knowledge graph state into Gemini's context to fight context decay in long meetings.

### Cost Model

| Component | Per hour |
|-----------|----------|
| Gemini 2.5 Flash Live API (audio) | ~$1-2 |
| Gemini 2.5 Flash Live API (audio + video) | ~$3-5 |
| Context injections | ~$0.10 |
| Audio copilot (if enabled) | ~$0.50 |
| **Total per meeting** | **$2-6** |

## 6. Claude Code Agent Orchestration

### Agent Spawner

A Python process subscribes to the event log and manages Claude Code CLI instances. Two agent types:

**Research Agents** investigate, verify, and synthesize knowledge:
- Triggered by: high-novelty claims, new branches, "delve_deeper" suggestions
- Execution: `claude -p '{prompt}' --allowedTools 'WebSearch,WebFetch,Read,Grep,Glob'` in a tmux session
- Output: parsed JSON written as `verified_finding` events
- Timeout: 5 minutes

**Scaffold Agents** generate code, architecture docs, and prototypes:
- Triggered by: detected requirements, accumulated related requirements
- Execution: Claude Code CLI in an isolated git worktree
- Output: parsed JSON written as `artifact` events, open questions as `suggestion` events
- Timeout: 15 minutes

### Agent Lifecycle

```
requirement event
    |
    v
SPAWN DECISION (check trust region, agent count, branch load, cooldown)
    |
    v (approved)
PROVISIONING (git worktree for scaffold, tmux session, prompt build)
    |
    v
RUNNING (monitor via tmux capture-pane, enforce timeout)
    |
    +-- success --> REPORT (write event to log) --> CLEANUP
    +-- timeout/error --> RETRY (1x) or ABANDON (write failure event) --> CLEANUP
```

### Worktree Management

Each scaffold agent gets its own git worktree branching from main:

```
~/projects/forgestream/              <-- main repo (untouched during meeting)
~/projects/forgestream-sc-001/       <-- scaffold: auth-service
~/projects/forgestream-sc-002/       <-- scaffold: data-pipeline
```

Post-meeting, successful scaffolds can be merged, kept as reference branches, or fed into the next meeting.

### Resource Limits (Trust Region Governed)

| Parameter | Initial (Meeting 1) | Earned (Meeting 10+) |
|-----------|---------------------|----------------------|
| `max_concurrent_research` | 2 | 5 |
| `max_concurrent_scaffold` | 2 | 6 |
| `spawn_cooldown` | 60s | 10s |
| `scaffold_timeout` | 10m | 20m |
| `auto_spawn` | false | true |
| `branch_auto_allocate` | false | true |

### Cross-Agent Knowledge Flow

Agents don't communicate directly. They benefit from each other through the event log:

1. Gemini extracts claim -> claim event
2. Research agent finds supporting evidence -> verified_finding event
3. Scaffold agent reads verified_finding, builds informed implementation -> artifact event
4. Scaffold reports open question -> suggestion event
5. User asks expert in meeting -> Gemini extracts answer -> claim event
6. Cycle continues. Each iteration refines. E increases.

## 7. User Interfaces

### Terminal TUI (Primary)

Built with Python `textual`. Keyboard-driven. Used during meetings.

**Panels:**

| Panel | Content | Update frequency |
|-------|---------|-----------------|
| Header | Session metadata, E(pi), trust level | Every event |
| Live Feed | Scrolling claims with linkage annotations | Per claim |
| Suggestion Queue | Priority-sorted, color-coded | Recalculated per event |
| Branches | Tree with potential/momentum/ROI | Per branch_point or claim |
| Agents | Status of running Claude Code instances | Polled every 5s |
| Hotkeys | Context-sensitive keyboard shortcuts | Static |

**Key interactions:**

| Key | Action |
|-----|--------|
| `m` | Cycle mode (A -> B -> C -> A) |
| `b` + arrow | Focus a branch |
| `s` | Dismiss top suggestion |
| `a` + number | Agent detail panel |
| `p` / `r` | Pause / resume agent spawning |
| `q` | Quiet mode (minimal bar) |
| `/scaffold "..."` | Manual scaffold trigger |
| `/research "..."` | Manual research trigger |
| `/merge N` | Merge branch N into main |
| `/seed N` | Promote seed to branch |
| `Space` | Bookmark current moment |
| `Esc` | Reset to main view |

**Quiet mode** collapses to a single status bar. Only critical suggestions force a visual notification.

**Suggestion priority levels:**

| Level | Score | Display |
|-------|-------|---------|
| `critical` | 0.9-1.0 | Red, top of queue |
| `strategic` | 0.7-0.9 | Amber |
| `delve_deeper` | 0.5-0.7 | Blue |
| `good_to_probe` | 0.3-0.5 | Green |
| `nice_to_know` | 0.0-0.3 | Grey, bottom |

Suggestion priority score:

```
priority = relevance_to_current_branch
         * novelty
         * actionability
         * confidence
         * (1 - time_decay)
         + contradiction_boost    # +0.5 if relates to contradiction
         + convergence_boost      # +0.3 if connects unrelated branches
```

### Web Dashboard (Secondary)

FastAPI backend + D3.js/htmx frontend. Read-only view of the event log.

**Components:**

| Component | Purpose | Interaction |
|-----------|---------|-------------|
| Knowledge Graph | Force-directed concept/requirement/artifact visualization | Click nodes, filter by branch |
| Evaluator Trajectory | Real-time E(pi) at micro/meso/macro levels | Hover for breakdown |
| Meeting Timeline | Horizontal timeline with key event markers | Click to jump, scrub to replay |
| Artifact Tracker | Scaffold worktrees with compile/test status | Click to view diff/design |
| SOS Convergence Panel | Axiom status, trust region, predicted convergence | Monitor system health |

**Cross-meeting view** (between meetings): merged knowledge graph, E_macro trajectory across meetings, transfer map, seed garden.

The dashboard creates no events. It has no write access. Unidirectional data flow preserved.

## 8. SOS Runtime Governance

### The SOS Governor

A dedicated Python process that monitors the event log and enforces the three axioms.

**Axiom 1 (Monotone Improvement):**
- Evaluator computed over sliding window of ~20 events
- If E(window_n+1) < E(window_n) for 3 consecutive windows -> DEGRADATION ALERT
- Trust region contracts, more actions require confirmation

**Axiom 2 (Bounded Step):**
- Semantic drift: cosine_distance(new_claim, branch_centroid) <= epsilon_semantic
- Resource delta: change in active agents <= epsilon_resource
- Scaffold scope: files_created <= epsilon_scope
- Violations tagged `trust_region_ok = false`, deprioritized by synthesis engine

**Axiom 3 (Constraint Preservation):**
- Verified claims immutable (append-only log)
- Knowledge graph type structure monotone
- Compilation invariants preserved (separate worktrees)
- Source chains required for verified findings
- Violations trigger CONSTRAINT BREACH: hard alert, trust region contracts sharply, auto-spawn paused

### Trust Region Dynamics

```
epsilon(t) = epsilon_base * competence_multiplier(t) * stability_factor(t)

epsilon_base = 0.3

competence_multiplier = sigmoid(
    alpha * consecutive_meetings_with_improving_E_macro
    - beta * total_axiom_violations
)
range: [0.5, 3.0]

stability_factor = 1.0 - volatility(E_micro, last_N_events)
range: [0.3, 1.0]
```

The system earns autonomy through demonstrated convergence.

### Evaluator Hierarchy

| Level | Scope | Update frequency | Affects |
|-------|-------|-----------------|---------|
| **Micro** | Per-event, within meeting | Every event | Suggestion queue, agent spawn urgency |
| **Meso** | Per-meeting | Every 5 min + meeting end | Post-meeting report, cross-meeting comparison |
| **Macro** | Cross-meeting trajectory | Post-meeting | Trust region, resource limits, autonomy level |

The morphism between levels (Theorem 3.15 applied): if E_micro converges within a meeting, E_meso converges too. If E_meso converges across meetings, E_macro converges. The composition gives per-event convergence implying system-level convergence (Theorem 3.17 rate transfer).

## 9. Self-Improvement Mechanisms

### 1. Evaluator Weight Tuning (GRPO-style)

After each meeting:
1. Compute E_meso with current weights
2. Generate N perturbed weight vectors
3. Retroactively compute E_meso for each perturbation against the meeting's event log
4. Rank by correlation with post-meeting human feedback
5. Update weights toward best-performing perturbation

GRPO applied to the evaluator itself. The evaluator learns what "good" means.

### 2. Prompt Evolution

After each meeting, for each agent that ran:
1. Collect: prompt used, output produced, human action (used/modified/discarded)
2. Score output usefulness
3. Good prompts are preserved, bad prompts adjusted
4. Next meeting composes from best-performing variants + new context

Prompt quality is monotone non-decreasing (ECEF applied to prompts).

### 3. Domain Model Accumulation

The knowledge graph persists across meetings. Each meeting adds. Cross-meeting knowledge reuse is tracked. When knowledge from Meeting N is reused in Meeting M, E_macro increases (the system is learning).

### 4. Branch Pattern Learning

Over many meetings, the system learns:
- Which branch patterns lead to high-value artifacts -> auto-promote similar
- Which seeds became productive -> detect similar seeds earlier
- Which branches had low ROI -> deprioritize similar divergences

The branch calculus incorporates historical outcomes as a prior.

### Post-Meeting Synthesis

When a meeting ends:
1. SOS Governor emits final evaluator_snapshot
2. Synthesis Engine generates meeting report (knowledge summary, requirements, artifacts, branches, seeds, contradictions, carry-forward suggestions)
3. Evaluator weight tuning runs
4. Prompt evolution runs
5. Cross-session indexes rebuilt
6. Report written to event log + `docs/meetings/YYYY-MM-DD-{slug}.md`
7. Human review queue: proof obligations, unresolved contradictions, seeds awaiting promotion, scaffolds awaiting review

## 10. Meeting Modes

Three modes, switchable at any time during a meeting:

| Mode | Name | Focus | Typical meeting |
|------|------|-------|----------------|
| **A** | Extract | What the expert needs built | Domain expert describing systems |
| **B** | Collaborative | Architectural decisions and trade-offs | Design session with engineers |
| **C** | Knowledge | Domain expertise, mental models, tacit knowledge | Knowledge extraction interviews |

Mode switching is triggered by the user (`m` key in TUI) or detected by the Synthesis Engine (conversation pattern shifts). Mode changes are recorded as `mode_switch` events. The Gemini system instruction updates to match the new mode.

## 11. Audio Copilot (Experimental)

A second Gemini Live API session generating audio output to the user via earpiece:
- Only speaks during detected pauses (>2 seconds silence)
- Maximum 10 words per injection
- Minimum 30 second cooldown between injections
- Only surfaces `critical` and `strategic` priority suggestions
- Conservative by design: interrupting a meeting is worse than a missed suggestion
- Trust region governs cooldown: consistent uptake shortens it

## 12. Branch Calculus

For each active branch:

```
branch_potential = sum(event novelty * relevance) / branch_age
branch_momentum  = d(potential)/dt
branch_cost      = active_agents_on_branch * time_on_branch
branch_roi       = potential / cost
```

The TUI displays these compactly. When `branch_roi` drops below threshold, the engine suggests merging. When `branch_momentum` is strongly positive, it suggests allocating more agents.

**Seed detection:** When the Synthesis Engine finds a cluster of concepts disconnected from the main knowledge graph (below a similarity threshold), it emits a `seed` event. Seeds can be promoted to full branches or archived.

## 13. Orchestrator Architecture

The orchestrator uses a hybrid topology (Option C) with a core process and fault-isolated child workers.

### Core Principle: Governor as Immune System

The Governor is a post-write observer, not a pre-write gate. Events are NEVER rejected based on semantic content. The append-only log captures everything. The Governor observes the trajectory and adjusts system behavior in response.

- **Pre-write**: Structural validation only (required fields, source chain for verified_findings). Microseconds. Never blocks valid data.
- **Post-write**: Evaluator computation, axiom trajectory checking, trust region adjustment. Async, non-blocking.

### Runtime Topology

```
Core Process (asyncio):
  - Event Bus: in-memory fanout of new events to local subscribers
  - TUI: textual app, subscribes to event bus for instant updates
  - Governor: post-write observer, adjusts trust region
  - Structural Validator: pre-write, trivially thin

Child Processes (fault-isolated, coordinate via PostgreSQL):
  - gemini-worker: audio -> claim events
  - synthesis-worker: claims -> requirements/contradictions/branches/seeds
  - agent-worker: requirements -> Claude Code spawning -> artifacts
  - dashboard-worker: events -> web UI (FastAPI)
```

### Event Lifecycle

1. Worker produces event
2. Core process receives (in-memory bus or NOTIFY)
3. Pre-write: structural check (< 1ms, never rejects valid data)
4. Write to PostgreSQL (append-only, NOTIFY fires)
5. Post-write (async, non-blocking):
   - Governor: compute evaluator, check axioms, tag event
   - TUI: update display immediately
   - Other workers: receive via their own LISTEN

### Fault Isolation

If a child worker crashes, the supervisor restarts it. The worker replays events from the log to rebuild state (event sourcing). Other workers are unaffected. The TUI stays responsive. The meeting continues.

## 14. Extension Points

The architecture is designed with explicit extension points:

| Extension Point | What it accepts | When it matters |
|----------------|----------------|-----------------|
| **Evaluator function** | Any function `EventLog x TimeWindow -> float` satisfying the SOS axioms | Future autoresearch RL equation |
| **Event types** | New event type definitions with typed payloads | New agent types, new knowledge primitives |
| **Agent templates** | New prompt templates for research/scaffold/any role | Specialized domain agents |
| **MCP tools** | New tool servers available to Claude Code agents | ProofForge integration, new data sources |
| **Morphism registry** | New SOS-to-SOS morphisms for cross-domain transfer | Connecting ForgeStream to ProofForge SOS |
| **Mode definitions** | New meeting modes beyond A/B/C | Specialized meeting types |

## 14. Implementation Phases

### Phase 1: Foundation

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-1 | Event System | PostgreSQL schema, append-only rules, LISTEN/NOTIFY, Python event library | Nothing | Small |
| SP-2 | Knowledge Graph | Graph materialization from events, entity/edge types, rebuild-from-log | SP-1 | Medium |

### Phase 2: Intelligence

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-3 | Gemini Live API | WebSocket client, audio streaming, claim extraction, context injection | SP-1 | Medium |
| SP-4 | Synthesis Engine | Requirement/contradiction/branch/seed detection, suggestion queue | SP-1, SP-2 | Large |

### Phase 3: Agents

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-5 | Agent Spawner | tmux management, git worktree lifecycle, agent registry, monitoring | SP-1 | Medium |
| SP-6 | Agent Templates | Research/scaffold prompt templates, output parsing, event writing | SP-5 | Medium |

### Phase 4: Interface

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-7 | Terminal TUI | textual app, all panels, hotkeys, quiet mode | SP-1, SP-4 | Medium |
| SP-8 | Web Dashboard | FastAPI, knowledge graph viz, evaluator chart, timeline, live WebSocket | SP-1, SP-2 | Large |

### Phase 5: Governance

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-9 | SOS Governor | Axiom checking, trust region dynamics, evaluator computation | SP-1 | Medium |
| SP-10 | Self-Improvement | GRPO weight tuning, prompt evolution, branch pattern learning, post-meeting synthesis | SP-9 | Large |

### Phase 6: Experimental

| SP | Name | Delivers | Depends on | Effort |
|----|------|----------|------------|--------|
| SP-11 | Audio Copilot | Gemini TTS, pause detection, injection rules | SP-3, SP-4 | Small |
| SP-12 | Cross-Meeting Transfer | Merged graph, transfer detection, morphism registry, seed garden | SP-2, SP-9 | Large |

### Milestones

1. **"First Event"** (SP-1): Write and read events, append-only enforced
2. **"First Meeting"** (SP-1 + SP-3 + minimal SP-7): Gemini extracts claims in real-time, visible in terminal
3. **"First Scaffold"** (SP-5 + SP-6): Trigger Claude Code agents, produce code in worktrees
4. **"The Brain Comes Online"** (SP-2 + SP-4): Knowledge graph, synthesis, auto-spawning agents
5. **"Full TUI"** (SP-7): Complete terminal interface for meetings
6. **"The Conscience"** (SP-9): SOS axioms enforced at runtime
7. **"It Learns"** (SP-10): Self-improvement across meetings
8. **"The Dashboard"** (SP-8): Full web visualization
9. **"Transfer"** (SP-12): Cross-meeting knowledge reuse with morphisms
10. **"The Whisper"** (SP-11): Audio copilot

## 15. Project Structure

```
~/projects/forgestream/
├── forgestream/              # Python package
│   ├── events/               # SP-1: Event system
│   │   ├── schema.py
│   │   ├── store.py
│   │   └── subscribe.py
│   ├── graph/                # SP-2: Knowledge graph
│   │   ├── model.py
│   │   ├── materializer.py
│   │   └── query.py
│   ├── gemini/               # SP-3: Gemini Live API
│   │   ├── client.py
│   │   ├── audio.py
│   │   ├── extraction.py
│   │   └── context.py
│   ├── synthesis/            # SP-4: Synthesis engine
│   │   ├── engine.py
│   │   ├── requirements.py
│   │   ├── contradictions.py
│   │   ├── branches.py
│   │   ├── seeds.py
│   │   └── suggestions.py
│   ├── agents/               # SP-5, SP-6: Agent orchestration
│   │   ├── spawner.py
│   │   ├── worktrees.py
│   │   ├── registry.py
│   │   ├── monitor.py
│   │   └── templates/
│   │       ├── research.py
│   │       └── scaffold.py
│   ├── governor/             # SP-9, SP-10: SOS governance
│   │   ├── axioms.py
│   │   ├── evaluator.py      # PLUGGABLE
│   │   ├── trust_region.py
│   │   └── improvement.py
│   ├── tui/                  # SP-7: Terminal UI
│   │   ├── app.py
│   │   ├── panels/
│   │   │   ├── feed.py
│   │   │   ├── suggestions.py
│   │   │   ├── branches.py
│   │   │   └── agents.py
│   │   └── quiet.py
│   ├── dashboard/            # SP-8: Web dashboard
│   │   ├── server.py
│   │   ├── api.py
│   │   ├── ws.py
│   │   └── static/
│   └── copilot/              # SP-11: Audio copilot
│       ├── tts.py
│       └── injection.py
├── migrations/               # PostgreSQL migrations
├── tests/
├── docs/
│   └── meetings/             # Post-meeting reports
├── pyproject.toml
└── README.md
```
