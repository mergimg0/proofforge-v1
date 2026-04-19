# Milestone B: Full Autonomous System — Design Spec

**Date:** 2026-03-27
**Status:** Approved
**Repository:** `~/projects/forgestream/`
**Depends on:** Milestone A (complete)

---

## 1. Vision

Milestone B transforms ForgeStream from a batch audio analyzer into a fully autonomous meeting system. Three audio input modes (file replay, microphone, system audio loopback) feed into the Gemini Live API via WebSocket. Claude Code agents spawn automatically to research and scaffold during meetings. Post-meeting GRPO tuning improves the system's evaluator and prompts across meetings.

After Milestone B, the full SOS orbit is operational: meetings produce knowledge → agents build artifacts → the evaluator measures quality → GRPO tunes the weights → the next meeting is better. Convergence begins.

## 2. AudioSource Abstraction

All audio input modes implement a common interface producing PCM 16kHz mono int16 chunks.

### Interface

```python
class AudioSource:
    sample_rate: int = 16000      # Hz
    channels: int = 1             # mono
    chunk_duration: float = 0.5   # seconds per chunk (16KB)

    async def start() -> None
    async def stop() -> None
    async def chunks() -> AsyncIterator[bytes]

    @property
    def is_active() -> bool
```

Every source yields identical 16KB chunks (16000 samples/sec * 2 bytes/sample * 0.5s). The GeminiLiveStream consumes chunks from any source without knowing which is active.

### Three Implementations

| Source | Input | Library | Speed | Use case |
|--------|-------|---------|-------|----------|
| FileReplaySource | .m4a/.mp3/.wav | soundfile + pydub | Configurable (1x-4x) | Testing, past meetings |
| MicrophoneSource | System microphone | sounddevice | Real-time (1x) | In-person meetings |
| SystemAudioSource | BlackHole virtual device | sounddevice | Real-time (1x) | Zoom/Meet/Teams |

### FileReplaySource

Reads audio files, resamples to 16kHz mono PCM, yields chunks at configurable speed.

- Accepts single file or folder of files (natural sort order)
- Speed multiplier: 1x (simulates real-time), 2x, 4x, or 0 (as fast as possible)
- When a file ends, moves to the next file in the folder
- When all files are exhausted, emits is_active = False

### MicrophoneSource

Captures from the default system microphone via sounddevice.

- Lists available input devices on start
- Uses default input device unless overridden
- Chunk callback: sounddevice fills a buffer, source yields it
- For in-person meetings: mic captures all speakers in the room

### SystemAudioSource

Captures system audio output via BlackHole virtual audio device.

- Requires BlackHole installed: `brew install blackhole-2ch`
- Searches for BlackHole device in sounddevice device list
- For full meeting capture (your voice + remote): user creates an aggregate device in Audio MIDI Setup combining mic + BlackHole, then ForgeStream captures from the aggregate device
- Falls back to MicrophoneSource if BlackHole not found

### Runtime Source Switching

The TUI supports switching sources mid-meeting:

```
/source mic          → switch to MicrophoneSource
/source system       → switch to SystemAudioSource
/source file X.m4a   → switch to FileReplaySource
```

The GeminiLiveStream pauses the current source, starts the new one, and continues streaming. The WebSocket session stays alive — no knowledge graph reset, no context loss.

## 3. GeminiLiveStream Client

Manages the Gemini Live API WebSocket session. Connects AudioSource to the Orchestrator.

### Architecture

```
AudioSource.chunks() → PCM bytes (16KB every 0.5s)
    │
    ▼
GeminiLiveStream
    ├─ Send Loop (asyncio.Task)
    │   └─ AudioSource → WebSocket send()
    │
    ├─ Receive Loop (asyncio.Task)
    │   └─ WebSocket receive() → parse JSONL → ClaimExtractor → Orchestrator
    │
    └─ Context Injection Loop (asyncio.Task)
        └─ Every 10 min: knowledge graph summary → WebSocket send() as text
    │
    ▼
Orchestrator.process_event() → PostgreSQL + Firestore + EventBus + TUI
```

### Connection

```python
from google import genai

client = genai.Client(vertexai=True, project="forgestream-ai", location="europe-west2")
session = await client.aio.live.connect(
    model="gemini-2.5-flash",
    config={
        "response_modalities": ["TEXT"],
        "system_instruction": MODE_INSTRUCTIONS[mode],
    },
)
```

### Send Loop

```python
async def _send_loop(self):
    async for chunk in self.audio_source.chunks():
        await self._session.send({"data": chunk, "mime_type": "audio/pcm"})
```

### Receive Loop

```python
async def _receive_loop(self):
    async for response in self._session.receive():
        if hasattr(response, "text") and response.text:
            for claim_data in self._parse_jsonl(response.text):
                event = self.extractor.parse_claim(claim_data)
                await self.orchestrator.process_event(event)
```

### Context Injection

Every 10 minutes, injects a knowledge graph summary to fight Gemini context decay:

```python
async def _context_injection_loop(self):
    while self._active:
        await asyncio.sleep(600)
        summary = self.context_builder.build_injection(
            graph=self._current_graph,
            active_branches=self._active_branch_names,
        )
        await self._session.send({"text": summary})
```

Uses the existing ContextBuilder from SP-3.

### Session Lifecycle

```
start_meeting(audio_source, mode)
  ├─ Connect to Gemini Live API WebSocket
  ├─ Set system instruction for mode
  ├─ Start AudioSource
  ├─ Launch send loop
  ├─ Launch receive loop
  └─ Launch context injection loop

end_meeting()
  ├─ Stop AudioSource
  ├─ Cancel all loops
  ├─ Close WebSocket
  └─ Trigger post-meeting synthesis
```

### Error Handling

- WebSocket drops: reconnect automatically, resume from current AudioSource position
- Non-JSONL responses from Gemini: skip, log warning
- AudioSource failure: emit system event, pause processing, keep session alive for source switch
- Gemini rate limit: exponential backoff (60s, 120s, 180s)

## 4. Claude Code Agent Spawning

Wires the existing SpawnPolicy, AgentRegistry, WorktreeManager, and prompt templates to the orchestrator's event stream.

### AgentDispatcher

Subscribes to the orchestrator EventBus. When a requirement event arrives:

1. Check SpawnPolicy.can_spawn() (trust region, agent count, cooldown)
2. If allowed:
   a. Register agent in AgentRegistry
   b. Build prompt from template + live knowledge graph context
   c. Create git worktree (scaffold agents only)
   d. Write prompt to temp file (avoids shell escaping)
   e. Launch `claude -p` in a tmux session
   f. Start monitoring loop

### CLI Invocation

Research agents:
```bash
tmux new-session -d -s research-{id} \
  "claude -p \"$(cat /tmp/forgestream/prompt-{id}.md)\" \
   --allowedTools 'WebSearch,WebFetch,Read,Grep,Glob' \
   2>&1 | tee /tmp/forgestream/agent-{id}.out"
```

Scaffold agents:
```bash
tmux new-session -d -s scaffold-{id} \
  "cd {worktree_path} && \
   claude -p \"$(cat /tmp/forgestream/prompt-{id}.md)\" \
   2>&1 | tee /tmp/forgestream/agent-{id}.out"
```

### Prompt Enrichment

Template prompts get enriched with live context:

- Recent verified findings (last 5 from event log)
- Active requirements (related to this agent's task)
- Branch context (which conversation thread spawned this)
- Meeting mode context (extract/collaborative/knowledge)

### Monitoring

Background async task polls every 5 seconds:

- Check agent output file for completion markers (JSON output)
- Check if tmux session still exists
- Enforce timeouts (research: 5min, scaffold: 15min)

### On Agent Completion

- Research agent: write verified_finding event → knowledge graph updated → suggestions enriched
- Scaffold agent: write artifact event + suggestion events for open questions → worktree preserved
- Failed agent: write failure event, retry once, then abandon

### TUI Integration

The Agents panel (already built in SP-7) shows live status. The monitoring loop updates the registry, and the TUI's EventBus subscription refreshes the panel.

## 5. Post-Meeting GRPO Tuning

Runs when a meeting ends. Makes Meeting N+1 better than Meeting N.

### Four Phases

**Phase 1: Report Generation**
- MeetingSynthesizer.generate_summary() produces the meeting report
- Write meeting_summary event to the log
- Write markdown report to docs/meetings/YYYY-MM-DD-{slug}.md
- Build human review queue (proof obligations, unresolved contradictions, seeds, scaffolds)

**Phase 2: Evaluator Weight Tuning (GRPO)**
- Load current weights from data/weights.json
- WeightTuner.tune(weights, events, feedback_score)
- Save updated weights
- Write evaluator_snapshot event with new E value

**Phase 3: Prompt Evolution**
- Score each agent prompt + output pair
- PromptEvolution.score_prompt() for each agent that ran
- Save scores to data/prompt_history.json
- Next meeting composes prompts from best-scoring variants

**Phase 4: Knowledge Persistence**
- Rebuild cross-session knowledge graph indexes
- Detect transfer candidates (shared concepts across meetings)
- Update trust region based on E_macro trajectory
- Save trust region state to data/trust_region.json

### Persistent State Files

```
~/projects/forgestream/data/
├── weights.json           # Current evaluator weights
├── weights_history.json   # Weight trajectory across meetings
├── prompt_history.json    # Prompt variant scores
└── trust_region.json      # Trust region state (epsilon, violations, improvements)
```

### Human Feedback Signal

After meeting ends, the TUI prompts:

```
Meeting ended. How useful was this session? (1-10, or Enter to skip):
```

If provided, the explicit score is the GRPO target. If skipped, an auto-score is computed:

```
auto_score = (
    0.3 * requirement_to_scaffold_ratio
    + 0.3 * verified_findings_per_claim
    + 0.2 * suggestion_uptake_rate
    + 0.2 * branch_merge_rate
)
```

### Trust Region Update

```python
e_meso = evaluator.compute(meeting_events)
if e_meso > previous_e_meso:
    trust_region.record_meeting_result(e_macro_improved=True, axiom_violations=0)
else:
    trust_region.record_meeting_result(e_macro_improved=False, axiom_violations=0)
```

### Meeting Report Format

Written to docs/meetings/YYYY-MM-DD-{slug}.md:

```markdown
# Meeting: {name}
**Date:** {date}
**Duration:** {duration}
**Mode:** {mode}
**E(pi) final:** {e_value}

## Knowledge Extracted
- {claim_count} claims, {concept_count} concepts, {req_count} requirements

## Top Requirements
1. {requirement_1}
2. {requirement_2}

## Branches Explored
- main ({n} claims)
- branch-1: {description} ({n} claims)

## Seeds Detected
- {seed_1} (novelty: {score})

## Artifacts Produced
- {artifact_1} (compiles: yes/no, tests: yes/no)

## Human Review Queue
- [ ] Proof obligations tagged during meeting
- [ ] Unresolved contradictions
- [ ] Seeds awaiting promotion
- [ ] Scaffolds awaiting review

## SOS Status
- Axiom 1 (Monotone): {status}
- Axiom 2 (Bounded): {status}
- Axiom 3 (Constraint): {status}
- Trust region: epsilon = {value}
- Evaluator weights: {weights}
```

## 6. Dependencies

```
sounddevice          # Audio capture (mic + BlackHole)
soundfile            # Audio file reading
pydub                # Audio format conversion (m4a → wav → PCM)
ffmpeg               # Required by pydub for m4a/mp3 decoding (brew install ffmpeg)
google-genai         # Gemini Live API (already installed)
```

BlackHole installation: `brew install blackhole-2ch`

## 7. File Structure

### New files

```
forgestream/
├── audio/
│   ├── __init__.py
│   ├── source.py           # AudioSource base class
│   ├── file_replay.py      # FileReplaySource
│   ├── microphone.py       # MicrophoneSource
│   └── system_audio.py     # SystemAudioSource
├── live_stream.py           # GeminiLiveStream client
├── agent_dispatcher.py      # Agent spawning from events
└── post_meeting.py          # Post-meeting synthesis + GRPO

data/
├── weights.json
├── weights_history.json
├── prompt_history.json
└── trust_region.json

tests/
├── audio/
│   ├── __init__.py
│   ├── test_file_replay.py
│   ├── test_microphone.py
│   └── test_system_audio.py
├── test_live_stream.py
├── test_agent_dispatcher.py
└── test_post_meeting.py
```

### Modified files

```
forgestream/
├── tui/app.py              # Add /source commands, /end command, feedback prompt
├── config.py               # Add audio source config, data directory paths
└── orchestrator.py          # Add agent dispatcher integration
```

## 8. Implementation Order

| Task | What | Depends on | Effort |
|------|------|------------|--------|
| B-1 | AudioSource base + FileReplaySource | Nothing | Small |
| B-2 | MicrophoneSource | B-1 | Small |
| B-3 | SystemAudioSource (BlackHole) | B-1 | Small |
| B-4 | GeminiLiveStream client | B-1 | Medium |
| B-5 | AgentDispatcher | Orchestrator (done) | Medium |
| B-6 | PostMeetingSynthesis | Evaluator + WeightTuner (done) | Medium |
| B-7 | TUI integration (/source, /end, feedback) | B-4, B-5, B-6 | Small |
| B-8 | Integration tests | All above | Small |
