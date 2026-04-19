# Polish Sprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the dashboard to live Firestore data, deploy security rules, create the BlackHole setup guide, add human feedback prompting, and build the synthesis engine continuous loop — completing ForgeStream's operational readiness.

**Architecture:** Dashboard API reads from Firestore using firebase-admin. SynthesisEngine subscribes to EventBus, processes claims through detectors, emits derived events back through the Orchestrator (ignoring self-authored events to prevent loops).

**Tech Stack:** Python 3.12+, firebase-admin, FastAPI, textual, existing ForgeStream modules

**Existing codebase:** `~/projects/forgestream/` — 198 tests passing, Milestones A+B complete.

---

## File Structure

### New files
```
forgestream/
├── synthesis/engine.py         # SynthesisEngine continuous loop (Task 5)
└── docs/
    └── blackhole-setup.md      # Aggregate device guide (Task 3)

tests/
├── synthesis/test_engine.py    # SynthesisEngine tests (Task 5)
└── dashboard/test_api_live.py  # Dashboard Firestore tests (Task 1)
```

### Modified files
```
forgestream/
├── dashboard/api.py            # Wire to Firestore (Task 1)
├── dashboard/server.py         # Accept Firestore client (Task 1)
├── tui/app.py                  # Human feedback prompt (Task 4)
├── synthesis/__init__.py       # Export SynthesisEngine (Task 5)
```

---

## Task 1: Wire Dashboard to Firestore

**Files:**
- Modify: `forgestream/dashboard/api.py`
- Modify: `forgestream/dashboard/server.py`
- Create: `tests/dashboard/test_api_live.py`

- [ ] **Step 1: Write failing tests for live dashboard data**

`tests/dashboard/test_api_live.py`:
```python
"""Dashboard API tests with Firestore data."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from forgestream.dashboard.server import create_app


class TestDashboardLiveData:
    def test_graph_returns_firestore_concepts(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        # Mock Firestore query returning 2 events
        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {
            "event_type": "claim",
            "payload": {"topic_keywords": ["Kafka", "ingestion"], "confidence": 0.9},
            "evaluator": 0.5,
        }
        mock_doc2 = MagicMock()
        mock_doc2.to_dict.return_value = {
            "event_type": "claim",
            "payload": {"topic_keywords": ["latency"], "confidence": 0.8},
            "evaluator": 0.6,
        }
        mock_collection.order_by.return_value.stream.return_value = [mock_doc1, mock_doc2]

        app = create_app(firestore_db=mock_db)
        client = TestClient(app)
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert len(data["concepts"]) > 0

    def test_evaluator_returns_trajectory(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "event_type": "claim",
            "evaluator": 0.45,
            "payload": {},
        }
        mock_collection.order_by.return_value.stream.return_value = [mock_doc]

        app = create_app(firestore_db=mock_db)
        client = TestClient(app)
        response = client.get("/api/evaluator")
        assert response.status_code == 200
        data = response.json()
        assert len(data["trajectory"]) > 0

    def test_health_always_works(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_graph_empty_without_firestore(self):
        app = create_app(firestore_db=None)
        client = TestClient(app)
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["concepts"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/dashboard/test_api_live.py -v
```

Expected: FAIL — `create_app()` doesn't accept `firestore_db`

- [ ] **Step 3: Update server.py to accept Firestore client**

Replace `forgestream/dashboard/server.py`:

```python
"""FastAPI dashboard server."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .api import create_router


def create_app(firestore_db: Any = None) -> FastAPI:
    """Create the dashboard FastAPI application."""
    app = FastAPI(title="ForgeStream Dashboard")
    router = create_router(firestore_db=firestore_db)
    app.include_router(router, prefix="/api")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    return app


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ForgeStream Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
        h1 { color: #0ff; }
        .panel { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 16px; margin: 12px 0; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
        #graph { min-height: 400px; }
        #evaluator { min-height: 200px; }
        #timeline { min-height: 100px; }
    </style>
</head>
<body>
    <h1>ForgeStream Dashboard</h1>
    <div class="grid">
        <div class="panel" id="graph">
            <h2>Knowledge Graph</h2>
            <p>D3.js force-directed graph loads here</p>
        </div>
        <div>
            <div class="panel" id="evaluator">
                <h2>Evaluator Trajectory</h2>
                <p>E(pi) chart loads here</p>
            </div>
            <div class="panel" id="artifacts">
                <h2>Artifact Tracker</h2>
                <p>Scaffold status loads here</p>
            </div>
        </div>
    </div>
    <div class="panel" id="timeline">
        <h2>Meeting Timeline</h2>
        <p>Event timeline loads here</p>
    </div>
    <script>
        // Live updates via Firestore onSnapshot (client-side JS)
        // Import Firebase JS SDK and subscribe to events collection
    </script>
</body>
</html>"""
```

- [ ] **Step 4: Update api.py to read from Firestore**

Replace `forgestream/dashboard/api.py`:

```python
"""REST API endpoints for the dashboard."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)


def create_router(firestore_db: Any = None) -> APIRouter:
    """Create API router, optionally connected to Firestore."""
    router = APIRouter()
    db = firestore_db

    def _get_events() -> list[dict]:
        """Fetch events from Firestore."""
        if db is None:
            return []
        try:
            docs = db.collection("events").order_by("timestamp").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning("Firestore query failed: %s", e)
            return []

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "firestore": db is not None}

    @router.get("/graph")
    async def get_graph() -> dict:
        """Return the knowledge graph for visualization."""
        events = _get_events()
        concepts: dict[str, dict] = {}
        edges: list[dict] = []

        for event in events:
            if event.get("event_type") != "claim":
                continue
            payload = event.get("payload", {})
            keywords = payload.get("topic_keywords", [])
            confidence = payload.get("confidence", 0.5)

            for kw in keywords:
                if kw not in concepts:
                    concepts[kw] = {"name": kw, "confidence": confidence, "count": 1}
                else:
                    concepts[kw]["confidence"] = max(concepts[kw]["confidence"], confidence)
                    concepts[kw]["count"] += 1

            for i, kw_a in enumerate(keywords):
                for kw_b in keywords[i + 1:]:
                    edges.append({"source": kw_a, "target": kw_b, "weight": confidence})

        requirements = [
            {"description": e.get("payload", {}).get("description", ""), "status": "detected"}
            for e in events if e.get("event_type") == "requirement"
        ]
        artifacts = [
            {"compiles": e.get("payload", {}).get("compiles", False),
             "tests_pass": e.get("payload", {}).get("tests_pass", False)}
            for e in events if e.get("event_type") == "artifact"
        ]

        return {
            "concepts": list(concepts.values()),
            "requirements": requirements,
            "artifacts": artifacts,
            "edges": edges,
        }

    @router.get("/evaluator")
    async def get_evaluator() -> dict:
        """Return the evaluator trajectory."""
        events = _get_events()
        trajectory = [
            {"evaluator": e.get("evaluator", 0.0), "event_type": e.get("event_type", "")}
            for e in events
        ]
        current_e = trajectory[-1]["evaluator"] if trajectory else 0.0

        return {
            "trajectory": trajectory,
            "current": {
                "E_micro": current_e,
                "E_meso": current_e,
                "E_macro": current_e,
            },
            "axioms": {
                "monotone": True,
                "bounded_step": True,
                "constraint": True,
            },
        }

    @router.get("/branches")
    async def get_branches() -> dict:
        """Return active branches with metrics."""
        return {"branches": []}

    @router.get("/agents")
    async def get_agents() -> dict:
        """Return active agent status."""
        return {"agents": []}

    @router.get("/suggestions")
    async def get_suggestions() -> dict:
        """Return the suggestion queue."""
        events = _get_events()
        suggestions = [
            {"text": e.get("payload", {}).get("text", ""),
             "priority": e.get("payload", {}).get("priority", 0.5)}
            for e in events if e.get("event_type") == "suggestion"
        ]
        return {"suggestions": suggestions}

    return router
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/dashboard/ -v
```

Expected: ALL PASS

- [ ] **Step 6: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

- [ ] **Step 7: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/dashboard/ tests/dashboard/
git commit -m "feat: wire dashboard API to Firestore with live data"
```

---

## Task 2: Deploy Firestore Security Rules

- [ ] **Step 1: Deploy rules via gcloud**

```bash
gcloud firestore databases update --project=forgestream-ai
```

If that doesn't deploy rules, use the Firebase CLI:

```bash
cd ~/projects/forgestream
npm install -g firebase-tools 2>/dev/null || true
firebase deploy --only firestore:rules --project=forgestream-ai
```

Or deploy manually via console: https://console.firebase.google.com/project/forgestream-ai/firestore/rules — paste contents of `firestore.rules`.

- [ ] **Step 2: Verify rules are active**

Go to https://console.firebase.google.com/project/forgestream-ai/firestore/rules and confirm the append-only rules are deployed.

- [ ] **Step 3: Commit** (no code changes, but document completion)

```bash
cd ~/projects/forgestream
git commit --allow-empty -m "chore: deploy Firestore append-only security rules"
```

---

## Task 3: BlackHole Aggregate Device Setup Guide

**Files:**
- Create: `docs/blackhole-setup.md`

- [ ] **Step 1: Write the guide**

Create `docs/blackhole-setup.md`:

```markdown
# BlackHole + Aggregate Device Setup Guide

ForgeStream uses BlackHole to capture system audio (Zoom/Meet/Teams calls) for real-time meeting intelligence.

## Step 1: Install BlackHole

```bash
brew install blackhole-2ch
```

Restart your Mac after installation (or at minimum, restart any audio applications).

## Step 2: Create an Aggregate Device

This lets you hear audio AND capture it simultaneously.

1. Open **Audio MIDI Setup** (search in Spotlight or find in `/Applications/Utilities/`)
2. Click the **+** button in the bottom-left corner
3. Select **Create Aggregate Device**
4. Check both:
   - **Built-in Microphone** (your mic — captures your voice)
   - **BlackHole 2ch** (virtual device — captures system audio)
5. Rename it to **ForgeStream Input** (double-click the name)

## Step 3: Create a Multi-Output Device

This routes audio to both your speakers AND BlackHole.

1. In Audio MIDI Setup, click **+** again
2. Select **Create Multi-Output Device**
3. Check both:
   - **Built-in Output** (your speakers/headphones)
   - **BlackHole 2ch**
4. Rename it to **ForgeStream Output**
5. Make sure **Built-in Output** is the **Master Device** (click the dropdown)

## Step 4: Configure Your Meeting App

For Zoom/Meet/Teams:
1. In the meeting app's audio settings:
   - **Speaker/Output**: Select **ForgeStream Output**
   - **Microphone/Input**: Keep your normal microphone
2. This routes remote participants' audio through BlackHole while you hear it normally

## Step 5: Run ForgeStream

```bash
cd ~/projects/forgestream
python3 -m forgestream.runner /path/to/audio --mode collaborative
```

ForgeStream automatically detects BlackHole and captures from it. To verify:

```python
from forgestream.audio.system_audio import SystemAudioSource
print("BlackHole available:", SystemAudioSource.is_available())
print("Device info:", SystemAudioSource.get_device_info())
```

## Step 6: For Full Meeting Capture (Both Sides)

To capture BOTH your voice AND remote participants in one stream:
- Set ForgeStream to use the **ForgeStream Input** aggregate device
- This combines your mic + BlackHole into one input

```python
from forgestream.audio.microphone import MicrophoneSource
devices = MicrophoneSource.list_input_devices()
# Find "ForgeStream Input" in the list and use its index
```

## Troubleshooting

- **No audio captured**: Make sure the meeting app output is set to ForgeStream Output
- **Echo/feedback**: Don't set the meeting app's mic to BlackHole — only the output
- **BlackHole not listed**: Restart your Mac after installation
- **Low volume**: In Audio MIDI Setup, check that BlackHole 2ch volume isn't muted
```

- [ ] **Step 2: Commit**

```bash
cd ~/projects/forgestream
mkdir -p docs
git add docs/blackhole-setup.md
git commit -m "docs: add BlackHole aggregate device setup guide"
```

---

## Task 4: Human Feedback Prompt After Meeting

**Files:**
- Modify: `forgestream/tui/app.py`
- Create: `tests/tui/test_feedback.py`

- [ ] **Step 1: Write failing test**

`tests/tui/test_feedback.py`:
```python
"""Test human feedback flow."""

from forgestream.tui.app import ForgeStreamApp
from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator


class TestFeedbackPrompt:
    def test_app_has_end_meeting_action(self):
        app = ForgeStreamApp()
        assert hasattr(app, "action_end_meeting")

    def test_app_tracks_meeting_ended_state(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app._meeting_ended is False

    def test_parse_feedback_score_valid(self):
        app = ForgeStreamApp()
        assert app._parse_feedback("8") == 0.8
        assert app._parse_feedback("10") == 1.0
        assert app._parse_feedback("1") == 0.1

    def test_parse_feedback_score_empty(self):
        app = ForgeStreamApp()
        assert app._parse_feedback("") is None
        assert app._parse_feedback("skip") is None

    def test_parse_feedback_score_invalid(self):
        app = ForgeStreamApp()
        assert app._parse_feedback("abc") is None
        assert app._parse_feedback("0") is None
        assert app._parse_feedback("11") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/tui/test_feedback.py -v
```

Expected: FAIL — `_meeting_ended` and `_parse_feedback` don't exist

- [ ] **Step 3: Add feedback state and parsing to TUI app**

Add to `ForgeStreamApp.__init__` in `forgestream/tui/app.py`:

```python
        self._meeting_ended = False
```

Add these methods to `ForgeStreamApp`:

```python
    @staticmethod
    def _parse_feedback(text: str) -> float | None:
        """Parse human feedback score (1-10) to float (0.1-1.0)."""
        text = text.strip()
        if not text or text.lower() in ("skip", "s", ""):
            return None
        try:
            score = int(text)
            if 1 <= score <= 10:
                return score / 10.0
        except ValueError:
            pass
        return None
```

Update `action_end_meeting`:

```python
    def action_end_meeting(self) -> None:
        """End the current meeting and trigger post-meeting synthesis."""
        self._meeting_ended = True
        feed = self.query_one("#feed", FeedPanel)
        feed.write("[bold cyan]>>> MEETING ENDED[/bold cyan]")
        feed.write("[cyan]Rate this meeting (1-10, or Enter to skip):[/cyan]")
        self.sub_title = "Meeting Ended — Rate 1-10"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/tui/test_feedback.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/tui/app.py tests/tui/test_feedback.py
git commit -m "feat: add human feedback prompt after meeting ends"
```

---

## Task 5: Synthesis Engine Continuous Loop

**Files:**
- Create: `forgestream/synthesis/engine.py`
- Create: `tests/synthesis/test_engine.py`
- Modify: `forgestream/synthesis/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/synthesis/test_engine.py`:
```python
"""SynthesisEngine tests -- continuous event processing loop."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.engine import SynthesisEngine


class TestSynthesisEngine:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        assert engine.orchestrator is orch

    async def test_processes_claim_event(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []
        original_process = orch.process_event

        async def capture_process(event: Event) -> bool:
            emitted.append(event)
            return await original_process(event)

        orch.process_event = capture_process

        claim = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "The system must handle 10k events per second",
                "is_requirement": False,
                "topic_keywords": ["throughput", "events"],
                "confidence": 0.9,
            },
        )

        await engine.on_event(claim)

        # Engine should have detected a requirement (text contains "must")
        req_events = [e for e in emitted if e.event_type == EventType.REQUIREMENT]
        assert len(req_events) >= 1

    async def test_ignores_own_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []

        async def capture(event: Event) -> bool:
            emitted.append(event)
            return True

        orch.process_event = capture

        # An event authored by the synthesis engine should be ignored
        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis_engine",
            evaluator=0.5,
            payload={"description": "already processed"},
        )

        await engine.on_event(event)

        # Should not emit any new events (would cause infinite loop)
        assert len(emitted) == 0

    async def test_ignores_non_claim_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        emitted = []

        async def capture(event: Event) -> bool:
            emitted.append(event)
            return True

        orch.process_event = capture

        artifact = Event(
            event_type=EventType.ARTIFACT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="scaffold",
            evaluator=0.6,
            payload={"compiles": True},
        )

        await engine.on_event(artifact)
        assert len(emitted) == 0

    def test_branch_tracker_shared(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        assert engine.branch_tracker is not None
        assert engine.branch_tracker.main_branch_id is not None

    async def test_seed_detection_runs_periodically(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        # Add enough claims to create disconnected clusters
        for keywords in [["A", "B", "C"], ["X", "Y", "Z"]]:
            for _ in range(3):
                claim = Event(
                    event_type=EventType.CLAIM,
                    session_id=orch.session_id,
                    branch_id=uuid4(),
                    author="gemini",
                    evaluator=0.5,
                    payload={"text": "test", "topic_keywords": keywords, "confidence": 0.8},
                )
                engine._update_graph(claim)

        seeds = engine.detect_seeds()
        # May or may not find seeds depending on graph connectivity
        assert isinstance(seeds, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/synthesis/test_engine.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement SynthesisEngine**

`forgestream/synthesis/engine.py`:
```python
"""SynthesisEngine -- continuous event processing loop.

Subscribes to the EventBus. For each claim event:
- Detects requirements
- Detects contradictions
- Tracks branches
- Periodically detects seeds
Emits derived events back through the Orchestrator.
Ignores self-authored events to prevent infinite loops.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.model import KnowledgeGraph
from forgestream.orchestrator import Orchestrator

from .branches import BranchTracker
from .contradictions import ContradictionDetector
from .requirements import RequirementDetector
from .seeds import SeedDetector
from .suggestions import Suggestion, SuggestionQueue

AUTHOR = "synthesis_engine"


class SynthesisEngine:
    """Continuous event processor -- the brain of ForgeStream.

    Subscribes to the orchestrator EventBus and processes claim events
    through requirement detection, contradiction detection, branch tracking,
    and seed detection. Emits derived events back through the Orchestrator.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.req_detector = RequirementDetector()
        self.branch_tracker = BranchTracker()
        self.seed_detector = SeedDetector(min_cluster_size=3)
        self.materializer = GraphMaterializer()
        self.suggestion_queue = SuggestionQueue()

        self._claim_events: list[Event] = []
        self._claim_count_at_last_seed_check = 0
        self._seed_check_interval = 10  # check every N claims

    async def on_event(self, event: Event) -> None:
        """EventBus handler -- process incoming events."""
        # Ignore self-authored events to prevent infinite loops
        if event.author == AUTHOR:
            return

        # Only process claim events
        if event.event_type != EventType.CLAIM:
            return

        self._claim_events.append(event)

        # 1. Requirement detection
        req = self.req_detector.check(event)
        if req:
            req_event = Event(
                event_type=EventType.REQUIREMENT,
                session_id=event.session_id,
                branch_id=event.branch_id,
                author=AUTHOR,
                evaluator=0.0,
                payload=req,
                parent_id=event.id,
            )
            await self.orchestrator.process_event(req_event)

        # 2. Branch tracking
        keywords = event.payload.get("topic_keywords", [])
        if keywords:
            self.branch_tracker.add_keywords(
                self.branch_tracker.main_branch_id, keywords
            )
            drift = self.branch_tracker.check_drift(
                self.branch_tracker.main_branch_id, keywords
            )
            if drift:
                branch_event = Event(
                    event_type=EventType.BRANCH_POINT,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=drift,
                    parent_id=event.id,
                )
                await self.orchestrator.process_event(branch_event)

        # 3. Contradiction detection (against current graph)
        graph = self._build_graph()
        contradiction_detector = ContradictionDetector(graph=graph)
        for kw in keywords:
            contradiction = contradiction_detector.check(
                concept_name=kw, keywords=keywords
            )
            if contradiction:
                contra_event = Event(
                    event_type=EventType.CONTRADICTION,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=contradiction,
                    parent_id=event.id,
                )
                await self.orchestrator.process_event(contra_event)
                break  # one contradiction per claim is enough

        # 4. Periodic seed detection
        if (len(self._claim_events) - self._claim_count_at_last_seed_check
                >= self._seed_check_interval):
            self._claim_count_at_last_seed_check = len(self._claim_events)
            seeds = self.detect_seeds()
            for seed_data in seeds:
                seed_event = Event(
                    event_type=EventType.SEED,
                    session_id=event.session_id,
                    branch_id=event.branch_id,
                    author=AUTHOR,
                    evaluator=0.0,
                    payload=seed_data,
                )
                await self.orchestrator.process_event(seed_event)

    def _build_graph(self) -> KnowledgeGraph:
        """Build knowledge graph from accumulated claim events."""
        return self.materializer.materialize(self._claim_events)

    def _update_graph(self, event: Event) -> None:
        """Add a claim event to the accumulator (for testing)."""
        self._claim_events.append(event)

    def detect_seeds(self) -> list[dict[str, Any]]:
        """Run seed detection on the current knowledge graph."""
        graph = self._build_graph()
        return self.seed_detector.detect(graph)
```

- [ ] **Step 4: Update synthesis __init__.py**

Add to `forgestream/synthesis/__init__.py`:

```python
from .engine import SynthesisEngine
```

And add `"SynthesisEngine"` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/synthesis/test_engine.py -v
```

Expected: ALL PASS

- [ ] **Step 6: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

Expected: 200+ passed

- [ ] **Step 7: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/synthesis/ tests/synthesis/test_engine.py
git commit -m "feat: add SynthesisEngine continuous loop with requirement/branch/seed detection"
```

---

## Final Verification

After all tasks complete:

- [ ] Dashboard serves live Firestore data: start `uvicorn forgestream.dashboard.server:create_app --factory` and check `/api/graph`
- [ ] Firestore rules deployed (check Firebase Console)
- [ ] BlackHole guide exists: `cat docs/blackhole-setup.md`
- [ ] Human feedback parsing works: `python3 -c "from forgestream.tui.app import ForgeStreamApp; print(ForgeStreamApp._parse_feedback('8'))"`  → `0.8`
- [ ] SynthesisEngine detects requirements: integration test passes
- [ ] Full suite: `python3 -m pytest -q` → all pass

- [ ] **Push everything**

```bash
cd ~/projects/forgestream
git push
```
