# Milestone A: Usable Meeting Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform ForgeStream from independent modules into a usable meeting tool with persistent events (PostgreSQL + Firestore dual-write), tuned branch detection, and a live TUI — ready for real meetings.

**Architecture:** Orchestrator connects to PostgreSQL EventStore for local persistence and syncs events to Firestore for cloud access/dashboard. Branch threshold tuned from real meeting data. TUI runs as the main process with the orchestrator's EventBus feeding live updates to all panels.

**Tech Stack:** Python 3.12+, psycopg 3, firebase-admin SDK, textual, pytest

**Existing codebase:** `~/projects/forgestream/` — 48 source files, 144 tests, all passing. See design spec at `~/projects/proofforge/docs/superpowers/specs/2026-03-27-forgestream-design.md`.

---

## File Structure

### New files
```
forgestream/
├── firestore_sync.py          # Dual-write to Firestore (Task 3)
└── runner.py                  # Audio meeting runner with TUI (Task 5)

tests/
├── test_firestore_sync.py     # Firestore sync tests (Task 3)
└── test_runner.py             # Runner integration tests (Task 5)

firestore.rules                # Append-only security rules (Task 2)
```

### Modified files
```
forgestream/
├── synthesis/branches.py      # Tune DRIFT_THRESHOLD (Task 1)
├── orchestrator.py            # Wire PostgreSQL + Firestore writes (Task 4)
├── tui/app.py                 # Accept orchestrator, subscribe to EventBus (Task 5)
├── tui/panels/feed.py         # Subscribe to claim events (Task 5)
├── tui/panels/suggestions.py  # Subscribe to suggestion events (Task 5)
├── tui/panels/branches.py     # Subscribe to branch events (Task 5)
├── config.py                  # Add Firebase config fields (Task 2)

tests/
├── synthesis/test_branches.py # Updated threshold tests (Task 1)
├── test_orchestrator.py       # PostgreSQL + Firestore integration (Task 4)
```

---

## Task 1: Tune Branch Sensitivity

**Files:**
- Modify: `forgestream/synthesis/branches.py`
- Modify: `tests/synthesis/test_branches.py`

- [ ] **Step 1: Write test for new threshold**

Add to `tests/synthesis/test_branches.py`:

```python
class TestBranchThreshold:
    def test_related_topics_dont_branch(self):
        """Topics sharing 2+ keywords should NOT trigger a branch."""
        tracker = BranchTracker()
        tracker.add_keywords(tracker.main_branch_id,
                             ["Kafka", "ingestion", "pipeline", "data", "streaming"])
        # Related topic — shares "Kafka" and adds closely related concepts
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["Kafka", "consumer", "offset", "partition"],
        )
        assert drift is None

    def test_unrelated_topics_do_branch(self):
        """Completely unrelated topics SHOULD trigger a branch."""
        tracker = BranchTracker()
        tracker.add_keywords(tracker.main_branch_id,
                             ["Kafka", "ingestion", "pipeline", "data", "streaming"])
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["quantum", "entanglement", "physics"],
        )
        assert drift is not None

    def test_partially_related_no_branch(self):
        """Topics with some overlap should NOT branch."""
        tracker = BranchTracker()
        for _ in range(3):
            tracker.add_keywords(tracker.main_branch_id,
                                 ["rust", "agents", "lean_4", "proof", "verification"])
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["rust", "performance", "memory_safety"],
        )
        assert drift is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/synthesis/test_branches.py::TestBranchThreshold -v
```

Expected: `test_related_topics_dont_branch` and `test_partially_related_no_branch` FAIL (current threshold 0.7 is too aggressive)

- [ ] **Step 3: Update DRIFT_THRESHOLD**

In `forgestream/synthesis/branches.py`, change:

```python
DRIFT_THRESHOLD = 0.7  # Jaccard distance threshold for branching
```

to:

```python
DRIFT_THRESHOLD = 0.85  # Jaccard distance — only truly unrelated topics branch
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/synthesis/test_branches.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run full suite — no regressions**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

Expected: 144+ passed

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/synthesis/branches.py tests/synthesis/test_branches.py
git commit -m "fix: tune branch DRIFT_THRESHOLD from 0.7 to 0.85 to reduce noise"
```

---

## Task 2: Set Up Firebase + Firestore

**Files:**
- Create: `firestore.rules`
- Modify: `forgestream/config.py`

- [ ] **Step 1: Enable Firestore on the forgestream-ai GCP project**

```bash
gcloud services enable firestore.googleapis.com --project=forgestream-ai
```

- [ ] **Step 2: Create Firestore database in europe-west2**

```bash
gcloud firestore databases create \
  --project=forgestream-ai \
  --location=europe-west2 \
  --type=firestore-native
```

- [ ] **Step 3: Install firebase-admin SDK**

```bash
cd ~/projects/forgestream
python3 -m pip install firebase-admin
```

- [ ] **Step 4: Create append-only security rules**

Create `firestore.rules`:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Events collection: append-only (ECEF guarantee)
    match /events/{eventId} {
      allow create: if true;
      allow update, delete: if false;
      allow read: if true;
    }

    // Sessions collection: read-only after creation
    match /sessions/{sessionId} {
      allow create: if true;
      allow update: if true;  // session metadata can be updated (e.g. end time)
      allow delete: if false;
      allow read: if true;
    }
  }
}
```

- [ ] **Step 5: Deploy security rules**

```bash
# Initialize Firebase in the project if not already done
cd ~/projects/forgestream
firebase init firestore --project=forgestream-ai

# Or deploy rules directly
gcloud firestore databases update --project=forgestream-ai
```

Note: If `firebase` CLI isn't available, the rules can be deployed via the Firebase Console at https://console.firebase.google.com/project/forgestream-ai/firestore/rules

- [ ] **Step 6: Add Firebase config to ForgeStreamConfig**

In `forgestream/config.py`, add to the `ForgeStreamConfig` dataclass:

```python
    # Firebase / Firestore
    firebase_project: str = "forgestream-ai"
    firestore_enabled: bool = True
```

And add to `load_config()`:

```python
        firestore_enabled=os.environ.get("FORGESTREAM_FIRESTORE_ENABLED", "true").lower() == "true",
```

- [ ] **Step 7: Commit**

```bash
cd ~/projects/forgestream
git add firestore.rules forgestream/config.py
git commit -m "feat: set up Firestore with append-only security rules"
```

---

## Task 3: Firestore Sync Module

**Files:**
- Create: `forgestream/firestore_sync.py`
- Create: `tests/test_firestore_sync.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_firestore_sync.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.firestore_sync import FirestoreSync


class TestFirestoreSync:
    def test_init_disabled(self):
        """When disabled, sync creates no Firebase connection."""
        sync = FirestoreSync(project_id="test", enabled=False)
        assert sync.enabled is False

    @patch("forgestream.firestore_sync.firestore")
    @patch("forgestream.firestore_sync.firebase_admin")
    def test_event_to_firestore_doc(self, mock_admin, mock_firestore):
        """Events serialize correctly for Firestore."""
        sync = FirestoreSync(project_id="test", enabled=False)
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test claim", "confidence": 0.85},
        )
        doc = sync._event_to_doc(event)
        assert doc["event_type"] == "claim"
        assert doc["author"] == "gemini"
        assert doc["payload"]["text"] == "test claim"
        assert isinstance(doc["id"], str)
        assert isinstance(doc["session_id"], str)

    @patch("forgestream.firestore_sync.firestore")
    @patch("forgestream.firestore_sync.firebase_admin")
    def test_sync_disabled_is_noop(self, mock_admin, mock_firestore):
        """Sync with enabled=False does nothing."""
        sync = FirestoreSync(project_id="test", enabled=False)
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test"},
        )
        sync.sync_event(event)  # should not raise

    @patch("forgestream.firestore_sync.firestore")
    @patch("forgestream.firestore_sync.firebase_admin")
    def test_sync_enabled_writes_to_collection(self, mock_admin, mock_firestore):
        """When enabled, sync writes to the events collection."""
        mock_db = MagicMock()
        mock_firestore.client.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection

        sync = FirestoreSync(project_id="test", enabled=True)
        sync._db = mock_db

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "written to firestore"},
        )
        sync.sync_event(event)

        mock_db.collection.assert_called_with("events")
        mock_collection.document.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_firestore_sync.py -v
```

Expected: FAIL — `ImportError: cannot import name 'FirestoreSync'`

- [ ] **Step 3: Implement FirestoreSync**

Create `forgestream/firestore_sync.py`:

```python
"""Firestore sync — async dual-write for cloud access and dashboard.

Events are written to Firestore in the background after PostgreSQL write.
Firestore is the cloud sync layer; PostgreSQL stays the local source of truth.
"""

from __future__ import annotations

import logging
from typing import Any

from .events.schema import Event

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    firebase_admin = None  # type: ignore
    firestore = None  # type: ignore


class FirestoreSync:
    """Syncs events to Firestore for cloud access and real-time dashboard.

    - Append-only: documents are created, never updated or deleted
    - Background: sync failures don't block the main pipeline
    - Optional: disabled gracefully when firebase-admin isn't installed
    """

    def __init__(self, project_id: str, enabled: bool = True) -> None:
        self.project_id = project_id
        self.enabled = enabled and HAS_FIREBASE
        self._db = None
        self._app = None

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Firebase Admin SDK and Firestore client."""
        try:
            # Use application default credentials (gcloud auth)
            self._app = firebase_admin.initialize_app(
                credential=credentials.ApplicationDefault(),
                options={"projectId": self.project_id},
            )
            self._db = firestore.client()
            logger.info("Firestore sync initialized for project %s", self.project_id)
        except Exception as e:
            logger.warning("Firestore sync disabled: %s", e)
            self.enabled = False

    def sync_event(self, event: Event) -> None:
        """Sync an event to Firestore. Fire-and-forget."""
        if not self.enabled or self._db is None:
            return

        try:
            doc = self._event_to_doc(event)
            self._db.collection("events").document(str(event.id)).set(doc)
        except Exception as e:
            # Never block the main pipeline for a Firestore failure
            logger.warning("Firestore sync failed for event %s: %s", event.id, e)

    def sync_session(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Sync session metadata to Firestore."""
        if not self.enabled or self._db is None:
            return

        try:
            self._db.collection("sessions").document(session_id).set(metadata)
        except Exception as e:
            logger.warning("Firestore session sync failed: %s", e)

    @staticmethod
    def _event_to_doc(event: Event) -> dict[str, Any]:
        """Convert an Event to a Firestore-compatible document."""
        return {
            "id": str(event.id),
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "parent_id": str(event.parent_id) if event.parent_id else None,
            "branch_id": str(event.branch_id),
            "author": event.author,
            "evaluator": event.evaluator,
            "payload": event.payload,
            "degradation_flag": event.degradation_flag,
            "trust_region_ok": event.trust_region_ok,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_firestore_sync.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run full suite — no regressions**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/firestore_sync.py tests/test_firestore_sync.py
git commit -m "feat: add FirestoreSync for cloud dual-write"
```

---

## Task 4: Wire Orchestrator to PostgreSQL + Firestore

**Files:**
- Modify: `forgestream/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test for PostgreSQL + Firestore write**

Add to `tests/test_orchestrator.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

class TestOrchestratorPersistence:
    async def test_process_event_writes_to_store(self, db_conn):
        """Events are persisted to PostgreSQL."""
        from forgestream.events.store import EventStore

        config = ForgeStreamConfig()
        store = EventStore(conn=db_conn)
        orch = Orchestrator(config, store=store)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "persisted claim", "topic_keywords": ["test"]},
        )
        result = await orch.process_event(event)
        assert result is True

        # Verify it's in PostgreSQL
        events = await store.get_events(session_id=orch.session_id)
        assert len(events) == 1
        assert events[0].payload["text"] == "persisted claim"

    async def test_process_event_syncs_to_firestore(self):
        """Events are synced to Firestore when enabled."""
        config = ForgeStreamConfig()
        mock_sync = MagicMock()
        orch = Orchestrator(config, firestore_sync=mock_sync)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={"text": "synced to firestore"},
        )
        await orch.process_event(event)

        mock_sync.sync_event.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_orchestrator.py::TestOrchestratorPersistence -v
```

Expected: FAIL — `Orchestrator.__init__() got unexpected keyword argument 'store'`

- [ ] **Step 3: Update Orchestrator to accept store and firestore_sync**

Replace the `Orchestrator.__init__` and `process_event` in `forgestream/orchestrator.py`:

```python
class Orchestrator:
    """The conductor. Manages the event lifecycle:

    1. Receive event (from worker or direct)
    2. Structural validation (pre-write, < 1ms)
    3. Write to PostgreSQL (append-only)
    4. Sync to Firestore (fire-and-forget)
    5. Publish to in-memory event bus (instant TUI update)
    6. Governor post-write observation (async)
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        store: EventStore | None = None,
        firestore_sync: FirestoreSync | None = None,
    ) -> None:
        self.config = config
        self.session_id = uuid4()
        self.event_bus = EventBus()
        self.validator = StructuralValidator()
        self.evaluator = Evaluator()
        self.store = store
        self.firestore_sync = firestore_sync
        self._event_buffer: list[Event] = []

    async def process_event(self, event: Event) -> bool:
        """Process a single event through the full lifecycle.

        Returns True if the event was accepted, False if structurally invalid.
        """
        # Step 1: Structural validation
        result = self.validator.validate(event)
        if not result.valid:
            return False

        # Step 2: Apply downgrade if needed
        if result.downgrade_to is not None:
            event = Event(
                event_type=result.downgrade_to,
                session_id=event.session_id,
                branch_id=event.branch_id,
                author=event.author,
                evaluator=event.evaluator,
                payload=event.payload,
                id=event.id,
                timestamp=event.timestamp,
                parent_id=event.parent_id,
            )

        # Step 3: Compute evaluator value
        self._event_buffer.append(event)
        event.evaluator = self.evaluator.compute(self._event_buffer[-20:])

        # Step 4: Write to PostgreSQL
        if self.store is not None:
            await self.store.append(event)

        # Step 5: Sync to Firestore (fire-and-forget)
        if self.firestore_sync is not None:
            self.firestore_sync.sync_event(event)

        # Step 6: Publish to in-memory bus (instant TUI update)
        await self.event_bus.publish(event)

        return True

    @property
    def event_count(self) -> int:
        return len(self._event_buffer)
```

Add imports at the top of `forgestream/orchestrator.py`:

```python
from .events.store import EventStore
from .firestore_sync import FirestoreSync
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_orchestrator.py -v
```

Expected: ALL PASS (including old tests — they pass `store=None` implicitly)

- [ ] **Step 5: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: wire Orchestrator to PostgreSQL + Firestore dual-write"
```

---

## Task 5: Wire TUI to Live Event Bus

**Files:**
- Modify: `forgestream/tui/app.py`
- Modify: `forgestream/tui/panels/feed.py`
- Modify: `forgestream/tui/panels/suggestions.py`
- Modify: `forgestream/tui/panels/branches.py`
- Create: `forgestream/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Update FeedPanel to accept events programmatically**

Replace `forgestream/tui/panels/feed.py`:

```python
"""Live feed panel -- scrolling log of claims."""

from textual.widgets import RichLog

from forgestream.events.schema import Event, EventType


class FeedPanel(RichLog):
    """Scrolling log of claim events with linkage annotations."""

    DEFAULT_CSS = """
    FeedPanel {
        height: 1fr;
        border: solid green;
    }
    """

    def on_event_received(self, event: Event) -> None:
        """Handle an event from the EventBus."""
        if event.event_type == EventType.CLAIM:
            confidence = event.payload.get("confidence", 0.5)
            speaker = event.payload.get("speaker", "unknown")
            text = event.payload.get("text", "")
            conf_color = (
                "green" if confidence >= 0.7
                else "yellow" if confidence >= 0.4
                else "red"
            )
            self.write(
                f"[dim]{event.timestamp.strftime('%H:%M:%S')}[/dim] "
                f"[{conf_color}][{speaker}][/{conf_color}] {text} "
                f"[dim]conf:{confidence:.2f}[/dim]"
            )
            if event.payload.get("is_requirement"):
                self.write("         [yellow]>>> REQUIREMENT DETECTED[/yellow]")
            if event.payload.get("is_question"):
                self.write("         [blue]??? QUESTION[/blue]")
        elif event.event_type == EventType.ARTIFACT:
            compiles = event.payload.get("compiles", False)
            status = "[green]✓[/green]" if compiles else "[red]✗[/red]"
            self.write(f"  {status} Scaffold: {event.author}")
```

- [ ] **Step 2: Update SuggestionsPanel to refresh on events**

Replace `forgestream/tui/panels/suggestions.py`:

```python
"""Suggestions panel -- priority-sorted queue display."""

from textual.widgets import Static

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.suggestions import Priority, Suggestion, SuggestionQueue

PRIORITY_COLORS = {
    Priority.CRITICAL: "red",
    Priority.STRATEGIC: "yellow",
    Priority.DELVE_DEEPER: "blue",
    Priority.GOOD_TO_PROBE: "green",
    Priority.NICE_TO_KNOW: "dim",
}

PRIORITY_ICONS = {
    Priority.CRITICAL: "!!",
    Priority.STRATEGIC: "▲",
    Priority.DELVE_DEEPER: "◆",
    Priority.GOOD_TO_PROBE: "○",
    Priority.NICE_TO_KNOW: "·",
}


class SuggestionsPanel(Static):
    """Displays the suggestion priority queue."""

    DEFAULT_CSS = """
    SuggestionsPanel {
        height: 1fr;
        border: solid cyan;
    }
    """

    def __init__(self, queue: SuggestionQueue | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.queue = queue or SuggestionQueue()

    def on_event_received(self, event: Event) -> None:
        """Handle suggestion-related events."""
        if event.event_type == EventType.SUGGESTION:
            self.queue.add(Suggestion(
                text=event.payload.get("text", ""),
                priority_score=event.payload.get("priority", 0.5),
            ))
            self.refresh_display()

    def refresh_display(self) -> None:
        lines = ["[bold]SUGGESTION QUEUE[/bold]\n"]
        for suggestion in self.queue.get_all()[:10]:
            color = PRIORITY_COLORS.get(suggestion.category, "white")
            icon = PRIORITY_ICONS.get(suggestion.category, " ")
            lines.append(
                f"[{color}]{icon} {suggestion.category.value.upper()}[/{color}]\n"
                f"  {suggestion.text}\n"
            )
        if not self.queue.get_all():
            lines.append("[dim]No suggestions yet[/dim]")
        self.update("\n".join(lines))
```

- [ ] **Step 3: Update BranchesPanel to refresh on events**

Replace `forgestream/tui/panels/branches.py`:

```python
"""Branches panel -- conversation tree with metrics."""

from textual.widgets import Static

from forgestream.events.schema import Event, EventType
from forgestream.synthesis.branches import BranchTracker


class BranchesPanel(Static):
    """Displays conversation branches with potential/momentum/ROI."""

    DEFAULT_CSS = """
    BranchesPanel {
        height: auto;
        max-height: 8;
        border: solid magenta;
    }
    """

    def __init__(self, tracker: BranchTracker | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.tracker = tracker or BranchTracker()

    def on_event_received(self, event: Event) -> None:
        """Update branch tracking on claim events."""
        if event.event_type == EventType.CLAIM:
            keywords = event.payload.get("topic_keywords", [])
            if keywords:
                self.tracker.add_keywords(self.tracker.main_branch_id, keywords)
                self.refresh_display()

    def refresh_display(self) -> None:
        lines = ["[bold]BRANCHES[/bold]"]
        for branch in self.tracker.all_branches[:12]:
            metrics = self.tracker.get_metrics(branch.id)
            pot = metrics["potential"]
            claims = metrics["claim_count"]
            prefix = "━" * min(18, max(1, claims * 2))
            indent = "" if branch.parent_branch_id is None else "├─ "
            lines.append(
                f"  {indent}{branch.name} {prefix} pot:{pot:.2f} [{claims} claims]"
            )
        self.update("\n".join(lines))
```

- [ ] **Step 4: Update ForgeStreamApp to wire EventBus to panels**

Replace `forgestream/tui/app.py`:

```python
"""ForgeStream TUI -- primary meeting interface."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from forgestream.events.schema import Event
from forgestream.orchestrator import Orchestrator

from .panels.agents import AgentsPanel
from .panels.branches import BranchesPanel
from .panels.feed import FeedPanel
from .panels.suggestions import SuggestionsPanel


class EvaluatorBar(Static):
    """Compact evaluator display in the header area."""

    DEFAULT_CSS = """
    EvaluatorBar {
        height: 1;
        dock: top;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def update_evaluator(self, value: float, event_count: int, mode: str) -> None:
        bar_len = 20
        filled = int(value * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        self.update(
            f"E(π)={value:.3f} [{bar}] events:{event_count} mode:{mode.upper()}"
        )


class ForgeStreamApp(App):
    """ForgeStream terminal UI for live meetings."""

    TITLE = "ForgeStream"
    SUB_TITLE = "Live Meeting Intelligence"

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-content {
        layout: horizontal;
        height: 1fr;
    }
    #left-column {
        width: 60%;
    }
    #right-column {
        width: 40%;
    }
    #bottom-bar {
        height: auto;
        max-height: 12;
        layout: horizontal;
    }
    #branches-container {
        width: 60%;
    }
    #agents-container {
        width: 40%;
    }
    """

    BINDINGS = [
        Binding("m", "cycle_mode", "Mode"),
        Binding("s", "dismiss_suggestion", "Dismiss"),
        Binding("p", "pause_agents", "Pause"),
        Binding("r", "resume_agents", "Resume"),
        Binding("q", "toggle_quiet", "Quiet"),
        Binding("space", "bookmark", "Bookmark"),
        Binding("escape", "reset_view", "Reset"),
    ]

    def __init__(self, orchestrator: Orchestrator | None = None, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self._mode = "extract"
        self._quiet = False
        self._event_count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield EvaluatorBar(id="evaluator-bar")
        with Horizontal(id="main-content"):
            with Vertical(id="left-column"):
                yield FeedPanel(id="feed")
            with Vertical(id="right-column"):
                yield SuggestionsPanel(id="suggestions")
        with Horizontal(id="bottom-bar"):
            with Vertical(id="branches-container"):
                yield BranchesPanel(id="branches")
            with Vertical(id="agents-container"):
                yield AgentsPanel(id="agents")
        yield Footer()

    async def on_mount(self) -> None:
        """Subscribe to the orchestrator's event bus when mounted."""
        if self.orchestrator:
            self.orchestrator.event_bus.subscribe(self._on_event)

    async def _on_event(self, event: Event) -> None:
        """Route events from the EventBus to the appropriate panels."""
        self._event_count += 1

        # Feed panel gets all events
        feed = self.query_one("#feed", FeedPanel)
        feed.on_event_received(event)

        # Suggestions panel
        suggestions = self.query_one("#suggestions", SuggestionsPanel)
        suggestions.on_event_received(event)

        # Branches panel
        branches = self.query_one("#branches", BranchesPanel)
        branches.on_event_received(event)

        # Evaluator bar
        evaluator_bar = self.query_one("#evaluator-bar", EvaluatorBar)
        evaluator_bar.update_evaluator(
            event.evaluator, self._event_count, self._mode
        )

    def action_cycle_mode(self) -> None:
        modes = ["extract", "collaborative", "knowledge"]
        idx = modes.index(self._mode)
        self._mode = modes[(idx + 1) % len(modes)]
        self.sub_title = f"Mode: {self._mode.upper()}"

    def action_dismiss_suggestion(self) -> None:
        panel = self.query_one("#suggestions", SuggestionsPanel)
        panel.queue.dismiss()
        panel.refresh_display()

    def action_pause_agents(self) -> None:
        self.sub_title = "AGENTS PAUSED"

    def action_resume_agents(self) -> None:
        self.sub_title = f"Mode: {self._mode.upper()}"

    def action_toggle_quiet(self) -> None:
        self._quiet = not self._quiet

    def action_bookmark(self) -> None:
        feed = self.query_one("#feed", FeedPanel)
        feed.write("[bold yellow]>>> BOOKMARK <<<[/bold yellow]")

    def action_reset_view(self) -> None:
        self._quiet = False
```

- [ ] **Step 5: Create the audio meeting runner with TUI**

Create `forgestream/runner.py`:

```python
"""ForgeStream runner -- process audio through TUI with live updates.

Usage:
    python -m forgestream.runner /path/to/audio.m4a
    python -m forgestream.runner /path/to/folder/ --mode collaborative
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

from .config import ForgeStreamConfig, load_config
from .gemini.extraction import ClaimExtractor
from .orchestrator import Orchestrator
from .synthesis.requirements import RequirementDetector
from .events.schema import Event, EventType


EXTRACTION_PROMPT = """You are an ECEF knowledge extractor analyzing a recorded conversation.

Listen to the full audio and extract EVERY substantive claim made by each speaker.

For each claim, output a JSON object on its own line (JSONL format):
{{"text": "...", "speaker": "Speaker 1/Speaker 2/etc", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": true/false, "is_question": true/false, "timestamp_approx": "MM:SS"}}

Extract EVERY substantive claim. Each distinct statement gets its own JSON line.
Output ONLY the JSON lines, no other text."""


def natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def extract_claims(audio_path: Path, config: ForgeStreamConfig) -> list[dict]:
    """Extract claims from audio via Gemini Vertex AI."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=config.gemini_use_vertex,
        project=config.gemini_project,
        location=config.gemini_location,
    )

    mime_types = {
        ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".wav": "audio/wav",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    }
    mime_type = mime_types.get(audio_path.suffix.lower(), "audio/mp4")
    audio_bytes = audio_path.read_bytes()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.gemini_model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    EXTRACTION_PROMPT,
                ],
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 60 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    else:
        return []

    claims = []
    if response.text:
        for line in response.text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                claims.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return claims


async def run_with_tui(audio_input: str, config: ForgeStreamConfig) -> None:
    """Extract claims from audio and feed through orchestrator + TUI."""
    from .tui.app import ForgeStreamApp

    # Set up orchestrator
    orch = Orchestrator(config)

    # Optionally connect PostgreSQL
    store = None
    try:
        import psycopg
        conn = await psycopg.AsyncConnection.connect(config.postgres_dsn)
        from .events.store import EventStore
        store = EventStore(conn=conn)
        orch = Orchestrator(config, store=store)
    except Exception:
        pass  # PostgreSQL optional — continue without persistence

    # Optionally connect Firestore
    firestore_sync = None
    if config.firestore_enabled:
        try:
            from .firestore_sync import FirestoreSync
            firestore_sync = FirestoreSync(
                project_id=config.firebase_project,
                enabled=True,
            )
            orch = Orchestrator(config, store=store, firestore_sync=firestore_sync)
        except Exception:
            pass

    # Resolve audio files
    input_path = Path(audio_input)
    audio_extensions = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}

    if input_path.is_dir():
        audio_files = sorted(
            [f for f in input_path.iterdir() if f.suffix.lower() in audio_extensions],
            key=natural_sort_key,
        )
    elif input_path.is_file():
        audio_files = [input_path]
    else:
        print(f"Error: {audio_input} not found")
        sys.exit(1)

    # Extract claims from all audio files
    all_claims: list[dict] = []
    for i, audio_file in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Extracting from {audio_file.name}...")
        claims = extract_claims(audio_file, config)
        for claim in claims:
            claim["_source_file"] = audio_file.name
        all_claims.extend(claims)
        if i < len(audio_files) - 1:
            time.sleep(15)  # Rate limit cooldown

    print(f"Extracted {len(all_claims)} claims. Launching TUI...")

    # Create TUI app with orchestrator
    app = ForgeStreamApp(orchestrator=orch)

    # Feed claims into orchestrator after TUI mounts
    branch_id = uuid4()
    extractor = ClaimExtractor(session_id=orch.session_id, branch_id=branch_id)
    req_detector = RequirementDetector()

    async def feed_claims() -> None:
        """Feed claims into the orchestrator with delays for visual effect."""
        await asyncio.sleep(1)  # Wait for TUI to mount
        for claim_data in all_claims:
            event = extractor.parse_claim(claim_data)
            await orch.process_event(event)

            # Check for requirements and create suggestion events
            req = req_detector.check(event)
            if req:
                suggestion_event = Event(
                    event_type=EventType.SUGGESTION,
                    session_id=orch.session_id,
                    branch_id=branch_id,
                    author="synthesis",
                    evaluator=0.0,
                    payload={
                        "text": f"Scaffold: {req['description'][:60]}",
                        "priority": 0.7,
                    },
                )
                await orch.process_event(suggestion_event)

            await asyncio.sleep(0.15)  # Pace for readability

    # Run TUI with background claim feeding
    app.run_worker(feed_claims)
    await app.run_async()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ForgeStream: Audio meeting with live TUI")
    parser.add_argument("audio", help="Audio file or folder of audio files")
    parser.add_argument("--mode", choices=["extract", "collaborative", "knowledge"],
                        default="collaborative")
    args = parser.parse_args()

    config = load_config()
    config.meeting_mode = args.mode

    asyncio.run(run_with_tui(args.audio, config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write runner test**

Create `tests/test_runner.py`:

```python
"""Runner tests -- verify the TUI integration path."""

from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator
from forgestream.tui.app import ForgeStreamApp


class TestRunner:
    def test_app_accepts_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch

    def test_app_works_without_orchestrator(self):
        app = ForgeStreamApp()
        assert app.orchestrator is None
```

- [ ] **Step 7: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -v
```

Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/tui/ forgestream/runner.py tests/test_runner.py
git commit -m "feat: wire TUI to EventBus with live claim/suggestion/branch updates"
```

---

## Task 6: Integration Test — Full Pipeline

**Files:**
- Create: `tests/test_milestone_a.py`

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/test_milestone_a.py`:

```python
"""Milestone A integration test — full pipeline: claims → orchestrator → persistence → TUI."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore
from forgestream.gemini.extraction import ClaimExtractor
from forgestream.governor.evaluator import Evaluator
from forgestream.graph.materializer import GraphMaterializer
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.branches import BranchTracker
from forgestream.synthesis.requirements import RequirementDetector
from forgestream.tui.app import ForgeStreamApp


class TestMilestoneAIntegration:
    async def test_full_pipeline_with_persistence(self, db_conn):
        """Claims flow through orchestrator to PostgreSQL and event bus."""
        config = ForgeStreamConfig()
        store = EventStore(conn=db_conn)
        mock_firestore = MagicMock()

        orch = Orchestrator(config, store=store, firestore_sync=mock_firestore)

        # Simulate claims
        extractor = ClaimExtractor(session_id=orch.session_id, branch_id=uuid4())
        claims_data = [
            {"text": "We need Kafka", "confidence": 0.9, "topic_keywords": ["Kafka"],
             "is_requirement": True},
            {"text": "Latency under 100ms", "confidence": 0.85, "topic_keywords": ["latency"],
             "is_requirement": True},
            {"text": "Using Python", "confidence": 0.7, "topic_keywords": ["Python"],
             "is_requirement": False},
        ]

        bus_events = []
        async def capture(event: Event):
            bus_events.append(event)
        orch.event_bus.subscribe(capture)

        for claim_data in claims_data:
            event = extractor.parse_claim(claim_data)
            result = await orch.process_event(event)
            assert result is True

        # Verify PostgreSQL persistence
        stored = await store.get_events(session_id=orch.session_id)
        assert len(stored) == 3

        # Verify Firestore sync called
        assert mock_firestore.sync_event.call_count == 3

        # Verify event bus received all events
        assert len(bus_events) == 3

        # Verify knowledge graph builds from stored events
        graph = GraphMaterializer().materialize(stored)
        assert graph.get_concept("Kafka") is not None
        assert graph.get_concept("latency") is not None

        # Verify evaluator computed
        assert all(e.evaluator > 0 for e in bus_events)

    async def test_branch_threshold_reduces_noise(self):
        """With tuned threshold, related topics don't create branches."""
        tracker = BranchTracker()

        # Establish main topic
        for _ in range(5):
            tracker.add_keywords(tracker.main_branch_id,
                                 ["Kafka", "ingestion", "pipeline", "data", "streaming"])

        # Related sub-topic — should NOT branch
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["Kafka", "consumer", "partition"],
        )
        assert drift is None

        # Unrelated topic — SHOULD branch
        drift = tracker.check_drift(
            tracker.main_branch_id,
            new_keywords=["quantum", "physics", "entanglement"],
        )
        assert drift is not None

    def test_tui_wires_to_orchestrator(self):
        """TUI app accepts orchestrator and can receive events."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch
        assert app.title == "ForgeStream"
```

- [ ] **Step 2: Run the integration test**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_milestone_a.py -v
```

Expected: ALL PASS

- [ ] **Step 3: Run complete test suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

Expected: 150+ passed, 0 failed

- [ ] **Step 4: Commit**

```bash
cd ~/projects/forgestream
git add tests/test_milestone_a.py
git commit -m "test: add Milestone A integration test — full pipeline verification"
```

- [ ] **Step 5: Push everything**

```bash
cd ~/projects/forgestream
git push
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] Branch threshold tuned: `python3 -c "from forgestream.synthesis.branches import BranchTracker; print(BranchTracker.DRIFT_THRESHOLD)"` → `0.85`
- [ ] Firestore initialized: check https://console.firebase.google.com/project/forgestream-ai/firestore
- [ ] PostgreSQL write works: integration test passes
- [ ] Firestore sync works: mock test passes
- [ ] TUI accepts orchestrator: `python3 -c "from forgestream.tui.app import ForgeStreamApp; from forgestream.orchestrator import Orchestrator; from forgestream.config import ForgeStreamConfig; app = ForgeStreamApp(orchestrator=Orchestrator(ForgeStreamConfig())); print('OK')"`
- [ ] Full test suite: `python3 -m pytest -q` → all pass
- [ ] Runner launches: `python3 -m forgestream.runner /path/to/audio.m4a` processes audio and shows TUI

## Usage After Milestone A

```bash
# Process a single recording with live TUI
cd ~/projects/forgestream
python3 -m forgestream.runner /path/to/recording.m4a

# Process a folder of recordings
python3 -m forgestream.runner /path/to/folder/ --mode collaborative

# Events persist to PostgreSQL (local) and Firestore (cloud)
# View in Firestore: https://console.firebase.google.com/project/forgestream-ai/firestore
```
