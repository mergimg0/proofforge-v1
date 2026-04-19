# Wiring Sprint: SynthesisEngine + Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subscribe the SynthesisEngine to the Orchestrator EventBus (auto-detect requirements, contradictions, branches, seeds from live claims) and start the dashboard with a real Firestore connection during meeting sessions.

**Architecture:** SynthesisEngine subscribes to EventBus alongside TUI panels. Dashboard server starts with Firestore client injected. A unified session launcher boots all components.

**Tech Stack:** Existing ForgeStream modules, firebase-admin, FastAPI/uvicorn

**Existing codebase:** `~/projects/forgestream/` — 213 tests, Milestones A+B + Polish Sprint complete.

---

## Task 1: Wire SynthesisEngine to Orchestrator

**Files:**
- Modify: `forgestream/orchestrator.py`
- Create: `tests/test_synthesis_wiring.py`

- [ ] **Step 1: Write failing test**

`tests/test_synthesis_wiring.py`:
```python
"""Test SynthesisEngine wiring to Orchestrator EventBus."""

from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.orchestrator import Orchestrator
from forgestream.synthesis.engine import SynthesisEngine


class TestSynthesisWiring:
    async def test_engine_receives_events_from_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)

        # Subscribe engine to event bus
        orch.event_bus.subscribe(engine.on_event)

        # Process a claim that contains a requirement pattern
        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={
                "text": "The system must handle 10k events per second",
                "is_requirement": False,
                "topic_keywords": ["throughput", "events"],
                "confidence": 0.9,
            },
        )
        await orch.process_event(event)

        # Engine should have accumulated the claim
        assert len(engine._claim_events) >= 1

    async def test_engine_emits_requirement_back_to_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        orch.event_bus.subscribe(engine.on_event)

        all_events = []

        async def capture(event: Event):
            all_events.append(event)

        orch.event_bus.subscribe(capture)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.0,
            payload={
                "text": "We need a real-time dashboard for monitoring",
                "is_requirement": True,
                "topic_keywords": ["dashboard", "monitoring"],
                "confidence": 0.85,
            },
        )
        await orch.process_event(event)

        # Should have the original claim + a requirement event from the engine
        req_events = [e for e in all_events if e.event_type == EventType.REQUIREMENT]
        assert len(req_events) >= 1
        assert req_events[0].author == "synthesis_engine"

    async def test_engine_ignores_own_events(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = SynthesisEngine(orchestrator=orch)
        orch.event_bus.subscribe(engine.on_event)

        # Process a requirement event authored by synthesis_engine
        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis_engine",
            evaluator=0.5,
            payload={"description": "already processed"},
        )
        await orch.process_event(event)

        # Engine should NOT have processed this (it ignores self-authored events)
        assert len(engine._claim_events) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_synthesis_wiring.py -v
```

Expected: Tests should actually PASS since SynthesisEngine already has the logic — we're just verifying the wiring works. If any fail, debug.

- [ ] **Step 3: Add convenience method to Orchestrator**

Add to `forgestream/orchestrator.py`, in the `Orchestrator` class:

```python
    def attach_synthesis_engine(self) -> "SynthesisEngine":
        """Create and attach a SynthesisEngine to this orchestrator's EventBus."""
        from .synthesis.engine import SynthesisEngine
        engine = SynthesisEngine(orchestrator=self)
        self.event_bus.subscribe(engine.on_event)
        return engine
```

- [ ] **Step 4: Run tests**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_synthesis_wiring.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q -k "not writes_to_store and not milestone_a"
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/orchestrator.py tests/test_synthesis_wiring.py
git commit -m "feat: wire SynthesisEngine to Orchestrator EventBus"
```

---

## Task 2: Start Dashboard with Firestore Client

**Files:**
- Create: `forgestream/dashboard/launcher.py`
- Create: `tests/dashboard/test_launcher.py`

- [ ] **Step 1: Write failing test**

`tests/dashboard/test_launcher.py`:
```python
"""Test dashboard launcher with Firestore."""

from unittest.mock import MagicMock, patch

from forgestream.config import ForgeStreamConfig
from forgestream.dashboard.launcher import create_live_app


class TestDashboardLauncher:
    def test_creates_app_without_firestore(self):
        config = ForgeStreamConfig(firestore_enabled=False)
        app = create_live_app(config)
        assert app is not None

    @patch("forgestream.dashboard.launcher.firebase_admin")
    @patch("forgestream.dashboard.launcher.firestore")
    def test_creates_app_with_firestore(self, mock_firestore, mock_admin):
        mock_db = MagicMock()
        mock_firestore.client.return_value = mock_db

        config = ForgeStreamConfig(firestore_enabled=True)
        app = create_live_app(config)
        assert app is not None

    def test_app_serves_health(self):
        from fastapi.testclient import TestClient

        config = ForgeStreamConfig(firestore_enabled=False)
        app = create_live_app(config)
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/dashboard/test_launcher.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement dashboard launcher**

`forgestream/dashboard/launcher.py`:
```python
"""Dashboard launcher — creates a FastAPI app with optional Firestore connection."""

from __future__ import annotations

import logging

from ..config import ForgeStreamConfig
from .server import create_app

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    firebase_admin = None  # type: ignore
    firestore = None  # type: ignore


def create_live_app(config: ForgeStreamConfig) -> "FastAPI":
    """Create the dashboard app, optionally connected to Firestore.

    If Firestore is enabled and firebase-admin is installed, the dashboard
    reads live event data from Firestore. Otherwise, it serves empty responses.
    """
    db = None

    if config.firestore_enabled and HAS_FIREBASE:
        try:
            # Check if already initialized
            try:
                app = firebase_admin.get_app()
            except ValueError:
                app = firebase_admin.initialize_app(
                    credential=credentials.ApplicationDefault(),
                    options={"projectId": config.firebase_project},
                )
            db = firestore.client()
            logger.info("Dashboard connected to Firestore (project: %s)", config.firebase_project)
        except Exception as e:
            logger.warning("Dashboard Firestore connection failed: %s", e)

    return create_app(firestore_db=db)
```

- [ ] **Step 4: Run tests**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/dashboard/test_launcher.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q -k "not writes_to_store and not milestone_a"
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/dashboard/launcher.py tests/dashboard/test_launcher.py
git commit -m "feat: add dashboard launcher with Firestore connection"
```

---

## Task 3: Unified Session Launcher

**Files:**
- Modify: `forgestream/__main__.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing test**

`tests/test_session.py`:
```python
"""Test unified session launcher."""

from forgestream.config import ForgeStreamConfig
from forgestream.orchestrator import Orchestrator


class TestSessionSetup:
    def test_attach_synthesis_engine(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = orch.attach_synthesis_engine()
        assert engine is not None
        assert engine.orchestrator is orch
        # Engine should be subscribed to event bus
        assert len(orch.event_bus._subscribers) >= 1

    def test_full_session_components(self):
        """All session components can be created together."""
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        engine = orch.attach_synthesis_engine()

        # Verify engine is wired
        assert engine.orchestrator is orch

        # Verify we can create the TUI
        from forgestream.tui.app import ForgeStreamApp
        app = ForgeStreamApp(orchestrator=orch)
        assert app.orchestrator is orch
```

- [ ] **Step 2: Run test**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_session.py -v
```

Expected: ALL PASS

- [ ] **Step 3: Update __main__.py start command**

Update the `_cmd_start` function in `forgestream/__main__.py`:

```python
def _cmd_start(args: argparse.Namespace) -> int:
    from .config import load_config
    from .orchestrator import Orchestrator

    config = load_config()
    config.meeting_mode = args.mode
    if args.name:
        config.meeting_name = args.name

    print(f"ForgeStream starting...")
    print(f"  Mode: {config.meeting_mode}")
    print(f"  Name: {config.meeting_name or '(unnamed)'}")
    print(f"  PostgreSQL: {config.postgres_dsn}")
    print(f"  Firestore: {'enabled' if config.firestore_enabled else 'disabled'}")

    # Create orchestrator with synthesis engine
    orch = Orchestrator(config)
    engine = orch.attach_synthesis_engine()
    print(f"  SynthesisEngine: attached")

    # Start dashboard if requested
    if args.dashboard:
        from .dashboard.launcher import create_live_app
        import threading
        import uvicorn

        dashboard_app = create_live_app(config)

        def run_dashboard():
            uvicorn.run(
                dashboard_app,
                host=config.dashboard_host,
                port=config.dashboard_port,
                log_level="warning",
            )

        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        print(f"  Dashboard: http://{config.dashboard_host}:{config.dashboard_port}")

    print(f"\n  Ready. Use the runner to process audio:")
    print(f"  python3 -m forgestream.runner <audio> --mode {config.meeting_mode}")
    return 0
```

- [ ] **Step 4: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q -k "not writes_to_store and not milestone_a"
```

- [ ] **Step 5: Commit and push**

```bash
cd ~/projects/forgestream
git add forgestream/__main__.py tests/test_session.py
git commit -m "feat: unified session launcher with synthesis engine + dashboard"
git push
```

---

## Verification

After all tasks:

- [ ] `python3 -c "from forgestream.orchestrator import Orchestrator; from forgestream.config import ForgeStreamConfig; o = Orchestrator(ForgeStreamConfig()); e = o.attach_synthesis_engine(); print('Engine attached:', e is not None)"` → `Engine attached: True`
- [ ] `python3 -c "from forgestream.dashboard.launcher import create_live_app; from forgestream.config import ForgeStreamConfig; app = create_live_app(ForgeStreamConfig(firestore_enabled=False)); print('Dashboard:', app.title)"` → `Dashboard: ForgeStream Dashboard`
- [ ] Full test suite passes
