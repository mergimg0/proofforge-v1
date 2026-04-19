# ForgeStream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ForgeStream — a live meeting intelligence system with SOS-governed autonomous Claude Code agent swarms, powered by Gemini Live API and an append-only ECEF event log.

**Architecture:** Event-driven, append-only log as single coordination mechanism. Gemini Live API writes claim events. Claude Code CLI agents (research + scaffold) read events and write findings/artifacts. SOS axioms enforced at runtime on every write. TUI + web dashboard for user interface.

**Tech Stack:** Python 3.12+, PostgreSQL (psycopg 3, async), google-genai SDK, textual (TUI), FastAPI + D3.js (dashboard), pytest + pytest-asyncio (testing)

**Design Spec:** `~/projects/proofforge/docs/superpowers/specs/2026-03-27-forgestream-design.md`

---

## Dependency Graph

```
SP-1 (Event System) ─────┬──→ SP-2 (Knowledge Graph)
                          ├──→ SP-3 (Gemini Live API)
                          ├──→ SP-5 (Agent Spawner)
                          └──→ SP-9 (SOS Governor)

SP-1 + SP-2 ──→ SP-4 (Synthesis Engine)
SP-5 ──→ SP-6 (Agent Templates)
SP-1 + SP-4 ──→ SP-7 (Terminal TUI)
SP-1 + SP-2 ──→ SP-8 (Web Dashboard)
SP-9 ──→ SP-10 (Self-Improvement)
SP-3 + SP-4 ──→ SP-11 (Audio Copilot)
SP-2 + SP-9 ──→ SP-12 (Cross-Meeting Transfer)
```

## Progress Tracker

- [x] SP-1: Event System (completed 2026-03-27)
- [x] SP-2: Knowledge Graph (completed 2026-03-27)
- [x] SP-3: Gemini Live API (completed 2026-03-27)
- [x] SP-5: Agent Spawner (completed 2026-03-27)
- [x] SP-9: SOS Governor (completed 2026-03-27)
- [x] SP-4: Synthesis Engine (completed 2026-03-27)
- [x] SP-6: Agent Templates (completed 2026-03-27)
- [x] SP-7: Terminal TUI (completed 2026-03-27)
- [x] SP-8: Web Dashboard (completed 2026-03-27)
- [x] SP-10: Self-Improvement (completed 2026-03-27)
- [x] SP-11: Audio Copilot (completed 2026-03-27)
- [x] SP-12: Cross-Meeting Transfer (completed 2026-03-27)
- [x] SP-13: Configuration + CLI Entry Point (completed 2026-03-27)
- [x] SP-14: Orchestrator (completed 2026-03-27)
- [x] SP-15: Integration Testing (completed 2026-03-27)

## Blockers

(Document any blockers encountered during execution here)

---

## SP-1: Event System

The foundation. Append-only PostgreSQL event log with LISTEN/NOTIFY subscriptions.

### File Structure

```
~/projects/forgestream/
├── pyproject.toml
├── forgestream/
│   ├── __init__.py
│   └── events/
│       ├── __init__.py
│       ├── schema.py       # Event dataclass + EventType enum
│       ├── store.py        # EventStore (async PostgreSQL writer/reader)
│       └── subscribe.py    # EventSubscriber (LISTEN/NOTIFY)
├── migrations/
│   └── 001_create_events.sql
└── tests/
    ├── __init__.py
    ├── conftest.py         # DB fixtures
    └── events/
        ├── __init__.py
        ├── test_schema.py
        ├── test_store.py
        └── test_subscribe.py
```

### Task 1.1: Project Bootstrap

**Files:**
- Create: `~/projects/forgestream/pyproject.toml`
- Create: `~/projects/forgestream/forgestream/__init__.py`
- Create: `~/projects/forgestream/forgestream/events/__init__.py`
- Create: `~/projects/forgestream/tests/__init__.py`
- Create: `~/projects/forgestream/tests/events/__init__.py`

- [ ] **Step 1: Create project directory and initialize git**

```bash
mkdir -p ~/projects/forgestream
cd ~/projects/forgestream
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "forgestream"
version = "0.1.0"
description = "Live meeting intelligence with SOS-governed agent swarms"
requires-python = ">=3.12"
dependencies = [
    "psycopg[binary]>=3.2,<4",
    "pydantic>=2.0,<3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
]
gemini = [
    "google-genai>=1.0",
]
tui = [
    "textual>=0.80",
]
dashboard = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
]
all = [
    "forgestream[dev,gemini,tui,dashboard]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create package init files**

`forgestream/__init__.py`:
```python
"""ForgeStream: Live meeting intelligence with SOS-governed agent swarms."""
```

`forgestream/events/__init__.py`:
```python
"""ECEF append-only event system."""
from .schema import Event, EventType
from .store import EventStore
from .subscribe import EventSubscriber

__all__ = ["Event", "EventType", "EventStore", "EventSubscriber"]
```

`tests/__init__.py` and `tests/events/__init__.py`: empty files.

- [ ] **Step 4: Install in dev mode**

```bash
cd ~/projects/forgestream
pip install -e ".[dev]"
```

- [ ] **Step 5: Verify pytest runs (no tests yet)**

```bash
cd ~/projects/forgestream
pytest --co -q
```

Expected: `no tests ran`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml forgestream/ tests/
git commit -m "feat(SP-1): bootstrap forgestream project"
```

### Task 1.2: Event Schema

**Files:**
- Create: `forgestream/events/schema.py`
- Create: `tests/events/test_schema.py`

- [ ] **Step 1: Write failing tests for Event schema**

`tests/events/test_schema.py`:
```python
from datetime import datetime, timezone
from uuid import UUID, uuid4

from forgestream.events.schema import Event, EventType


class TestEventType:
    def test_all_event_types_exist(self):
        expected = [
            "claim", "contradiction", "requirement", "verified_finding",
            "artifact", "suggestion", "branch_point", "seed",
            "evaluator_snapshot", "mode_switch", "merge", "meeting_summary",
        ]
        for t in expected:
            assert EventType(t) is not None

    def test_event_type_is_string(self):
        assert EventType.CLAIM == "claim"
        assert EventType.ARTIFACT == "artifact"


class TestEvent:
    def test_create_event_with_required_fields(self):
        session_id = uuid4()
        branch_id = uuid4()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=session_id,
            branch_id=branch_id,
            author="gemini",
            evaluator=0.5,
            payload={"text": "Expert said X", "confidence": 0.9},
        )
        assert isinstance(event.id, UUID)
        assert event.session_id == session_id
        assert event.branch_id == branch_id
        assert event.event_type == EventType.CLAIM
        assert event.author == "gemini"
        assert event.evaluator == 0.5
        assert event.payload["text"] == "Expert said X"
        assert isinstance(event.timestamp, datetime)
        assert event.parent_id is None
        assert event.degradation_flag is False
        assert event.trust_region_ok is True

    def test_create_event_with_parent(self):
        parent_id = uuid4()
        event = Event(
            event_type=EventType.CONTRADICTION,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.6,
            payload={"claim_a_id": str(uuid4()), "claim_b_id": str(uuid4())},
            parent_id=parent_id,
        )
        assert event.parent_id == parent_id

    def test_create_event_with_degradation(self):
        event = Event(
            event_type=EventType.EVALUATOR_SNAPSHOT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="governor",
            evaluator=0.3,
            payload={"E_micro": 0.3},
            degradation_flag=True,
            trust_region_ok=False,
        )
        assert event.degradation_flag is True
        assert event.trust_region_ok is False

    def test_event_to_dict(self):
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test"},
        )
        d = event.to_dict()
        assert d["event_type"] == "claim"
        assert d["author"] == "gemini"
        assert isinstance(d["id"], str)
        assert isinstance(d["session_id"], str)

    def test_event_from_dict(self):
        event_id = uuid4()
        session_id = uuid4()
        branch_id = uuid4()
        now = datetime.now(timezone.utc)
        d = {
            "id": str(event_id),
            "event_type": "claim",
            "session_id": str(session_id),
            "branch_id": str(branch_id),
            "author": "gemini",
            "evaluator": 0.7,
            "payload": {"text": "hello"},
            "timestamp": now.isoformat(),
            "parent_id": None,
            "degradation_flag": False,
            "trust_region_ok": True,
        }
        event = Event.from_dict(d)
        assert event.id == event_id
        assert event.event_type == EventType.CLAIM
        assert event.session_id == session_id
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
pytest tests/events/test_schema.py -v
```

Expected: FAIL — `ImportError: cannot import name 'Event' from 'forgestream.events.schema'`

- [ ] **Step 3: Implement Event schema**

`forgestream/events/schema.py`:
```python
"""ECEF event schema — the atomic unit of ForgeStream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    """All event types in the ECEF event log."""

    CLAIM = "claim"
    CONTRADICTION = "contradiction"
    REQUIREMENT = "requirement"
    VERIFIED_FINDING = "verified_finding"
    ARTIFACT = "artifact"
    SUGGESTION = "suggestion"
    BRANCH_POINT = "branch_point"
    SEED = "seed"
    EVALUATOR_SNAPSHOT = "evaluator_snapshot"
    MODE_SWITCH = "mode_switch"
    MERGE = "merge"
    MEETING_SUMMARY = "meeting_summary"


@dataclass
class Event:
    """An immutable event in the ECEF append-only log.

    Once written, events are never updated or deleted.
    """

    event_type: EventType
    session_id: UUID
    branch_id: UUID
    author: str
    evaluator: float
    payload: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    parent_id: UUID | None = None
    degradation_flag: bool = False
    trust_region_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "session_id": str(self.session_id),
            "branch_id": str(self.branch_id),
            "author": self.author,
            "evaluator": self.evaluator,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "degradation_flag": self.degradation_flag,
            "trust_region_ok": self.trust_region_ok,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Deserialize from a dict (e.g., from database row)."""
        return cls(
            id=UUID(d["id"]),
            event_type=EventType(d["event_type"]),
            session_id=UUID(d["session_id"]),
            branch_id=UUID(d["branch_id"]),
            author=d["author"],
            evaluator=d["evaluator"],
            payload=d["payload"],
            timestamp=(
                datetime.fromisoformat(d["timestamp"])
                if isinstance(d["timestamp"], str)
                else d["timestamp"]
            ),
            parent_id=UUID(d["parent_id"]) if d.get("parent_id") else None,
            degradation_flag=d.get("degradation_flag", False),
            trust_region_ok=d.get("trust_region_ok", True),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
pytest tests/events/test_schema.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/events/schema.py tests/events/test_schema.py
git commit -m "feat(SP-1): add Event schema and EventType enum"
```

### Task 1.3: Database Migration

**Files:**
- Create: `migrations/001_create_events.sql`

- [ ] **Step 1: Write the migration SQL**

`migrations/001_create_events.sql`:
```sql
-- ForgeStream ECEF Event Log
-- APPEND-ONLY: no updates, no deletes, ever.

CREATE TABLE IF NOT EXISTS events (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type       TEXT NOT NULL,
    parent_id        UUID REFERENCES events(id),
    branch_id        UUID NOT NULL,
    author           TEXT NOT NULL,
    evaluator        FLOAT NOT NULL,
    payload          JSONB NOT NULL,
    degradation_flag BOOLEAN DEFAULT FALSE,
    trust_region_ok  BOOLEAN DEFAULT TRUE
);

-- Append-only enforcement at database level
CREATE OR REPLACE RULE no_updates AS ON UPDATE TO events DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_deletes AS ON DELETE TO events DO INSTEAD NOTHING;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id);
CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_id);

-- GIN index for JSONB payload queries
CREATE INDEX IF NOT EXISTS idx_events_payload ON events USING GIN (payload);
```

- [ ] **Step 2: Create the forgestream database and run migration**

```bash
docker exec continuous-claude-postgres psql -U claude -d postgres -c \
  "CREATE DATABASE forgestream OWNER claude;" 2>/dev/null || true

docker exec -i continuous-claude-postgres psql -U claude -d forgestream \
  < ~/projects/forgestream/migrations/001_create_events.sql
```

- [ ] **Step 3: Verify the table exists and rules are applied**

```bash
docker exec continuous-claude-postgres psql -U claude -d forgestream -c \
  "SELECT tablename FROM pg_tables WHERE schemaname='public';"

docker exec continuous-claude-postgres psql -U claude -d forgestream -c \
  "SELECT rulename FROM pg_rules WHERE tablename='events';"
```

Expected: table `events` exists, rules `no_updates` and `no_deletes` exist.

- [ ] **Step 4: Commit**

```bash
cd ~/projects/forgestream
git add migrations/
git commit -m "feat(SP-1): add PostgreSQL migration with append-only rules"
```

### Task 1.4: Event Store

**Files:**
- Create: `forgestream/events/store.py`
- Create: `tests/conftest.py`
- Create: `tests/events/test_store.py`

- [ ] **Step 1: Create test fixtures**

`tests/conftest.py`:
```python
import asyncio
from uuid import uuid4

import psycopg
import pytest


TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_conn():
    conn = await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=False)
    yield conn
    await conn.rollback()
    await conn.close()


@pytest.fixture
def session_id():
    return uuid4()


@pytest.fixture
def branch_id():
    return uuid4()
```

- [ ] **Step 2: Write failing tests for EventStore**

`tests/events/test_store.py`:
```python
from uuid import uuid4

import pytest

from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore

TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


class TestEventStore:
    @pytest.fixture
    async def store(self, db_conn):
        s = EventStore(conn=db_conn)
        yield s

    async def test_append_and_retrieve(self, store, session_id, branch_id):
        event = Event(
            event_type=EventType.CLAIM,
            session_id=session_id,
            branch_id=branch_id,
            author="gemini",
            evaluator=0.5,
            payload={"text": "Test claim", "confidence": 0.9},
        )
        stored = await store.append(event)
        assert stored.id == event.id

        events = await store.get_events(session_id=session_id)
        assert len(events) == 1
        assert events[0].id == event.id
        assert events[0].payload["text"] == "Test claim"

    async def test_append_multiple_events(self, store, session_id, branch_id):
        for i in range(5):
            event = Event(
                event_type=EventType.CLAIM,
                session_id=session_id,
                branch_id=branch_id,
                author="gemini",
                evaluator=0.5 + i * 0.05,
                payload={"text": f"Claim {i}"},
            )
            await store.append(event)

        events = await store.get_events(session_id=session_id)
        assert len(events) == 5
        # Events should be ordered by timestamp
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    async def test_filter_by_event_type(self, store, session_id, branch_id):
        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_id, author="gemini", evaluator=0.5,
            payload={"text": "a claim"},
        ))
        await store.append(Event(
            event_type=EventType.SUGGESTION, session_id=session_id,
            branch_id=branch_id, author="synthesis", evaluator=0.6,
            payload={"text": "a suggestion", "priority": 0.8},
        ))

        claims = await store.get_events(
            session_id=session_id, event_type=EventType.CLAIM
        )
        assert len(claims) == 1
        assert claims[0].event_type == EventType.CLAIM

    async def test_filter_by_branch(self, store, session_id):
        branch_a = uuid4()
        branch_b = uuid4()

        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_a, author="gemini", evaluator=0.5,
            payload={"text": "branch a"},
        ))
        await store.append(Event(
            event_type=EventType.CLAIM, session_id=session_id,
            branch_id=branch_b, author="gemini", evaluator=0.5,
            payload={"text": "branch b"},
        ))

        events = await store.get_events(
            session_id=session_id, branch_id=branch_a
        )
        assert len(events) == 1
        assert events[0].payload["text"] == "branch a"

    async def test_get_latest_evaluator(self, store, session_id, branch_id):
        for val in [0.3, 0.5, 0.7]:
            await store.append(Event(
                event_type=EventType.CLAIM, session_id=session_id,
                branch_id=branch_id, author="gemini", evaluator=val,
                payload={"text": "x"},
            ))

        latest = await store.get_latest_evaluator(session_id)
        assert latest == 0.7

    async def test_count_by_type(self, store, session_id, branch_id):
        for _ in range(3):
            await store.append(Event(
                event_type=EventType.CLAIM, session_id=session_id,
                branch_id=branch_id, author="g", evaluator=0.5,
                payload={},
            ))
        await store.append(Event(
            event_type=EventType.ARTIFACT, session_id=session_id,
            branch_id=branch_id, author="scaffold", evaluator=0.6,
            payload={},
        ))

        counts = await store.count_by_type(session_id)
        assert counts[EventType.CLAIM] == 3
        assert counts[EventType.ARTIFACT] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
pytest tests/events/test_store.py -v
```

Expected: FAIL — `ImportError: cannot import name 'EventStore'`

- [ ] **Step 4: Implement EventStore**

`forgestream/events/store.py`:
```python
"""Async PostgreSQL event store — append-only by design."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from .schema import Event, EventType

if TYPE_CHECKING:
    import psycopg


class EventStore:
    """Append-only event store backed by PostgreSQL."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self.conn = conn

    async def append(self, event: Event) -> Event:
        """Append an event to the log. The only write operation."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """INSERT INTO events
                   (id, session_id, timestamp, event_type, parent_id,
                    branch_id, author, evaluator, payload,
                    degradation_flag, trust_region_ok)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(event.id),
                    str(event.session_id),
                    event.timestamp,
                    event.event_type.value,
                    str(event.parent_id) if event.parent_id else None,
                    str(event.branch_id),
                    event.author,
                    event.evaluator,
                    json.dumps(event.payload),
                    event.degradation_flag,
                    event.trust_region_ok,
                ),
            )
        return event

    async def get_events(
        self,
        session_id: UUID,
        event_type: EventType | None = None,
        branch_id: UUID | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Query events with optional filters. Read-only."""
        conditions = ["session_id = %s"]
        params: list = [str(session_id)]

        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(event_type.value)
        if branch_id is not None:
            conditions.append("branch_id = %s")
            params.append(str(branch_id))
        if after is not None:
            conditions.append("timestamp > %s")
            params.append(after)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM events WHERE {where} ORDER BY timestamp ASC"
        if limit is not None:
            query += f" LIMIT {limit}"

        async with self.conn.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            cols = [desc.name for desc in cur.description]

        return [self._row_to_event(dict(zip(cols, row))) for row in rows]

    async def get_latest_evaluator(self, session_id: UUID) -> float | None:
        """Get the most recent evaluator value for a session."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT evaluator FROM events
                   WHERE session_id = %s
                   ORDER BY timestamp DESC LIMIT 1""",
                (str(session_id),),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def count_by_type(self, session_id: UUID) -> Counter[EventType]:
        """Count events by type for a session."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT event_type, COUNT(*) FROM events
                   WHERE session_id = %s GROUP BY event_type""",
                (str(session_id),),
            )
            rows = await cur.fetchall()
        return Counter({EventType(row[0]): row[1] for row in rows})

    @staticmethod
    def _row_to_event(row: dict) -> Event:
        """Convert a database row to an Event."""
        return Event(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            event_type=EventType(row["event_type"]),
            session_id=(
                row["session_id"]
                if isinstance(row["session_id"], UUID)
                else UUID(row["session_id"])
            ),
            branch_id=(
                row["branch_id"]
                if isinstance(row["branch_id"], UUID)
                else UUID(row["branch_id"])
            ),
            author=row["author"],
            evaluator=row["evaluator"],
            payload=(
                row["payload"]
                if isinstance(row["payload"], dict)
                else json.loads(row["payload"])
            ),
            timestamp=row["timestamp"],
            parent_id=(
                (
                    row["parent_id"]
                    if isinstance(row["parent_id"], UUID)
                    else UUID(row["parent_id"])
                )
                if row.get("parent_id")
                else None
            ),
            degradation_flag=row.get("degradation_flag", False),
            trust_region_ok=row.get("trust_region_ok", True),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
pytest tests/events/test_store.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/events/store.py tests/conftest.py tests/events/test_store.py
git commit -m "feat(SP-1): add async EventStore with PostgreSQL backend"
```

### Task 1.5: Event Subscriber

**Files:**
- Create: `forgestream/events/subscribe.py`
- Create: `tests/events/test_subscribe.py`

- [ ] **Step 1: Write failing tests for EventSubscriber**

`tests/events/test_subscribe.py`:
```python
import asyncio
import json
from uuid import uuid4

import psycopg
import pytest

from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore
from forgestream.events.subscribe import EventSubscriber

TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


class TestEventSubscriber:
    async def test_subscribe_receives_events(self):
        received = []

        async def on_event(event_type: str, payload: dict):
            received.append((event_type, payload))

        subscriber = EventSubscriber(TEST_DSN)
        await subscriber.start(channels=["claim"])

        subscriber.on_event = on_event

        # Write an event from a separate connection to trigger NOTIFY
        conn = await psycopg.AsyncConnection.connect(TEST_DSN)
        async with conn.cursor() as cur:
            event_data = json.dumps({
                "id": str(uuid4()),
                "event_type": "claim",
                "session_id": str(uuid4()),
            })
            await cur.execute(
                "SELECT pg_notify(%s, %s)", ("event_claim", event_data)
            )
        await conn.commit()
        await conn.close()

        # Give LISTEN/NOTIFY time to deliver
        await asyncio.sleep(0.5)
        await subscriber.poll()

        assert len(received) >= 1
        assert received[0][0] == "claim"

        await subscriber.stop()

    async def test_subscribe_multiple_channels(self):
        received = []

        async def on_event(event_type: str, payload: dict):
            received.append(event_type)

        subscriber = EventSubscriber(TEST_DSN)
        await subscriber.start(channels=["claim", "artifact"])
        subscriber.on_event = on_event

        conn = await psycopg.AsyncConnection.connect(TEST_DSN)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT pg_notify(%s, %s)",
                ("event_claim", json.dumps({"event_type": "claim"})),
            )
            await cur.execute(
                "SELECT pg_notify(%s, %s)",
                ("event_artifact", json.dumps({"event_type": "artifact"})),
            )
        await conn.commit()
        await conn.close()

        await asyncio.sleep(0.5)
        await subscriber.poll()

        assert "claim" in received
        assert "artifact" in received

        await subscriber.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
pytest tests/events/test_subscribe.py -v
```

Expected: FAIL — `ImportError: cannot import name 'EventSubscriber'`

- [ ] **Step 3: Implement EventSubscriber**

`forgestream/events/subscribe.py`:
```python
"""Event subscription via PostgreSQL LISTEN/NOTIFY."""

from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

import psycopg


class EventSubscriber:
    """Subscribes to event channels via PostgreSQL LISTEN/NOTIFY."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn: psycopg.AsyncConnection | None = None
        self.on_event: (
            Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None
        ) = None
        self._channels: list[str] = []

    async def start(self, channels: list[str]) -> None:
        """Connect and LISTEN on the given channels.

        Channel names are prefixed with 'event_' to namespace them.
        """
        self._channels = channels
        self.conn = await psycopg.AsyncConnection.connect(
            self.dsn, autocommit=True
        )
        for channel in channels:
            await self.conn.execute(f"LISTEN event_{channel}")

    async def poll(self) -> None:
        """Check for pending notifications and dispatch them."""
        if self.conn is None:
            return

        async for notify in self.conn.notifies():
            if self.on_event is not None:
                payload = json.loads(notify.payload) if notify.payload else {}
                event_type = payload.get(
                    "event_type",
                    notify.channel.removeprefix("event_"),
                )
                await self.on_event(event_type, payload)
            # Break after processing available notifications
            # (notifies() blocks if we don't)
            break

    async def poll_continuous(self) -> None:
        """Continuously poll for notifications. Blocking."""
        if self.conn is None:
            return

        async for notify in self.conn.notifies():
            if self.on_event is not None:
                payload = json.loads(notify.payload) if notify.payload else {}
                event_type = payload.get(
                    "event_type",
                    notify.channel.removeprefix("event_"),
                )
                await self.on_event(event_type, payload)

    async def stop(self) -> None:
        """Disconnect and stop listening."""
        if self.conn:
            for channel in self._channels:
                await self.conn.execute(f"UNLISTEN event_{channel}")
            await self.conn.close()
            self.conn = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
pytest tests/events/test_subscribe.py -v
```

Expected: All PASS

- [ ] **Step 5: Run full test suite — no regressions**

```bash
cd ~/projects/forgestream
pytest -v
```

Expected: All tests pass (schema + store + subscribe)

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/events/subscribe.py tests/events/test_subscribe.py
git commit -m "feat(SP-1): add EventSubscriber with LISTEN/NOTIFY"
```

### Task 1.6: Store NOTIFY Integration

The EventStore needs to emit NOTIFY when appending events, so subscribers receive them.

**Files:**
- Modify: `forgestream/events/store.py`
- Create: `tests/events/test_store_notify.py`

- [ ] **Step 1: Write failing test for NOTIFY on append**

`tests/events/test_store_notify.py`:
```python
import asyncio
import json
from uuid import uuid4

import psycopg
import pytest

from forgestream.events.schema import Event, EventType
from forgestream.events.store import EventStore
from forgestream.events.subscribe import EventSubscriber

TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


class TestStoreNotify:
    async def test_append_triggers_notify(self):
        received = []

        async def on_event(event_type: str, payload: dict):
            received.append(event_type)

        subscriber = EventSubscriber(TEST_DSN)
        await subscriber.start(channels=["claim"])
        subscriber.on_event = on_event

        # Append via a separate connection (store needs its own)
        conn = await psycopg.AsyncConnection.connect(TEST_DSN)
        store = EventStore(conn=conn)

        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={"text": "test"},
        )
        await store.append(event)
        await conn.commit()

        await asyncio.sleep(0.5)
        await subscriber.poll()

        assert "claim" in received

        await subscriber.stop()
        await conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/projects/forgestream
pytest tests/events/test_store_notify.py -v
```

Expected: FAIL — append doesn't trigger NOTIFY yet

- [ ] **Step 3: Add NOTIFY to EventStore.append**

Add to `forgestream/events/store.py`, at the end of the `append` method, after the INSERT:

```python
    async def append(self, event: Event) -> Event:
        """Append an event to the log. The only write operation."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """INSERT INTO events
                   (id, session_id, timestamp, event_type, parent_id,
                    branch_id, author, evaluator, payload,
                    degradation_flag, trust_region_ok)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(event.id),
                    str(event.session_id),
                    event.timestamp,
                    event.event_type.value,
                    str(event.parent_id) if event.parent_id else None,
                    str(event.branch_id),
                    event.author,
                    event.evaluator,
                    json.dumps(event.payload),
                    event.degradation_flag,
                    event.trust_region_ok,
                ),
            )
            # Notify subscribers
            notification = json.dumps({
                "id": str(event.id),
                "event_type": event.event_type.value,
                "session_id": str(event.session_id),
                "author": event.author,
            })
            channel = f"event_{event.event_type.value}"
            await cur.execute(
                "SELECT pg_notify(%s, %s)", (channel, notification)
            )
        return event
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/projects/forgestream
pytest tests/events/test_store_notify.py -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
cd ~/projects/forgestream
pytest -v
```

Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/events/store.py tests/events/test_store_notify.py
git commit -m "feat(SP-1): integrate NOTIFY into EventStore.append"
```

**SP-1 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-2: Knowledge Graph

Materialized view over the event log. Three entity types, four edge types. Rebuilt from events on startup.

### File Structure

```
forgestream/graph/
├── __init__.py
├── model.py          # Concept, Requirement, Artifact, Edge types
├── materializer.py   # EventLog → Graph (event sourcing)
└── query.py          # Graph queries (neighbors, paths, clusters)

tests/graph/
├── __init__.py
├── test_model.py
├── test_materializer.py
└── test_query.py
```

### Task 2.1: Graph Model

**Files:**
- Create: `forgestream/graph/__init__.py`
- Create: `forgestream/graph/model.py`
- Create: `tests/graph/__init__.py`
- Create: `tests/graph/test_model.py`

- [ ] **Step 1: Write failing tests for graph model**

`tests/graph/test_model.py`:
```python
from uuid import uuid4

from forgestream.graph.model import (
    Artifact,
    Concept,
    EdgeType,
    KnowledgeGraph,
    Requirement,
    RequirementStatus,
)


class TestConcept:
    def test_create_concept(self):
        c = Concept(name="Kafka Streams", domain="data-engineering", confidence=0.85)
        assert c.name == "Kafka Streams"
        assert c.verified is False
        assert c.source_events == []

    def test_concept_mark_verified(self):
        c = Concept(name="test", domain="test", confidence=0.9)
        c.verified = True
        assert c.verified is True


class TestRequirement:
    def test_create_requirement(self):
        r = Requirement(
            description="Sub-100ms ingestion pipeline",
            domain="data-engineering",
            complexity_estimate=0.7,
        )
        assert r.status == RequirementStatus.DETECTED
        assert r.linked_claims == []

    def test_requirement_status_transitions(self):
        r = Requirement(description="x", domain="x", complexity_estimate=0.5)
        r.status = RequirementStatus.SCAFFOLDING
        assert r.status == RequirementStatus.SCAFFOLDING
        r.status = RequirementStatus.BUILT
        assert r.status == RequirementStatus.BUILT


class TestKnowledgeGraph:
    def test_add_concept(self):
        g = KnowledgeGraph()
        c = Concept(name="Kafka", domain="data", confidence=0.8)
        g.add_concept(c)
        assert g.get_concept("Kafka") == c

    def test_add_edge(self):
        g = KnowledgeGraph()
        c1 = Concept(name="Kafka", domain="data", confidence=0.8)
        c2 = Concept(name="Flink", domain="data", confidence=0.7)
        g.add_concept(c1)
        g.add_concept(c2)
        g.add_edge(c1.name, c2.name, EdgeType.RELATES_TO, weight=0.6)

        edges = g.get_edges(c1.name)
        assert len(edges) == 1
        assert edges[0].target == c2.name
        assert edges[0].edge_type == EdgeType.RELATES_TO

    def test_add_requirement_with_supporting_concept(self):
        g = KnowledgeGraph()
        c = Concept(name="latency", domain="perf", confidence=0.9)
        r = Requirement(description="Sub-100ms", domain="perf", complexity_estimate=0.5)
        g.add_concept(c)
        g.add_requirement(r)
        g.add_edge(c.name, r.id, EdgeType.SUPPORTS)

        edges = g.get_edges(c.name)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SUPPORTS

    def test_get_disconnected_clusters(self):
        g = KnowledgeGraph()
        # Cluster 1
        c1 = Concept(name="A", domain="x", confidence=0.5)
        c2 = Concept(name="B", domain="x", confidence=0.5)
        g.add_concept(c1)
        g.add_concept(c2)
        g.add_edge("A", "B", EdgeType.RELATES_TO)

        # Cluster 2 (disconnected)
        c3 = Concept(name="C", domain="y", confidence=0.5)
        c4 = Concept(name="D", domain="y", confidence=0.5)
        g.add_concept(c3)
        g.add_concept(c4)
        g.add_edge("C", "D", EdgeType.RELATES_TO)

        clusters = g.get_disconnected_clusters()
        assert len(clusters) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/graph/test_model.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement graph model**

`forgestream/graph/model.py`:
```python
"""Knowledge graph model — materialized view over the event log."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class RequirementStatus(str, Enum):
    DETECTED = "detected"
    SCAFFOLDING = "scaffolding"
    BUILT = "built"
    VERIFIED = "verified"


class EdgeType(str, Enum):
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    FULFILLED_BY = "fulfilled_by"
    CONTRADICTS = "contradicts"


@dataclass
class Concept:
    name: str
    domain: str
    confidence: float
    verified: bool = False
    source_events: list[str] = field(default_factory=list)


@dataclass
class Requirement:
    description: str
    domain: str
    complexity_estimate: float
    id: str = field(default_factory=lambda: str(uuid4()))
    status: RequirementStatus = RequirementStatus.DETECTED
    linked_claims: list[str] = field(default_factory=list)


@dataclass
class Artifact:
    path: str
    branch: str
    compiles: bool
    tests_pass: bool
    id: str = field(default_factory=lambda: str(uuid4()))
    linked_requirements: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0


class KnowledgeGraph:
    """In-memory graph rebuilt from event log on startup."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._requirements: dict[str, Requirement] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._edges: dict[str, list[Edge]] = {}  # source -> [edges]

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.name] = concept

    def get_concept(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    @property
    def concepts(self) -> list[Concept]:
        return list(self._concepts.values())

    def add_requirement(self, req: Requirement) -> None:
        self._requirements[req.id] = req

    def get_requirement(self, req_id: str) -> Requirement | None:
        return self._requirements.get(req_id)

    @property
    def requirements(self) -> list[Requirement]:
        return list(self._requirements.values())

    def add_artifact(self, artifact: Artifact) -> None:
        self._artifacts[artifact.id] = artifact

    @property
    def artifacts(self) -> list[Artifact]:
        return list(self._artifacts.values())

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        edge = Edge(source=source, target=target, edge_type=edge_type, weight=weight)
        self._edges.setdefault(source, []).append(edge)

    def get_edges(
        self, source: str, edge_type: EdgeType | None = None
    ) -> list[Edge]:
        edges = self._edges.get(source, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_all_nodes(self) -> set[str]:
        nodes: set[str] = set()
        nodes.update(self._concepts.keys())
        nodes.update(self._requirements.keys())
        nodes.update(self._artifacts.keys())
        return nodes

    def get_neighbors(self, node: str) -> set[str]:
        neighbors: set[str] = set()
        for edge in self._edges.get(node, []):
            neighbors.add(edge.target)
        # Also check reverse edges
        for source, edges in self._edges.items():
            for edge in edges:
                if edge.target == node:
                    neighbors.add(source)
        return neighbors

    def get_disconnected_clusters(self) -> list[set[str]]:
        """Find connected components in the graph (for seed detection)."""
        all_nodes = self.get_all_nodes()
        visited: set[str] = set()
        clusters: list[set[str]] = []

        for node in all_nodes:
            if node in visited:
                continue
            cluster: set[str] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                for neighbor in self.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if cluster:
                clusters.append(cluster)

        return clusters
```

`forgestream/graph/__init__.py`:
```python
"""Knowledge graph — materialized view over the ECEF event log."""
from .model import (
    Artifact,
    Concept,
    Edge,
    EdgeType,
    KnowledgeGraph,
    Requirement,
    RequirementStatus,
)

__all__ = [
    "Artifact",
    "Concept",
    "Edge",
    "EdgeType",
    "KnowledgeGraph",
    "Requirement",
    "RequirementStatus",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/graph/test_model.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/graph/ tests/graph/
git commit -m "feat(SP-2): add KnowledgeGraph model with entities and edges"
```

### Task 2.2: Graph Materializer

**Files:**
- Create: `forgestream/graph/materializer.py`
- Create: `tests/graph/test_materializer.py`

- [ ] **Step 1: Write failing tests for materializer**

`tests/graph/test_materializer.py`:
```python
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.model import KnowledgeGraph, RequirementStatus


class TestGraphMaterializer:
    def test_materialize_claim_creates_concepts(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.CLAIM,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="gemini",
            evaluator=0.5,
            payload={
                "text": "We should use Kafka for ingestion",
                "confidence": 0.85,
                "topic_keywords": ["Kafka", "ingestion"],
            },
        )
        graph = m.materialize([event])
        assert graph.get_concept("Kafka") is not None
        assert graph.get_concept("ingestion") is not None

    def test_materialize_requirement_event(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.6,
            payload={
                "description": "Sub-100ms ingestion",
                "domain": "data-engineering",
                "complexity_estimate": 0.7,
                "linked_claims": [],
            },
        )
        graph = m.materialize([event])
        reqs = graph.requirements
        assert len(reqs) == 1
        assert reqs[0].description == "Sub-100ms ingestion"

    def test_materialize_artifact_event(self):
        m = GraphMaterializer()
        event = Event(
            event_type=EventType.ARTIFACT,
            session_id=uuid4(),
            branch_id=uuid4(),
            author="scaffold-001",
            evaluator=0.7,
            payload={
                "worktree_path": "../forgestream-sc-001",
                "branch_name": "sc/pipeline",
                "files_created": ["src/main.py"],
                "compiles": True,
                "tests_pass": True,
            },
        )
        graph = m.materialize([event])
        arts = graph.artifacts
        assert len(arts) == 1
        assert arts[0].compiles is True

    def test_materialize_contradiction_creates_edge(self):
        m = GraphMaterializer()
        events = [
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "Use strong consistency",
                    "confidence": 0.8,
                    "topic_keywords": ["strong_consistency"],
                },
            ),
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "Use eventual consistency",
                    "confidence": 0.7,
                    "topic_keywords": ["eventual_consistency"],
                },
            ),
            Event(
                event_type=EventType.CONTRADICTION,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="synthesis",
                evaluator=0.6,
                payload={
                    "concept_a": "strong_consistency",
                    "concept_b": "eventual_consistency",
                    "explanation": "Contradictory consistency models",
                },
            ),
        ]
        graph = m.materialize(events)
        from forgestream.graph.model import EdgeType
        edges = graph.get_edges("strong_consistency", EdgeType.CONTRADICTS)
        assert len(edges) == 1
        assert edges[0].target == "eventual_consistency"

    def test_rebuild_from_events_is_idempotent(self):
        m = GraphMaterializer()
        events = [
            Event(
                event_type=EventType.CLAIM,
                session_id=uuid4(),
                branch_id=uuid4(),
                author="gemini",
                evaluator=0.5,
                payload={
                    "text": "test",
                    "confidence": 0.8,
                    "topic_keywords": ["Kafka"],
                },
            ),
        ]
        graph1 = m.materialize(events)
        graph2 = m.materialize(events)
        assert len(graph1.concepts) == len(graph2.concepts)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/graph/test_materializer.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement GraphMaterializer**

`forgestream/graph/materializer.py`:
```python
"""Materializes the knowledge graph from the event log (event sourcing)."""

from __future__ import annotations

from forgestream.events.schema import Event, EventType

from .model import (
    Artifact,
    Concept,
    EdgeType,
    KnowledgeGraph,
    Requirement,
)


class GraphMaterializer:
    """Replays events to build the knowledge graph.

    Stateless: call materialize() with a full event list to get a fresh graph.
    """

    def materialize(self, events: list[Event]) -> KnowledgeGraph:
        """Build a knowledge graph from a list of events."""
        graph = KnowledgeGraph()

        for event in events:
            handler = self._handlers.get(event.event_type)
            if handler:
                handler(self, graph, event)

        return graph

    def _handle_claim(self, graph: KnowledgeGraph, event: Event) -> None:
        keywords = event.payload.get("topic_keywords", [])
        confidence = event.payload.get("confidence", 0.5)

        for keyword in keywords:
            existing = graph.get_concept(keyword)
            if existing is None:
                concept = Concept(
                    name=keyword,
                    domain="",
                    confidence=confidence,
                    source_events=[str(event.id)],
                )
                graph.add_concept(concept)
            else:
                # Update confidence (take max — monotone)
                existing.confidence = max(existing.confidence, confidence)
                existing.source_events.append(str(event.id))

        # Add edges between co-occurring keywords
        for i, kw_a in enumerate(keywords):
            for kw_b in keywords[i + 1 :]:
                graph.add_edge(kw_a, kw_b, EdgeType.RELATES_TO, weight=confidence)

    def _handle_requirement(self, graph: KnowledgeGraph, event: Event) -> None:
        req = Requirement(
            description=event.payload["description"],
            domain=event.payload.get("domain", ""),
            complexity_estimate=event.payload.get("complexity_estimate", 0.5),
            linked_claims=event.payload.get("linked_claims", []),
        )
        graph.add_requirement(req)

    def _handle_artifact(self, graph: KnowledgeGraph, event: Event) -> None:
        artifact = Artifact(
            path=event.payload.get("worktree_path", ""),
            branch=event.payload.get("branch_name", ""),
            compiles=event.payload.get("compiles", False),
            tests_pass=event.payload.get("tests_pass", False),
        )
        graph.add_artifact(artifact)

    def _handle_contradiction(self, graph: KnowledgeGraph, event: Event) -> None:
        concept_a = event.payload.get("concept_a", "")
        concept_b = event.payload.get("concept_b", "")
        if concept_a and concept_b:
            graph.add_edge(concept_a, concept_b, EdgeType.CONTRADICTS)

    _handlers = {
        EventType.CLAIM: _handle_claim,
        EventType.REQUIREMENT: _handle_requirement,
        EventType.ARTIFACT: _handle_artifact,
        EventType.CONTRADICTION: _handle_contradiction,
    }
```

Update `forgestream/graph/__init__.py` to add:
```python
from .materializer import GraphMaterializer
```

And add `"GraphMaterializer"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/graph/test_materializer.py -v
```

Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/graph/ tests/graph/
git commit -m "feat(SP-2): add GraphMaterializer with event sourcing"
```

### Task 2.3: Graph Queries

**Files:**
- Create: `forgestream/graph/query.py`
- Create: `tests/graph/test_query.py`

- [ ] **Step 1: Write failing tests**

`tests/graph/test_query.py`:
```python
from forgestream.graph.model import Concept, EdgeType, KnowledgeGraph
from forgestream.graph.query import GraphQuery


class TestGraphQuery:
    def _build_graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        for name in ["A", "B", "C", "D", "E"]:
            g.add_concept(Concept(name=name, domain="test", confidence=0.5))
        g.add_edge("A", "B", EdgeType.RELATES_TO, weight=0.8)
        g.add_edge("B", "C", EdgeType.RELATES_TO, weight=0.6)
        g.add_edge("D", "E", EdgeType.RELATES_TO, weight=0.7)
        return g

    def test_find_related(self):
        g = self._build_graph()
        q = GraphQuery(g)
        related = q.find_related("A", depth=2)
        assert "B" in related
        assert "C" in related
        assert "D" not in related

    def test_find_isolated_clusters(self):
        g = self._build_graph()
        q = GraphQuery(g)
        clusters = q.find_isolated_clusters(min_size=2)
        assert len(clusters) == 2

    def test_concept_density(self):
        g = self._build_graph()
        q = GraphQuery(g)
        density = q.concept_density()
        # 3 edges among 5 nodes: density = 2*3 / (5*4) = 0.3
        assert 0.0 < density <= 1.0

    def test_verified_ratio(self):
        g = KnowledgeGraph()
        c1 = Concept(name="A", domain="x", confidence=0.9, verified=True)
        c2 = Concept(name="B", domain="x", confidence=0.5)
        g.add_concept(c1)
        g.add_concept(c2)
        q = GraphQuery(g)
        assert q.verified_ratio() == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/graph/test_query.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement GraphQuery**

`forgestream/graph/query.py`:
```python
"""Query utilities for the knowledge graph."""

from __future__ import annotations

from .model import KnowledgeGraph


class GraphQuery:
    """Read-only queries over the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def find_related(self, node: str, depth: int = 1) -> set[str]:
        """BFS to find all nodes within N hops."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(node, 0)]

        while queue:
            current, d = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if d < depth:
                for neighbor in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))

        visited.discard(node)
        return visited

    def find_isolated_clusters(self, min_size: int = 1) -> list[set[str]]:
        """Find disconnected clusters above minimum size (for seed detection)."""
        clusters = self.graph.get_disconnected_clusters()
        return [c for c in clusters if len(c) >= min_size]

    def concept_density(self) -> float:
        """Graph density: 2 * edges / (nodes * (nodes-1))."""
        nodes = self.graph.get_all_nodes()
        n = len(nodes)
        if n < 2:
            return 0.0

        edge_count = sum(
            len(self.graph.get_edges(node)) for node in nodes
        )
        return (2 * edge_count) / (n * (n - 1))

    def verified_ratio(self) -> float:
        """Fraction of concepts that are verified."""
        concepts = self.graph.concepts
        if not concepts:
            return 0.0
        verified = sum(1 for c in concepts if c.verified)
        return verified / len(concepts)
```

Update `forgestream/graph/__init__.py` to add `GraphQuery` import and `__all__` entry.

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/graph/test_query.py -v
pytest -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/graph/ tests/graph/
git commit -m "feat(SP-2): add GraphQuery with BFS, clusters, density"
```

**SP-2 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-9: SOS Governor

Can be built in parallel with SP-3 and SP-5. Depends only on SP-1.

### File Structure

```
forgestream/governor/
├── __init__.py
├── evaluator.py      # Pluggable evaluator function
├── axioms.py         # Axiom checking (monotone, bounded, constraint)
└── trust_region.py   # Trust region dynamics

tests/governor/
├── __init__.py
├── test_evaluator.py
├── test_axioms.py
└── test_trust_region.py
```

### Task 9.1: Evaluator

**Files:**
- Create: `forgestream/governor/__init__.py`
- Create: `forgestream/governor/evaluator.py`
- Create: `tests/governor/__init__.py`
- Create: `tests/governor/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

`tests/governor/test_evaluator.py`:
```python
from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.governor.evaluator import Evaluator, EvaluatorMetrics


class TestEvaluator:
    def _make_events(self, types_and_payloads: list) -> list[Event]:
        sid = uuid4()
        bid = uuid4()
        return [
            Event(
                event_type=t,
                session_id=sid,
                branch_id=bid,
                author="test",
                evaluator=0.0,
                payload=p,
            )
            for t, p in types_and_payloads
        ]

    def test_compute_returns_float_between_0_and_1(self):
        evaluator = Evaluator()
        events = self._make_events([
            (EventType.CLAIM, {"confidence": 0.8, "topic_keywords": ["A"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["x"]}),
        ])
        score = evaluator.compute(events)
        assert 0.0 <= score <= 1.0

    def test_more_verified_findings_increases_score(self):
        evaluator = Evaluator()
        events_few = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
            (EventType.CLAIM, {"topic_keywords": ["B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.8, "sources": ["x"]}),
        ])
        events_many = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
            (EventType.CLAIM, {"topic_keywords": ["B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.8, "sources": ["x"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["y"]}),
        ])
        score_few = evaluator.compute(events_few)
        score_many = evaluator.compute(events_many)
        assert score_many >= score_few

    def test_metrics_breakdown(self):
        evaluator = Evaluator()
        events = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A", "B"]}),
            (EventType.VERIFIED_FINDING, {"confidence": 0.9, "sources": ["x"]}),
            (EventType.ARTIFACT, {"compiles": True, "tests_pass": True}),
        ])
        metrics = evaluator.compute_metrics(events)
        assert isinstance(metrics, EvaluatorMetrics)
        assert metrics.knowledge_density >= 0.0
        assert metrics.verification_rate >= 0.0
        assert metrics.scaffold_success >= 0.0

    def test_custom_weights(self):
        evaluator = Evaluator(weights={"knowledge": 1.0, "verification": 0.0,
                                        "scaffold": 0.0, "uptake": 0.0})
        events = self._make_events([
            (EventType.CLAIM, {"topic_keywords": ["A"]}),
        ])
        score = evaluator.compute(events)
        assert score > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/governor/test_evaluator.py -v
```

- [ ] **Step 3: Implement Evaluator**

`forgestream/governor/evaluator.py`:
```python
"""Pluggable evaluator function for the SOS Governor.

Replace compute() to change what 'improvement' means.
The SOS convergence theorems hold for ANY evaluator
that satisfies the three axioms.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgestream.events.schema import Event, EventType


@dataclass
class EvaluatorMetrics:
    knowledge_density: float
    verification_rate: float
    scaffold_success: float
    suggestion_uptake: float
    composite: float


class Evaluator:
    """Computes E(pi) over a window of events.

    Weights are configurable and will be tuned via GRPO
    in the self-improvement loop (SP-10).
    """

    DEFAULT_WEIGHTS = {
        "knowledge": 0.3,
        "verification": 0.3,
        "scaffold": 0.25,
        "uptake": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def compute(self, events: list[Event]) -> float:
        """Compute the composite evaluator score."""
        metrics = self.compute_metrics(events)
        return metrics.composite

    def compute_metrics(self, events: list[Event]) -> EvaluatorMetrics:
        """Compute individual metric components and the composite."""
        kd = self._knowledge_density(events)
        vr = self._verification_rate(events)
        ss = self._scaffold_success(events)
        su = self._suggestion_uptake(events)

        composite = (
            self.weights["knowledge"] * kd
            + self.weights["verification"] * vr
            + self.weights["scaffold"] * ss
            + self.weights["uptake"] * su
        )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        return EvaluatorMetrics(
            knowledge_density=kd,
            verification_rate=vr,
            scaffold_success=ss,
            suggestion_uptake=su,
            composite=composite,
        )

    @staticmethod
    def _knowledge_density(events: list[Event]) -> float:
        """Unique concepts extracted per claim event."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if not claims:
            return 0.0
        all_keywords: set[str] = set()
        for c in claims:
            all_keywords.update(c.payload.get("topic_keywords", []))
        return min(1.0, len(all_keywords) / max(len(claims), 1))

    @staticmethod
    def _verification_rate(events: list[Event]) -> float:
        """Ratio of verified findings to total research events."""
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        if not claims:
            return 0.0
        return min(1.0, len(findings) / max(len(claims), 1))

    @staticmethod
    def _scaffold_success(events: list[Event]) -> float:
        """Ratio of compiling artifacts to total artifacts."""
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        if not artifacts:
            return 0.0
        compiling = sum(
            1 for a in artifacts if a.payload.get("compiles", False)
        )
        return compiling / len(artifacts)

    @staticmethod
    def _suggestion_uptake(events: list[Event]) -> float:
        """Placeholder: returns 0.5 until we track suggestion dismissals."""
        suggestions = [e for e in events if e.event_type == EventType.SUGGESTION]
        if not suggestions:
            return 0.5
        # Future: track which suggestions were acted on vs dismissed
        return 0.5
```

`forgestream/governor/__init__.py`:
```python
"""SOS Runtime Governor — enforces convergence axioms."""
from .evaluator import Evaluator, EvaluatorMetrics

__all__ = ["Evaluator", "EvaluatorMetrics"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/governor/test_evaluator.py -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/governor/ tests/governor/
git commit -m "feat(SP-9): add pluggable Evaluator with metrics breakdown"
```

### Task 9.2: Axiom Checker

**Files:**
- Create: `forgestream/governor/axioms.py`
- Create: `tests/governor/test_axioms.py`

- [ ] **Step 1: Write failing tests**

`tests/governor/test_axioms.py`:
```python
from forgestream.governor.axioms import AxiomChecker, AxiomResult


class TestAxiomChecker:
    def test_monotone_improvement_holds(self):
        checker = AxiomChecker()
        trajectory = [0.3, 0.35, 0.4, 0.45, 0.5]
        result = checker.check_monotone(trajectory, window_size=3)
        assert result.holds is True

    def test_monotone_improvement_violated(self):
        checker = AxiomChecker()
        # 3 consecutive declining windows
        trajectory = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
        result = checker.check_monotone(trajectory, window_size=2)
        assert result.holds is False

    def test_monotone_allows_individual_dips(self):
        checker = AxiomChecker()
        # Individual dip but overall improving
        trajectory = [0.3, 0.4, 0.35, 0.45, 0.5]
        result = checker.check_monotone(trajectory, window_size=3)
        assert result.holds is True

    def test_bounded_step_holds(self):
        checker = AxiomChecker(epsilon=0.5)
        result = checker.check_bounded_step(
            semantic_drift=0.3, resource_delta=1, scope_delta=5
        )
        assert result.holds is True

    def test_bounded_step_violated_drift(self):
        checker = AxiomChecker(epsilon=0.3)
        result = checker.check_bounded_step(
            semantic_drift=0.8, resource_delta=0, scope_delta=0
        )
        assert result.holds is False
        assert "semantic" in result.reason.lower()

    def test_constraint_preservation_holds(self):
        checker = AxiomChecker()
        result = checker.check_constraint(
            verified_claims_intact=True,
            compilation_preserved=True,
            source_chain_valid=True,
        )
        assert result.holds is True

    def test_constraint_preservation_violated(self):
        checker = AxiomChecker()
        result = checker.check_constraint(
            verified_claims_intact=False,
            compilation_preserved=True,
            source_chain_valid=True,
        )
        assert result.holds is False
        assert "verified" in result.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/governor/test_axioms.py -v
```

- [ ] **Step 3: Implement AxiomChecker**

`forgestream/governor/axioms.py`:
```python
"""SOS axiom checking — runtime invariants for convergence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AxiomResult:
    axiom: str
    holds: bool
    reason: str = ""


class AxiomChecker:
    """Checks the three SOS axioms at runtime.

    Axiom 1 (Monotone Improvement): E(window_n+1) >= E(window_n)
    Axiom 2 (Bounded Step): step size within trust region epsilon
    Axiom 3 (Constraint Preservation): invariants maintained
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        consecutive_violations_threshold: int = 3,
    ) -> None:
        self.epsilon = epsilon
        self.consecutive_violations_threshold = consecutive_violations_threshold

    def check_monotone(
        self, trajectory: list[float], window_size: int = 3
    ) -> AxiomResult:
        """Check Axiom 1: moving average of E is non-decreasing.

        Allows individual dips but flags N consecutive declining windows.
        """
        if len(trajectory) < window_size + 1:
            return AxiomResult(axiom="monotone", holds=True, reason="insufficient data")

        windows = []
        for i in range(len(trajectory) - window_size + 1):
            window_avg = sum(trajectory[i : i + window_size]) / window_size
            windows.append(window_avg)

        consecutive_declines = 0
        for i in range(1, len(windows)):
            if windows[i] < windows[i - 1]:
                consecutive_declines += 1
                if consecutive_declines >= self.consecutive_violations_threshold:
                    return AxiomResult(
                        axiom="monotone",
                        holds=False,
                        reason=(
                            f"{consecutive_declines} consecutive declining windows "
                            f"detected (threshold: {self.consecutive_violations_threshold})"
                        ),
                    )
            else:
                consecutive_declines = 0

        return AxiomResult(axiom="monotone", holds=True)

    def check_bounded_step(
        self,
        semantic_drift: float,
        resource_delta: int,
        scope_delta: int,
    ) -> AxiomResult:
        """Check Axiom 2: all step components within trust region."""
        violations = []

        if semantic_drift > self.epsilon:
            violations.append(
                f"semantic drift {semantic_drift:.2f} > epsilon {self.epsilon:.2f}"
            )
        if resource_delta > self.epsilon * 10:
            violations.append(
                f"resource delta {resource_delta} > bound {self.epsilon * 10:.0f}"
            )
        if scope_delta > self.epsilon * 50:
            violations.append(
                f"scope delta {scope_delta} > bound {self.epsilon * 50:.0f}"
            )

        if violations:
            return AxiomResult(
                axiom="bounded_step",
                holds=False,
                reason="; ".join(violations),
            )
        return AxiomResult(axiom="bounded_step", holds=True)

    def check_constraint(
        self,
        verified_claims_intact: bool,
        compilation_preserved: bool,
        source_chain_valid: bool,
    ) -> AxiomResult:
        """Check Axiom 3: all constraints preserved."""
        violations = []

        if not verified_claims_intact:
            violations.append("verified claims modified or removed")
        if not compilation_preserved:
            violations.append("previously compiling artifact no longer compiles")
        if not source_chain_valid:
            violations.append("verified finding missing source chain")

        if violations:
            return AxiomResult(
                axiom="constraint",
                holds=False,
                reason="; ".join(violations),
            )
        return AxiomResult(axiom="constraint", holds=True)
```

Update `forgestream/governor/__init__.py` to add `AxiomChecker`, `AxiomResult`.

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/governor/test_axioms.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/governor/ tests/governor/
git commit -m "feat(SP-9): add AxiomChecker for runtime SOS invariants"
```

### Task 9.3: Trust Region

**Files:**
- Create: `forgestream/governor/trust_region.py`
- Create: `tests/governor/test_trust_region.py`

- [ ] **Step 1: Write failing tests**

`tests/governor/test_trust_region.py`:
```python
import math

from forgestream.governor.trust_region import TrustRegion


class TestTrustRegion:
    def test_initial_epsilon(self):
        tr = TrustRegion()
        assert tr.epsilon == 0.3  # conservative start

    def test_expand_on_good_meeting(self):
        tr = TrustRegion()
        initial = tr.epsilon
        tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        assert tr.epsilon > initial

    def test_contract_on_axiom_violation(self):
        tr = TrustRegion()
        # First expand a bit
        tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        expanded = tr.epsilon
        # Then violate
        tr.record_axiom_violation()
        assert tr.epsilon < expanded

    def test_epsilon_has_floor(self):
        tr = TrustRegion()
        for _ in range(100):
            tr.record_axiom_violation()
        assert tr.epsilon >= 0.15  # minimum floor

    def test_epsilon_has_ceiling(self):
        tr = TrustRegion()
        for _ in range(100):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)
        assert tr.epsilon <= 0.9  # maximum ceiling

    def test_resource_limits_scale_with_epsilon(self):
        tr = TrustRegion()
        limits_conservative = tr.get_resource_limits()

        for _ in range(10):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        limits_earned = tr.get_resource_limits()
        assert limits_earned["max_concurrent_research"] >= limits_conservative["max_concurrent_research"]
        assert limits_earned["max_concurrent_scaffold"] >= limits_conservative["max_concurrent_scaffold"]

    def test_auto_spawn_only_when_earned(self):
        tr = TrustRegion()
        assert tr.get_resource_limits()["auto_spawn"] is False

        # Earn trust
        for _ in range(8):
            tr.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        assert tr.get_resource_limits()["auto_spawn"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/governor/test_trust_region.py -v
```

- [ ] **Step 3: Implement TrustRegion**

`forgestream/governor/trust_region.py`:
```python
"""Trust region dynamics — the system earns autonomy through convergence."""

from __future__ import annotations

import math


class TrustRegion:
    """Manages the trust region epsilon that governs agent autonomy.

    epsilon starts conservative (0.3) and expands as the system
    demonstrates competence across meetings. Axiom violations contract it.
    """

    EPSILON_BASE = 0.3
    EPSILON_FLOOR = 0.15
    EPSILON_CEILING = 0.9
    AUTO_SPAWN_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._consecutive_improvements: int = 0
        self._total_violations: int = 0
        self._meeting_count: int = 0
        self._volatility: float = 0.0

    @property
    def epsilon(self) -> float:
        """Current trust region value."""
        competence = self._competence_multiplier()
        stability = self._stability_factor()
        raw = self.EPSILON_BASE * competence * stability
        return max(self.EPSILON_FLOOR, min(self.EPSILON_CEILING, raw))

    def record_meeting_result(
        self, e_macro_improved: bool, axiom_violations: int
    ) -> None:
        """Record the outcome of a meeting."""
        self._meeting_count += 1
        self._total_violations += axiom_violations

        if e_macro_improved:
            self._consecutive_improvements += 1
        else:
            self._consecutive_improvements = max(
                0, self._consecutive_improvements - 1
            )

    def record_axiom_violation(self) -> None:
        """Record a single axiom violation (contracts trust region)."""
        self._total_violations += 1
        self._consecutive_improvements = max(
            0, self._consecutive_improvements - 2
        )

    def set_volatility(self, volatility: float) -> None:
        """Update the micro-evaluator volatility."""
        self._volatility = max(0.0, min(1.0, volatility))

    def get_resource_limits(self) -> dict:
        """Resource limits scaled by current epsilon."""
        e = self.epsilon
        scale = e / self.EPSILON_CEILING  # 0 to 1

        return {
            "max_concurrent_research": max(2, int(2 + 3 * scale)),
            "max_concurrent_scaffold": max(2, int(2 + 4 * scale)),
            "spawn_cooldown_seconds": max(10, int(60 - 50 * scale)),
            "scaffold_timeout_minutes": max(10, int(10 + 10 * scale)),
            "auto_spawn": e >= self.AUTO_SPAWN_THRESHOLD,
            "branch_auto_allocate": e >= 0.7,
        }

    def _competence_multiplier(self) -> float:
        """Sigmoid based on improvements vs violations. Range [0.5, 3.0]."""
        alpha = 0.3
        beta = 0.5
        x = alpha * self._consecutive_improvements - beta * self._total_violations
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        return 0.5 + 2.5 * sigmoid

    def _stability_factor(self) -> float:
        """Low volatility = stable = higher factor. Range [0.3, 1.0]."""
        return max(0.3, 1.0 - self._volatility)
```

Update `forgestream/governor/__init__.py` to add `TrustRegion`.

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/governor/test_trust_region.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/governor/ tests/governor/
git commit -m "feat(SP-9): add TrustRegion with earned autonomy dynamics"
```

**SP-9 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-5: Agent Spawner

Manages Claude Code CLI instances via tmux and git worktrees.

### File Structure

```
forgestream/agents/
├── __init__.py
├── spawner.py        # Agent lifecycle management
├── worktrees.py      # Git worktree creation/cleanup
├── registry.py       # Track active agents
└── monitor.py        # tmux pane monitoring

tests/agents/
├── __init__.py
├── test_registry.py
├── test_worktrees.py
└── test_spawner.py
```

### Task 5.1: Agent Registry

**Files:**
- Create: `forgestream/agents/__init__.py`
- Create: `forgestream/agents/registry.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_registry.py`

- [ ] **Step 1: Write failing tests**

`tests/agents/test_registry.py`:
```python
from forgestream.agents.registry import AgentInfo, AgentRegistry, AgentStatus, AgentType


class TestAgentRegistry:
    def test_register_agent(self):
        reg = AgentRegistry()
        agent = reg.register(
            agent_type=AgentType.RESEARCH,
            task_description="Research Kafka Streams",
        )
        assert agent.status == AgentStatus.PROVISIONING
        assert agent.agent_type == AgentType.RESEARCH

    def test_get_active_agents(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "task 1")
        a2 = reg.register(AgentType.SCAFFOLD, "task 2")
        reg.update_status(a2.id, AgentStatus.RUNNING)

        active = reg.get_active()
        assert len(active) == 2  # provisioning + running are active

    def test_get_by_type(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "r1")
        reg.register(AgentType.RESEARCH, "r2")
        reg.register(AgentType.SCAFFOLD, "s1")

        research = reg.get_by_type(AgentType.RESEARCH)
        assert len(research) == 2

    def test_update_status(self):
        reg = AgentRegistry()
        agent = reg.register(AgentType.SCAFFOLD, "test")
        reg.update_status(agent.id, AgentStatus.RUNNING)
        assert reg.get(agent.id).status == AgentStatus.RUNNING

    def test_count_by_type(self):
        reg = AgentRegistry()
        reg.register(AgentType.RESEARCH, "r1")
        reg.register(AgentType.RESEARCH, "r2")
        reg.register(AgentType.SCAFFOLD, "s1")

        counts = reg.count_active_by_type()
        assert counts[AgentType.RESEARCH] == 2
        assert counts[AgentType.SCAFFOLD] == 1

    def test_completed_agents_not_active(self):
        reg = AgentRegistry()
        agent = reg.register(AgentType.RESEARCH, "done")
        reg.update_status(agent.id, AgentStatus.COMPLETED)

        active = reg.get_active()
        assert len(active) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_registry.py -v
```

- [ ] **Step 3: Implement AgentRegistry**

`forgestream/agents/registry.py`:
```python
"""Agent registry — tracks active Claude Code instances."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class AgentType(str, Enum):
    RESEARCH = "research"
    SCAFFOLD = "scaffold"


class AgentStatus(str, Enum):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentInfo:
    id: str
    agent_type: AgentType
    task_description: str
    status: AgentStatus = AgentStatus.PROVISIONING
    tmux_session: str = ""
    worktree_path: str = ""
    branch_name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class AgentRegistry:
    """In-memory registry of all agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    def register(
        self,
        agent_type: AgentType,
        task_description: str,
    ) -> AgentInfo:
        agent_id = f"{agent_type.value[:1]}-{str(uuid4())[:8]}"
        agent = AgentInfo(
            id=agent_id,
            agent_type=agent_type,
            task_description=task_description,
        )
        self._agents[agent_id] = agent
        return agent

    def get(self, agent_id: str) -> AgentInfo | None:
        return self._agents.get(agent_id)

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                agent.completed_at = datetime.now(timezone.utc)

    def get_active(self) -> list[AgentInfo]:
        return [
            a
            for a in self._agents.values()
            if a.status in (AgentStatus.PROVISIONING, AgentStatus.RUNNING)
        ]

    def get_by_type(self, agent_type: AgentType) -> list[AgentInfo]:
        return [
            a
            for a in self._agents.values()
            if a.agent_type == agent_type
            and a.status in (AgentStatus.PROVISIONING, AgentStatus.RUNNING)
        ]

    def count_active_by_type(self) -> Counter[AgentType]:
        active = self.get_active()
        return Counter(a.agent_type for a in active)
```

`forgestream/agents/__init__.py`:
```python
"""Agent orchestration — Claude Code CLI lifecycle management."""
from .registry import AgentInfo, AgentRegistry, AgentStatus, AgentType

__all__ = ["AgentInfo", "AgentRegistry", "AgentStatus", "AgentType"]
```

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/agents/test_registry.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/agents/ tests/agents/
git commit -m "feat(SP-5): add AgentRegistry for tracking Claude Code instances"
```

### Task 5.2: Worktree Manager

**Files:**
- Create: `forgestream/agents/worktrees.py`
- Create: `tests/agents/test_worktrees.py`

- [ ] **Step 1: Write failing tests**

`tests/agents/test_worktrees.py`:
```python
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from forgestream.agents.worktrees import WorktreeManager


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    # Create an initial commit
    (repo / "README.md").write_text("test")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
    )
    yield repo


class TestWorktreeManager:
    def test_create_worktree(self, temp_repo):
        mgr = WorktreeManager(repo_path=str(temp_repo))
        wt_path = mgr.create(slug="test-feature")
        assert Path(wt_path).exists()
        assert "test-feature" in wt_path

    def test_list_worktrees(self, temp_repo):
        mgr = WorktreeManager(repo_path=str(temp_repo))
        mgr.create(slug="feat-a")
        mgr.create(slug="feat-b")
        worktrees = mgr.list_worktrees()
        slugs = [wt["slug"] for wt in worktrees]
        assert "feat-a" in slugs
        assert "feat-b" in slugs

    def test_remove_worktree(self, temp_repo):
        mgr = WorktreeManager(repo_path=str(temp_repo))
        wt_path = mgr.create(slug="to-remove")
        assert Path(wt_path).exists()

        mgr.remove(slug="to-remove")
        assert not Path(wt_path).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_worktrees.py -v
```

- [ ] **Step 3: Implement WorktreeManager**

`forgestream/agents/worktrees.py`:
```python
"""Git worktree lifecycle management for scaffold agents."""

from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    """Creates and manages git worktrees for scaffold agent isolation."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path)
        self._parent = self.repo_path.parent

    def create(self, slug: str) -> str:
        """Create a new worktree with a dedicated branch.

        Returns the absolute path to the worktree directory.
        """
        wt_path = self._parent / f"{self.repo_path.name}-sc-{slug}"
        branch_name = f"sc/{slug}"

        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(wt_path)],
            cwd=self.repo_path,
            capture_output=True,
            check=True,
        )
        return str(wt_path)

    def remove(self, slug: str) -> None:
        """Remove a worktree and prune."""
        wt_path = self._parent / f"{self.repo_path.name}-sc-{slug}"
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=self.repo_path,
            capture_output=True,
        )
        # Also delete the branch
        branch_name = f"sc/{slug}"
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=self.repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repo_path,
            capture_output=True,
        )

    def list_worktrees(self) -> list[dict]:
        """List active scaffold worktrees."""
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        worktrees = []
        current: dict = {}
        prefix = f"{self.repo_path.name}-sc-"

        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current and current.get("slug"):
                    worktrees.append(current)
                path = line.split(" ", 1)[1]
                name = Path(path).name
                current = {
                    "path": path,
                    "slug": name.removeprefix(prefix) if prefix in name else "",
                }
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1]

        if current and current.get("slug"):
            worktrees.append(current)

        return worktrees
```

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/agents/test_worktrees.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/agents/worktrees.py tests/agents/test_worktrees.py
git commit -m "feat(SP-5): add WorktreeManager for scaffold agent isolation"
```

### Task 5.3: Agent Monitor + Spawner

**Files:**
- Create: `forgestream/agents/monitor.py`
- Create: `forgestream/agents/spawner.py`
- Create: `tests/agents/test_spawner.py`

- [ ] **Step 1: Write failing tests for Spawner**

`tests/agents/test_spawner.py`:
```python
from forgestream.agents.registry import AgentRegistry, AgentType
from forgestream.agents.spawner import SpawnDecision, SpawnPolicy
from forgestream.governor.trust_region import TrustRegion


class TestSpawnPolicy:
    def test_can_spawn_when_under_limits(self):
        registry = AgentRegistry()
        trust = TrustRegion()
        policy = SpawnPolicy(registry=registry, trust_region=trust)

        decision = policy.can_spawn(AgentType.RESEARCH)
        assert decision.allowed is True

    def test_cannot_spawn_over_limit(self):
        registry = AgentRegistry()
        trust = TrustRegion()
        policy = SpawnPolicy(registry=registry, trust_region=trust)

        # Fill up to limit (initial limit is 2)
        registry.register(AgentType.RESEARCH, "r1")
        registry.register(AgentType.RESEARCH, "r2")

        decision = policy.can_spawn(AgentType.RESEARCH)
        assert decision.allowed is False
        assert "limit" in decision.reason.lower()

    def test_spawn_limit_increases_with_trust(self):
        registry = AgentRegistry()
        trust = TrustRegion()

        # Earn trust
        for _ in range(10):
            trust.record_meeting_result(e_macro_improved=True, axiom_violations=0)

        policy = SpawnPolicy(registry=registry, trust_region=trust)
        limits = trust.get_resource_limits()

        # Should allow more than initial 2
        assert limits["max_concurrent_research"] > 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_spawner.py -v
```

- [ ] **Step 3: Implement Monitor and Spawner**

`forgestream/agents/monitor.py`:
```python
"""Monitor running Claude Code agents via tmux."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class AgentOutput:
    last_lines: list[str]
    is_running: bool


class AgentMonitor:
    """Monitors agent progress by reading tmux pane output."""

    @staticmethod
    def capture(tmux_session: str, lines: int = 5) -> AgentOutput:
        """Capture the last N lines from an agent's tmux session."""
        result = subprocess.run(
            [
                "tmux",
                "capture-pane",
                "-t",
                tmux_session,
                "-p",
                "-l",
                str(lines),
            ],
            capture_output=True,
            text=True,
        )

        output_lines = result.stdout.strip().splitlines() if result.stdout else []

        # Check if tmux session still exists
        check = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            capture_output=True,
        )
        is_running = check.returncode == 0

        return AgentOutput(last_lines=output_lines, is_running=is_running)

    @staticmethod
    def kill(tmux_session: str) -> None:
        """Kill an agent's tmux session."""
        subprocess.run(
            ["tmux", "kill-session", "-t", tmux_session],
            capture_output=True,
        )
```

`forgestream/agents/spawner.py`:
```python
"""Agent spawn policy and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .registry import AgentRegistry, AgentType
from forgestream.governor.trust_region import TrustRegion


@dataclass
class SpawnDecision:
    allowed: bool
    reason: str = ""


class SpawnPolicy:
    """Determines whether a new agent can be spawned based on trust region limits."""

    def __init__(
        self,
        registry: AgentRegistry,
        trust_region: TrustRegion,
    ) -> None:
        self.registry = registry
        self.trust_region = trust_region
        self._last_spawn: datetime | None = None

    def can_spawn(self, agent_type: AgentType) -> SpawnDecision:
        """Check if spawning a new agent is allowed."""
        limits = self.trust_region.get_resource_limits()

        # Check concurrent agent limit
        counts = self.registry.count_active_by_type()
        limit_key = (
            "max_concurrent_research"
            if agent_type == AgentType.RESEARCH
            else "max_concurrent_scaffold"
        )
        current_count = counts.get(agent_type, 0)
        max_count = limits[limit_key]

        if current_count >= max_count:
            return SpawnDecision(
                allowed=False,
                reason=f"{agent_type.value} agent limit reached ({current_count}/{max_count})",
            )

        # Check cooldown
        cooldown = limits["spawn_cooldown_seconds"]
        if self._last_spawn is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_spawn).total_seconds()
            if elapsed < cooldown:
                return SpawnDecision(
                    allowed=False,
                    reason=f"cooldown: {cooldown - elapsed:.0f}s remaining",
                )

        return SpawnDecision(allowed=True)

    def record_spawn(self) -> None:
        """Record that an agent was spawned (for cooldown tracking)."""
        self._last_spawn = datetime.now(timezone.utc)
```

Update `forgestream/agents/__init__.py` to add new exports.

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/agents/test_spawner.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/agents/ tests/agents/
git commit -m "feat(SP-5): add AgentMonitor, SpawnPolicy with trust region limits"
```

**SP-5 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-3: Gemini Live API Integration

### File Structure

```
forgestream/gemini/
├── __init__.py
├── client.py         # WebSocket connection manager
├── audio.py          # Audio capture + streaming
├── extraction.py     # Claim parsing from Gemini output
└── context.py        # Context injection (10-min summaries)

tests/gemini/
├── __init__.py
├── test_extraction.py
└── test_context.py
```

### Task 3.1: Claim Extraction

The Gemini WebSocket client itself requires API credentials and can't be unit-tested without mocking. Focus tests on the extraction/parsing layer.

**Files:**
- Create: `forgestream/gemini/__init__.py`
- Create: `forgestream/gemini/extraction.py`
- Create: `tests/gemini/__init__.py`
- Create: `tests/gemini/test_extraction.py`

- [ ] **Step 1: Write failing tests for claim extraction**

`tests/gemini/test_extraction.py`:
```python
from uuid import uuid4

from forgestream.events.schema import EventType
from forgestream.gemini.extraction import ClaimExtractor


class TestClaimExtractor:
    def test_parse_structured_claim(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        gemini_output = {
            "text": "We need sub-100ms latency for ingestion",
            "speaker": "Expert A",
            "confidence": 0.85,
            "tone_markers": ["emphasis"],
            "topic_keywords": ["latency", "ingestion"],
            "is_requirement": True,
            "is_question": False,
        }
        event = extractor.parse_claim(gemini_output)
        assert event.event_type == EventType.CLAIM
        assert event.payload["confidence"] == 0.85
        assert "latency" in event.payload["topic_keywords"]
        assert event.author == "gemini"

    def test_parse_adjusts_confidence_for_hesitation(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        gemini_output = {
            "text": "Maybe we should use Kafka",
            "speaker": "Expert A",
            "confidence": 0.8,
            "tone_markers": ["hesitation"],
            "topic_keywords": ["Kafka"],
            "is_requirement": False,
            "is_question": False,
        }
        event = extractor.parse_claim(gemini_output)
        # Hesitation reduces confidence
        assert event.payload["confidence"] < 0.8

    def test_parse_boosts_priority_for_emphasis(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        output = {
            "text": "This is CRITICAL",
            "speaker": "Expert",
            "confidence": 0.9,
            "tone_markers": ["emphasis", "excitement"],
            "topic_keywords": ["critical_requirement"],
            "is_requirement": True,
            "is_question": False,
        }
        event = extractor.parse_claim(output)
        assert event.payload.get("priority_boost", 0) > 0

    def test_parse_handles_missing_fields(self):
        extractor = ClaimExtractor(session_id=uuid4(), branch_id=uuid4())
        # Minimal output — only text
        gemini_output = {"text": "Something interesting"}
        event = extractor.parse_claim(gemini_output)
        assert event.event_type == EventType.CLAIM
        assert event.payload["text"] == "Something interesting"
        assert event.payload["confidence"] == 0.5  # default
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/gemini/test_extraction.py -v
```

- [ ] **Step 3: Implement ClaimExtractor**

`forgestream/gemini/extraction.py`:
```python
"""Parse Gemini Live API output into ECEF claim events."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from forgestream.events.schema import Event, EventType


HESITATION_PENALTY = 0.15
BACKTRACK_PENALTY = 0.2
EMPHASIS_BOOST = 0.2
EXCITEMENT_BOOST = 0.15


class ClaimExtractor:
    """Transforms structured Gemini output into Event objects."""

    def __init__(self, session_id: UUID, branch_id: UUID) -> None:
        self.session_id = session_id
        self.branch_id = branch_id

    def parse_claim(self, gemini_output: dict[str, Any]) -> Event:
        """Parse a single Gemini output into a claim event."""
        text = gemini_output.get("text", "")
        confidence = gemini_output.get("confidence", 0.5)
        tone_markers = gemini_output.get("tone_markers", [])
        topic_keywords = gemini_output.get("topic_keywords", [])

        # Adjust confidence based on tone markers
        priority_boost = 0.0

        if "hesitation" in tone_markers:
            confidence -= HESITATION_PENALTY
        if "backtracking" in tone_markers:
            confidence -= BACKTRACK_PENALTY
        if "emphasis" in tone_markers:
            priority_boost += EMPHASIS_BOOST
        if "excitement" in tone_markers:
            priority_boost += EXCITEMENT_BOOST

        confidence = max(0.0, min(1.0, confidence))

        payload: dict[str, Any] = {
            "text": text,
            "speaker": gemini_output.get("speaker", "unknown"),
            "confidence": confidence,
            "tone_markers": tone_markers,
            "topic_keywords": topic_keywords,
            "is_requirement": gemini_output.get("is_requirement", False),
            "is_question": gemini_output.get("is_question", False),
        }

        if gemini_output.get("audio_timestamp"):
            payload["audio_timestamp"] = gemini_output["audio_timestamp"]

        if priority_boost > 0:
            payload["priority_boost"] = priority_boost

        return Event(
            event_type=EventType.CLAIM,
            session_id=self.session_id,
            branch_id=self.branch_id,
            author="gemini",
            evaluator=0.0,  # filled by governor before write
            payload=payload,
        )
```

`forgestream/gemini/__init__.py`:
```python
"""Gemini Live API integration — meeting audio/video processing."""
from .extraction import ClaimExtractor

__all__ = ["ClaimExtractor"]
```

- [ ] **Step 4: Run tests and full suite**

```bash
pytest tests/gemini/test_extraction.py -v
pytest -v
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/gemini/ tests/gemini/
git commit -m "feat(SP-3): add ClaimExtractor with tone-adjusted confidence"
```

### Task 3.2: Context Injection

**Files:**
- Create: `forgestream/gemini/context.py`
- Create: `tests/gemini/test_context.py`

- [ ] **Step 1: Write failing tests**

`tests/gemini/test_context.py`:
```python
from forgestream.graph.model import Concept, KnowledgeGraph, Requirement
from forgestream.gemini.context import ContextBuilder


class TestContextBuilder:
    def test_build_summary(self):
        g = KnowledgeGraph()
        g.add_concept(Concept(name="Kafka", domain="data", confidence=0.9, verified=True))
        g.add_concept(Concept(name="latency", domain="perf", confidence=0.8))
        g.add_requirement(Requirement(
            description="Sub-100ms ingestion", domain="data",
            complexity_estimate=0.7,
        ))

        builder = ContextBuilder()
        summary = builder.build_injection(g, active_branches=["main", "burst-handling"])

        assert "Kafka" in summary
        assert "Sub-100ms" in summary
        assert "burst-handling" in summary
        assert isinstance(summary, str)
        assert len(summary) < 2000  # keep context injections small

    def test_empty_graph(self):
        g = KnowledgeGraph()
        builder = ContextBuilder()
        summary = builder.build_injection(g, active_branches=[])
        assert isinstance(summary, str)
```

- [ ] **Step 2: Run tests, implement, verify, commit**

`forgestream/gemini/context.py`:
```python
"""Build context injection summaries for Gemini Live API."""

from __future__ import annotations

from forgestream.graph.model import KnowledgeGraph


class ContextBuilder:
    """Generates concise summaries of the knowledge graph for Gemini context injection."""

    def build_injection(
        self,
        graph: KnowledgeGraph,
        active_branches: list[str],
    ) -> str:
        """Build a context injection string for Gemini.

        Injected every ~10 minutes to fight context decay.
        """
        parts = ["Current knowledge state:"]

        # Concepts
        concepts = graph.concepts
        verified = [c for c in concepts if c.verified]
        parts.append(
            f"- {len(verified)} verified concepts, "
            f"{len(concepts) - len(verified)} unverified"
        )

        if verified:
            names = ", ".join(c.name for c in verified[:10])
            parts.append(f"- Verified: {names}")

        # Requirements
        reqs = graph.requirements
        if reqs:
            parts.append(f"- {len(reqs)} requirements detected:")
            for r in reqs[:5]:
                parts.append(f"  - {r.description} (status: {r.status.value})")

        # Artifacts
        arts = graph.artifacts
        if arts:
            compiling = sum(1 for a in arts if a.compiles)
            parts.append(f"- {len(arts)} scaffolds ({compiling} compiling)")

        # Branches
        if active_branches:
            parts.append(f"- Active branches: {', '.join(active_branches)}")

        parts.append(
            "\nContinue extracting claims. "
            "Flag anything that relates to or contradicts the above."
        )

        return "\n".join(parts)
```

Update `forgestream/gemini/__init__.py` to add `ContextBuilder`.

```bash
pytest tests/gemini/ -v
pytest -v
cd ~/projects/forgestream
git add forgestream/gemini/ tests/gemini/
git commit -m "feat(SP-3): add ContextBuilder for 10-min Gemini context injection"
```

### Task 3.3: Gemini WebSocket Client

**Files:**
- Create: `forgestream/gemini/client.py`

This is infrastructure code that wraps the google-genai SDK. It requires API credentials and cannot be meaningfully unit-tested. Implementation follows the Gemini Live API reference.

- [ ] **Step 1: Implement client stub with clear interface**

`forgestream/gemini/client.py`:
```python
"""Gemini Live API WebSocket client."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Coroutine
from uuid import UUID

from .extraction import ClaimExtractor

# System instructions per mode
MODE_INSTRUCTIONS = {
    "extract": (
        "You are an ECEF knowledge extractor in a meeting. "
        "For each substantive claim, emit a JSON object with: "
        "text, speaker, confidence (0-1), tone_markers (list), "
        "topic_keywords (list), is_requirement (bool), is_question (bool). "
        "Focus on what the expert needs built. Extract requirements, "
        "constraints, tech preferences. Do NOT summarize. "
        "Extract EVERY claim. Emit in real-time."
    ),
    "collaborative": (
        "You are an ECEF knowledge extractor in a design discussion. "
        "For each substantive claim, emit a JSON object with: "
        "text, speaker, confidence (0-1), tone_markers (list), "
        "topic_keywords (list), is_requirement (bool), is_question (bool). "
        "Focus on architectural decisions, trade-offs, agreements "
        "and disagreements. Emit in real-time."
    ),
    "knowledge": (
        "You are an ECEF knowledge extractor doing expertise capture. "
        "For each substantive claim, emit a JSON object with: "
        "text, speaker, confidence (0-1), tone_markers (list), "
        "topic_keywords (list), is_requirement (bool), is_question (bool). "
        "Focus on domain knowledge, mental models, heuristics, "
        "and tacit knowledge. Emit in real-time."
    ),
}


class GeminiLiveClient:
    """Connects to Gemini Live API and streams audio for claim extraction.

    Requires GOOGLE_API_KEY environment variable.
    Uses google-genai SDK for WebSocket connection.
    """

    def __init__(
        self,
        session_id: UUID,
        branch_id: UUID,
        mode: str = "extract",
        on_claim: Callable[[dict[str, Any]], Coroutine] | None = None,
    ) -> None:
        self.session_id = session_id
        self.branch_id = branch_id
        self.mode = mode
        self.on_claim = on_claim
        self.extractor = ClaimExtractor(session_id, branch_id)
        self._session = None

    async def connect(self) -> None:
        """Establish WebSocket connection to Gemini Live API."""
        try:
            from google import genai

            client = genai.Client()
            config = {
                "response_modalities": ["TEXT"],
                "system_instruction": MODE_INSTRUCTIONS.get(
                    self.mode, MODE_INSTRUCTIONS["extract"]
                ),
            }
            self._session = await client.aio.live.connect(
                model="gemini-2.5-flash", config=config
            )
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. "
                "Install with: pip install 'forgestream[gemini]'"
            )

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send a chunk of PCM 16kHz audio to Gemini."""
        if self._session:
            await self._session.send({"data": audio_chunk, "mime_type": "audio/pcm"})

    async def send_context(self, text: str) -> None:
        """Inject context text into the Gemini session."""
        if self._session:
            await self._session.send({"text": text})

    async def receive_claims(self) -> AsyncIterator[dict[str, Any]]:
        """Receive and yield parsed claims from Gemini."""
        if not self._session:
            return

        async for response in self._session.receive():
            if hasattr(response, "text") and response.text:
                try:
                    claim_data = json.loads(response.text)
                    if self.on_claim:
                        await self.on_claim(claim_data)
                    yield claim_data
                except json.JSONDecodeError:
                    # Gemini may emit non-JSON text; skip it
                    continue

    def set_mode(self, mode: str) -> None:
        """Update the extraction mode (requires reconnect)."""
        self.mode = mode

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._session:
            await self._session.close()
            self._session = None
```

- [ ] **Step 2: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/gemini/client.py
git commit -m "feat(SP-3): add GeminiLiveClient WebSocket wrapper"
```

**SP-3 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-4: Synthesis Engine

The brain. Depends on SP-1 (events) and SP-2 (knowledge graph).

### File Structure

```
forgestream/synthesis/
├── __init__.py
├── engine.py           # Main event processing loop
├── requirements.py     # Requirement detection from claims
├── contradictions.py   # Contradiction detection
├── branches.py         # Branch management + calculus
├── seeds.py            # Seed detection (disconnected clusters)
└── suggestions.py      # Priority queue

tests/synthesis/
├── __init__.py
├── test_requirements.py
├── test_contradictions.py
├── test_branches.py
├── test_seeds.py
└── test_suggestions.py
```

### Task 4.1: Suggestion Queue

**Files:**
- Create: `forgestream/synthesis/__init__.py`
- Create: `forgestream/synthesis/suggestions.py`
- Create: `tests/synthesis/__init__.py`
- Create: `tests/synthesis/test_suggestions.py`

- [ ] **Step 1: Write failing tests**

`tests/synthesis/test_suggestions.py`:
```python
from forgestream.synthesis.suggestions import Priority, Suggestion, SuggestionQueue


class TestSuggestionQueue:
    def test_add_and_get_top(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="low priority", priority_score=0.2))
        q.add(Suggestion(text="high priority", priority_score=0.9))
        q.add(Suggestion(text="medium", priority_score=0.5))

        top = q.peek()
        assert top.text == "high priority"

    def test_priority_categories(self):
        assert Priority.from_score(0.95) == Priority.CRITICAL
        assert Priority.from_score(0.75) == Priority.STRATEGIC
        assert Priority.from_score(0.55) == Priority.DELVE_DEEPER
        assert Priority.from_score(0.35) == Priority.GOOD_TO_PROBE
        assert Priority.from_score(0.15) == Priority.NICE_TO_KNOW

    def test_dismiss_removes_top(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="top", priority_score=0.9))
        q.add(Suggestion(text="second", priority_score=0.5))

        q.dismiss()
        assert q.peek().text == "second"

    def test_decay_reduces_scores(self):
        q = SuggestionQueue()
        s = Suggestion(text="decaying", priority_score=0.8, decay_rate=0.1)
        q.add(s)
        q.apply_decay(steps=3)
        assert q.peek().priority_score < 0.8

    def test_get_all_by_priority(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="a", priority_score=0.95))
        q.add(Suggestion(text="b", priority_score=0.75))
        q.add(Suggestion(text="c", priority_score=0.15))

        critical = q.get_by_priority(Priority.CRITICAL)
        assert len(critical) == 1
```

- [ ] **Step 2: Implement, test, commit** (following same TDD pattern)

`forgestream/synthesis/suggestions.py`:
```python
"""Priority-scored suggestion queue."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class Priority(str, Enum):
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    DELVE_DEEPER = "delve_deeper"
    GOOD_TO_PROBE = "good_to_probe"
    NICE_TO_KNOW = "nice_to_know"

    @classmethod
    def from_score(cls, score: float) -> Priority:
        if score >= 0.9:
            return cls.CRITICAL
        if score >= 0.7:
            return cls.STRATEGIC
        if score >= 0.5:
            return cls.DELVE_DEEPER
        if score >= 0.3:
            return cls.GOOD_TO_PROBE
        return cls.NICE_TO_KNOW


@dataclass(order=True)
class Suggestion:
    priority_score: float = field(compare=True)
    text: str = field(compare=False)
    id: str = field(default_factory=lambda: str(uuid4()), compare=False)
    category: Priority = field(default=Priority.NICE_TO_KNOW, compare=False)
    linked_events: list[str] = field(default_factory=list, compare=False)
    decay_rate: float = field(default=0.02, compare=False)

    def __post_init__(self):
        self.category = Priority.from_score(self.priority_score)
        # Negate for max-heap behavior with heapq (min-heap)
        self.priority_score = self.priority_score


class SuggestionQueue:
    """Max-heap priority queue for meeting suggestions."""

    def __init__(self) -> None:
        self._items: list[Suggestion] = []

    def add(self, suggestion: Suggestion) -> None:
        suggestion.category = Priority.from_score(suggestion.priority_score)
        # Use negative score for max-heap
        heapq.heappush(self._items, suggestion)
        # Re-sort as max-heap
        self._items.sort(key=lambda s: s.priority_score, reverse=True)

    def peek(self) -> Suggestion | None:
        return self._items[0] if self._items else None

    def dismiss(self) -> Suggestion | None:
        if self._items:
            return self._items.pop(0)
        return None

    def apply_decay(self, steps: int = 1) -> None:
        for s in self._items:
            s.priority_score = max(0.0, s.priority_score - s.decay_rate * steps)
            s.category = Priority.from_score(s.priority_score)
        self._items.sort(key=lambda s: s.priority_score, reverse=True)

    def get_by_priority(self, priority: Priority) -> list[Suggestion]:
        return [s for s in self._items if s.category == priority]

    def get_all(self) -> list[Suggestion]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
```

`forgestream/synthesis/__init__.py`:
```python
"""Synthesis engine — the brain of ForgeStream."""
from .suggestions import Priority, Suggestion, SuggestionQueue

__all__ = ["Priority", "Suggestion", "SuggestionQueue"]
```

```bash
pytest tests/synthesis/test_suggestions.py -v
pytest -v
cd ~/projects/forgestream
git add forgestream/synthesis/ tests/synthesis/
git commit -m "feat(SP-4): add SuggestionQueue with priority levels and decay"
```

### Task 4.2-4.5: Requirements, Contradictions, Branches, Seeds

For each remaining synthesis module, follow the same TDD pattern: write failing tests, implement, verify, commit. The implementations use the KnowledgeGraph and EventStore from SP-1 and SP-2.

- [ ] **Step 1: Implement requirements detection** (`forgestream/synthesis/requirements.py`)
  - Scan claim events for requirement signals ("we need", "it should", "must have")
  - Emit requirement events with linked claims
  - Test: given claims with requirement language, verify requirement events are produced

- [ ] **Step 2: Implement contradiction detection** (`forgestream/synthesis/contradictions.py`)
  - Compare new claims against verified concepts in the knowledge graph
  - Use keyword overlap + semantic negation patterns
  - Emit contradiction events linking conflicting claims
  - Test: given two contradictory claims, verify contradiction event is produced

- [ ] **Step 3: Implement branch management** (`forgestream/synthesis/branches.py`)
  - Track conversation topic drift using keyword centroid comparison
  - Emit branch_point events when drift exceeds threshold
  - Calculate branch_potential, branch_momentum, branch_roi
  - Test: given a sequence of claims that drift from the main topic, verify branch_point emitted

- [ ] **Step 4: Implement seed detection** (`forgestream/synthesis/seeds.py`)
  - Use GraphQuery.find_isolated_clusters() to detect disconnected concept groups
  - Emit seed events for clusters with novelty > threshold
  - Test: given disconnected concept clusters, verify seed events emitted

- [ ] **Step 5: Implement main engine loop** (`forgestream/synthesis/engine.py`)
  - Subscribe to claim events via EventSubscriber
  - For each claim: update graph, check requirements, check contradictions, check branches, check seeds
  - Emit derived events (requirement, contradiction, suggestion, branch_point, seed)
  - Test: integration test with mock event stream

- [ ] **Step 6: Run full suite and commit**

```bash
pytest -v
cd ~/projects/forgestream
git add forgestream/synthesis/ tests/synthesis/
git commit -m "feat(SP-4): add Synthesis Engine with requirements, contradictions, branches, seeds"
```

**SP-4 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-6: Agent Templates

Prompt templates for research and scaffold agents. Depends on SP-5.

- [ ] **Step 1: Implement research template** (`forgestream/agents/templates/research.py`)
  - Build prompt from: research question, relevant claims, knowledge graph context
  - Define expected output JSON schema
  - Parse output into verified_finding event

- [ ] **Step 2: Implement scaffold template** (`forgestream/agents/templates/scaffold.py`)
  - Build prompt from: requirement description, verified findings, domain context
  - Define expected output JSON schema (files_created, compiles, tests_pass, open_questions)
  - Parse output into artifact + suggestion events

- [ ] **Step 3: Tests, full suite, commit**

```bash
pytest -v
cd ~/projects/forgestream
git add forgestream/agents/templates/ tests/agents/
git commit -m "feat(SP-6): add research and scaffold prompt templates"
```

**SP-6 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-7: Terminal TUI

Built with Python `textual`. Depends on SP-1 and SP-4.

- [ ] **Step 1: Create TUI app shell** (`forgestream/tui/app.py`)
  - textual.App subclass with header, footer, CSS layout
  - 5 panels: feed, suggestions, branches, agents, hotkeys

- [ ] **Step 2: Implement Feed panel** (`forgestream/tui/panels/feed.py`)
  - Scrolling log of claim events with linkage annotations
  - Subscribe to claim events via EventSubscriber

- [ ] **Step 3: Implement Suggestions panel** (`forgestream/tui/panels/suggestions.py`)
  - Color-coded priority queue display (red/amber/blue/green/grey)
  - Keyboard dismiss with `s` key

- [ ] **Step 4: Implement Branches panel** (`forgestream/tui/panels/branches.py`)
  - Tree display with potential/momentum/ROI metrics
  - Branch focus with `b` + arrow keys

- [ ] **Step 5: Implement Agents panel** (`forgestream/tui/panels/agents.py`)
  - Progress bars for running agents, status indicators
  - Detail view with `a` + number

- [ ] **Step 6: Implement Quiet mode** (`forgestream/tui/quiet.py`)
  - Collapse to single status bar
  - Only critical suggestions flash

- [ ] **Step 7: Wire up hotkeys** (m, b, s, a, p, r, q, Space, Esc, /commands)

- [ ] **Step 8: Manual testing, commit**

```bash
cd ~/projects/forgestream
git add forgestream/tui/ tests/tui/
git commit -m "feat(SP-7): add Terminal TUI with all panels and hotkeys"
```

**SP-7 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-8: Web Dashboard

FastAPI + D3.js. Depends on SP-1 and SP-2.

- [ ] **Step 1: FastAPI server** (`forgestream/dashboard/server.py`)
- [ ] **Step 2: REST API endpoints** (`forgestream/dashboard/api.py`) — graph, evaluator, timeline, artifacts
- [ ] **Step 3: WebSocket live updates** (`forgestream/dashboard/ws.py`)
- [ ] **Step 4: Frontend** (`forgestream/dashboard/static/`) — D3 force-directed graph, evaluator chart, timeline
- [ ] **Step 5: Tests and commit**

```bash
cd ~/projects/forgestream
git add forgestream/dashboard/ tests/dashboard/
git commit -m "feat(SP-8): add Web Dashboard with knowledge graph viz"
```

**SP-8 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-10: Self-Improvement

Depends on SP-9.

- [ ] **Step 1: GRPO weight tuning** (`forgestream/governor/improvement.py`)
  - Generate N perturbed weight vectors
  - Retroactively evaluate against meeting event log
  - Update weights toward best-performing perturbation

- [ ] **Step 2: Prompt evolution** (`forgestream/governor/improvement.py`)
  - Score agent prompts against their outputs
  - Preserve good variants, adjust bad ones

- [ ] **Step 3: Post-meeting synthesis** (`forgestream/governor/improvement.py`)
  - Generate meeting summary event
  - Write markdown report to docs/meetings/
  - Build human review queue

- [ ] **Step 4: Tests and commit**

```bash
cd ~/projects/forgestream
git add forgestream/governor/ tests/governor/
git commit -m "feat(SP-10): add self-improvement with GRPO weight tuning"
```

**SP-10 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-11: Audio Copilot

Depends on SP-3 and SP-4. Experimental.

- [ ] **Step 1: TTS session** (`forgestream/copilot/tts.py`) — second Gemini connection for audio output
- [ ] **Step 2: Injection rules** (`forgestream/copilot/injection.py`) — pause detection, cooldown, word limit, priority filter
- [ ] **Step 3: Tests and commit**

**SP-11 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## SP-12: Cross-Meeting Transfer

Depends on SP-2 and SP-9.

- [ ] **Step 1: Merged graph** — combine knowledge graphs across sessions
- [ ] **Step 2: Transfer detection** — identify domain morphisms between meetings
- [ ] **Step 3: Seed garden** — cross-meeting seed tracking and promotion
- [ ] **Step 4: Tests and commit**

**SP-12 COMPLETE.** Mark `[x]` in Progress Tracker.

---

## Final Verification

After all SPs are complete:

- [ ] Run full test suite: `pytest -v --tb=short`
- [ ] Verify no import errors: `python -c "import forgestream"`
- [ ] Verify TUI launches: `python -m forgestream.tui.app` (may need mock data)
- [ ] Verify dashboard launches: `uvicorn forgestream.dashboard.server:app`
- [ ] Review all commits: `git log --oneline`
