# Milestone B: Full Autonomous System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three audio input modes (file replay, microphone, system loopback), Gemini Live API WebSocket streaming, Claude Code agent spawning during meetings, and post-meeting GRPO self-improvement — completing the full SOS autonomous orbit.

**Architecture:** AudioSource abstraction yields PCM 16kHz mono chunks consumed by GeminiLiveStream. AgentDispatcher subscribes to requirement events and spawns Claude Code CLI in tmux. PostMeetingSynthesis runs GRPO weight tuning after each meeting. All components integrate through the existing Orchestrator EventBus.

**Tech Stack:** Python 3.12+, sounddevice, soundfile, pydub, google-genai (Live API), existing ForgeStream modules

**Design Spec:** `~/projects/proofforge/docs/superpowers/specs/2026-03-27-milestone-b-full-autonomy-design.md`

**Existing codebase:** `~/projects/forgestream/` — 156 tests passing, Milestone A complete.

---

## Prerequisites

```bash
cd ~/projects/forgestream
python3 -m pip install sounddevice soundfile pydub
brew install ffmpeg        # required by pydub for m4a/mp3
brew install blackhole-2ch # required for system audio capture (Mode C)
```

---

## File Structure

### New files
```
forgestream/
├── audio/
│   ├── __init__.py         # AudioSource exports
│   ├── source.py           # AudioSource abstract base class
│   ├── file_replay.py      # FileReplaySource (Mode A)
│   ├── microphone.py       # MicrophoneSource (Mode B)
│   └── system_audio.py     # SystemAudioSource (Mode C)
├── live_stream.py          # GeminiLiveStream WebSocket client
├── agent_dispatcher.py     # Agent spawning from requirement events
└── post_meeting.py         # Post-meeting synthesis + GRPO tuning

data/
├── weights.json            # Evaluator weights (created at runtime)
└── .gitkeep

tests/
├── audio/
│   ├── __init__.py
│   ├── test_source.py
│   ├── test_file_replay.py
│   └── test_microphone.py
├── test_live_stream.py
├── test_agent_dispatcher.py
└── test_post_meeting.py
```

### Modified files
```
forgestream/
├── config.py               # Add data_dir, audio device config
└── tui/app.py              # Add /source, /end commands
```

---

## Task B-1: AudioSource Base + FileReplaySource

**Files:**
- Create: `forgestream/audio/__init__.py`
- Create: `forgestream/audio/source.py`
- Create: `forgestream/audio/file_replay.py`
- Create: `tests/audio/__init__.py`
- Create: `tests/audio/test_source.py`
- Create: `tests/audio/test_file_replay.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p ~/projects/forgestream/forgestream/audio
mkdir -p ~/projects/forgestream/tests/audio
touch ~/projects/forgestream/tests/audio/__init__.py
```

- [ ] **Step 2: Write failing tests for AudioSource base**

`tests/audio/test_source.py`:
```python
import asyncio

from forgestream.audio.source import AudioSource


class TestAudioSource:
    def test_defaults(self):
        source = AudioSource()
        assert source.sample_rate == 16000
        assert source.channels == 1
        assert source.chunk_duration == 0.5
        assert source.is_active is False

    def test_chunk_size_bytes(self):
        source = AudioSource()
        # 16kHz * 2 bytes/sample * 1 channel * 0.5s = 16000 bytes
        assert source.chunk_size_bytes == 16000

    def test_cannot_iterate_base_class(self):
        source = AudioSource()
        # Base class chunks() should raise NotImplementedError
        try:
            gen = source.chunks()
            asyncio.get_event_loop().run_until_complete(gen.__anext__())
            assert False, "Should have raised"
        except (NotImplementedError, StopAsyncIteration):
            pass
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/test_source.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 4: Implement AudioSource base**

`forgestream/audio/source.py`:
```python
"""AudioSource abstract base class -- unified interface for all audio inputs."""

from __future__ import annotations

from typing import AsyncIterator


class AudioSource:
    """Abstract base for audio input sources.

    All sources yield PCM 16kHz mono int16 audio chunks.
    Chunk size: 16000 samples/sec * 2 bytes * 0.5s = 16000 bytes per chunk.
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5  # seconds

    def __init__(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def chunk_size_bytes(self) -> int:
        return int(self.sample_rate * 2 * self.channels * self.chunk_duration)

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    async def chunks(self) -> AsyncIterator[bytes]:
        raise NotImplementedError("Subclasses must implement chunks()")
        yield  # make it a generator
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/test_source.py -v
```

Expected: ALL PASS

- [ ] **Step 6: Write failing tests for FileReplaySource**

`tests/audio/test_file_replay.py`:
```python
import asyncio
import struct
import tempfile
import wave
from pathlib import Path

import pytest

from forgestream.audio.file_replay import FileReplaySource
from forgestream.audio.source import AudioSource


def create_test_wav(path: Path, duration_s: float = 2.0, sample_rate: int = 16000) -> None:
    """Create a test WAV file with a sine wave."""
    import math
    n_samples = int(sample_rate * duration_s)
    samples = [int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


class TestFileReplaySource:
    def test_is_audio_source(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            create_test_wav(Path(f.name))
            source = FileReplaySource(f.name)
            assert isinstance(source, AudioSource)

    async def test_yields_chunks(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
            create_test_wav(path, duration_s=1.5)

        source = FileReplaySource(str(path), speed=0)  # as fast as possible
        await source.start()

        chunks = []
        async for chunk in source.chunks():
            chunks.append(chunk)
            if len(chunks) > 10:
                break

        await source.stop()
        path.unlink()

        assert len(chunks) >= 2  # 1.5s / 0.5s = 3 chunks
        assert all(len(c) == source.chunk_size_bytes for c in chunks)

    async def test_folder_of_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                create_test_wav(Path(tmpdir) / f"part_{i}.wav", duration_s=1.0)

            source = FileReplaySource(tmpdir, speed=0)
            await source.start()

            chunks = []
            async for chunk in source.chunks():
                chunks.append(chunk)

            await source.stop()

            # 3 files * 1.0s / 0.5s = 6 chunks
            assert len(chunks) == 6

    async def test_not_active_after_exhausted(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
            create_test_wav(path, duration_s=0.5)

        source = FileReplaySource(str(path), speed=0)
        await source.start()

        async for _ in source.chunks():
            pass

        assert source.is_active is False
        path.unlink()

    def test_speed_multiplier(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            create_test_wav(Path(f.name))
            source = FileReplaySource(f.name, speed=2.0)
            assert source.speed == 2.0
```

- [ ] **Step 7: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/test_file_replay.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 8: Implement FileReplaySource**

`forgestream/audio/file_replay.py`:
```python
"""FileReplaySource -- replay pre-recorded audio files as PCM chunks."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import soundfile as sf

from .source import AudioSource


def _natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


class FileReplaySource(AudioSource):
    """Replays audio files as PCM 16kHz mono chunks.

    Accepts a single file or a folder of audio files (natural sort order).
    speed=1.0 is real-time, speed=0 is as fast as possible.
    """

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

    def __init__(self, path: str, speed: float = 1.0) -> None:
        super().__init__()
        self.path = Path(path)
        self.speed = speed
        self._files: list[Path] = []
        self._resolve_files()

    def _resolve_files(self) -> None:
        if self.path.is_dir():
            self._files = sorted(
                [f for f in self.path.iterdir() if f.suffix.lower() in self.AUDIO_EXTENSIONS],
                key=_natural_sort_key,
            )
        elif self.path.is_file():
            self._files = [self.path]

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks from all audio files in sequence."""
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        for audio_file in self._files:
            try:
                data, sr = sf.read(audio_file, dtype="int16", always_2d=True)
            except Exception:
                # Use pydub for formats soundfile can't handle (m4a, mp3)
                from pydub import AudioSegment
                seg = AudioSegment.from_file(str(audio_file))
                seg = seg.set_channels(1).set_frame_rate(self.sample_rate).set_sample_width(2)
                data = np.frombuffer(seg.raw_data, dtype=np.int16).reshape(-1, 1)
                sr = self.sample_rate

            # Resample if needed
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                indices = np.arange(0, len(data), 1 / ratio).astype(int)
                indices = indices[indices < len(data)]
                data = data[indices]

            # Convert to mono if stereo
            if data.ndim > 1 and data.shape[1] > 1:
                data = data.mean(axis=1).astype(np.int16)
            elif data.ndim > 1:
                data = data[:, 0]

            # Yield in chunks
            for i in range(0, len(data), chunk_samples):
                chunk_data = data[i : i + chunk_samples]
                if len(chunk_data) < chunk_samples:
                    # Pad last chunk with silence
                    chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)))
                yield chunk_data.astype(np.int16).tobytes()

                if self.speed > 0:
                    await asyncio.sleep(self.chunk_duration / self.speed)

        self._active = False
```

`forgestream/audio/__init__.py`:
```python
"""Audio input sources for ForgeStream."""
from .source import AudioSource
from .file_replay import FileReplaySource

__all__ = ["AudioSource", "FileReplaySource"]
```

- [ ] **Step 9: Install numpy (dependency of soundfile)**

```bash
cd ~/projects/forgestream
python3 -m pip install numpy soundfile pydub
```

- [ ] **Step 10: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/ -v
```

Expected: ALL PASS

- [ ] **Step 11: Run full suite — no regressions**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

Expected: 156+ passed

- [ ] **Step 12: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/audio/ tests/audio/
git commit -m "feat(B-1): add AudioSource base and FileReplaySource"
```

---

## Task B-2: MicrophoneSource

**Files:**
- Create: `forgestream/audio/microphone.py`
- Create: `tests/audio/test_microphone.py`

- [ ] **Step 1: Write failing tests**

`tests/audio/test_microphone.py`:
```python
from unittest.mock import patch, MagicMock

from forgestream.audio.microphone import MicrophoneSource
from forgestream.audio.source import AudioSource


class TestMicrophoneSource:
    def test_is_audio_source(self):
        source = MicrophoneSource()
        assert isinstance(source, AudioSource)

    def test_default_device(self):
        source = MicrophoneSource()
        assert source.device is None  # None = system default

    def test_custom_device(self):
        source = MicrophoneSource(device=3)
        assert source.device == 3

    @patch("forgestream.audio.microphone.sd")
    def test_list_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        devices = MicrophoneSource.list_input_devices()
        assert len(devices) == 2
        assert devices[0]["name"] == "Built-in Microphone"

    @patch("forgestream.audio.microphone.sd")
    async def test_start_opens_stream(self, mock_sd):
        source = MicrophoneSource()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await source.start()
        assert source.is_active is True
        mock_sd.InputStream.assert_called_once()

    @patch("forgestream.audio.microphone.sd")
    async def test_stop_closes_stream(self, mock_sd):
        source = MicrophoneSource()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await source.start()
        await source.stop()
        assert source.is_active is False
        mock_stream.stop.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/test_microphone.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement MicrophoneSource**

`forgestream/audio/microphone.py`:
```python
"""MicrophoneSource -- capture live audio from system microphone."""

from __future__ import annotations

import asyncio
import queue
from typing import Any, AsyncIterator

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

from .source import AudioSource


class MicrophoneSource(AudioSource):
    """Captures audio from the system microphone via sounddevice.

    Uses a callback-based InputStream that fills a queue.
    The chunks() async generator reads from the queue.
    """

    def __init__(self, device: int | None = None) -> None:
        super().__init__()
        self.device = device
        self._stream = None
        self._queue: queue.Queue[bytes] = queue.Queue()

    @staticmethod
    def list_input_devices() -> list[dict[str, Any]]:
        """List available audio input devices."""
        if not HAS_SOUNDDEVICE:
            return []
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Called by sounddevice when audio data is available."""
        # Convert to int16 and put in queue
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        self._queue.put(audio_int16.tobytes())

    async def start(self) -> None:
        """Open the audio input stream."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not installed. pip install sounddevice")

        chunk_samples = int(self.sample_rate * self.chunk_duration)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_samples,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._active = True

    async def stop(self) -> None:
        """Close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks from the microphone."""
        while self._active:
            try:
                chunk = self._queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                await asyncio.sleep(0.05)
```

Update `forgestream/audio/__init__.py`:
```python
"""Audio input sources for ForgeStream."""
from .source import AudioSource
from .file_replay import FileReplaySource
from .microphone import MicrophoneSource

__all__ = ["AudioSource", "FileReplaySource", "MicrophoneSource"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/test_microphone.py -v
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
git add forgestream/audio/ tests/audio/test_microphone.py
git commit -m "feat(B-2): add MicrophoneSource with sounddevice capture"
```

---

## Task B-3: SystemAudioSource (BlackHole)

**Files:**
- Create: `forgestream/audio/system_audio.py`

- [ ] **Step 1: Implement SystemAudioSource**

`forgestream/audio/system_audio.py`:
```python
"""SystemAudioSource -- capture system audio via BlackHole virtual device."""

from __future__ import annotations

from typing import Any

from .microphone import MicrophoneSource

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False


class SystemAudioSource(MicrophoneSource):
    """Captures system audio output via BlackHole virtual audio device.

    BlackHole must be installed: brew install blackhole-2ch

    For full meeting capture (your voice + remote participants):
    1. Open Audio MIDI Setup
    2. Create Aggregate Device combining mic + BlackHole
    3. Use that aggregate device as the input
    """

    BLACKHOLE_NAMES = ["BlackHole 2ch", "BlackHole 16ch", "BlackHole"]

    def __init__(self, device: int | None = None) -> None:
        if device is None:
            device = self._find_blackhole_device()
        super().__init__(device=device)

    @classmethod
    def _find_blackhole_device(cls) -> int | None:
        """Find the BlackHole device index."""
        if not HAS_SOUNDDEVICE:
            return None
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                for name in cls.BLACKHOLE_NAMES:
                    if name.lower() in d["name"].lower():
                        return i
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if BlackHole is installed and available."""
        return cls._find_blackhole_device() is not None

    @classmethod
    def get_device_info(cls) -> dict[str, Any] | None:
        """Get BlackHole device information."""
        if not HAS_SOUNDDEVICE:
            return None
        idx = cls._find_blackhole_device()
        if idx is not None:
            info = sd.query_devices(idx)
            return {"index": idx, "name": info["name"], "channels": info["max_input_channels"]}
        return None
```

Update `forgestream/audio/__init__.py`:
```python
"""Audio input sources for ForgeStream."""
from .source import AudioSource
from .file_replay import FileReplaySource
from .microphone import MicrophoneSource
from .system_audio import SystemAudioSource

__all__ = ["AudioSource", "FileReplaySource", "MicrophoneSource", "SystemAudioSource"]
```

- [ ] **Step 2: Write test**

Add to `tests/audio/test_microphone.py`:

```python
from forgestream.audio.system_audio import SystemAudioSource


class TestSystemAudioSource:
    def test_is_microphone_source(self):
        source = SystemAudioSource(device=0)
        assert isinstance(source, MicrophoneSource)

    @patch("forgestream.audio.system_audio.sd")
    def test_find_blackhole(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        idx = SystemAudioSource._find_blackhole_device()
        assert idx == 1

    @patch("forgestream.audio.system_audio.sd")
    def test_is_available(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        assert SystemAudioSource.is_available() is True

    @patch("forgestream.audio.system_audio.sd")
    def test_not_available_without_blackhole(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
        ]
        assert SystemAudioSource.is_available() is False
```

- [ ] **Step 3: Run tests and full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/audio/ -v
python3 -m pytest -q
```

- [ ] **Step 4: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/audio/ tests/audio/
git commit -m "feat(B-3): add SystemAudioSource with BlackHole device detection"
```

---

## Task B-4: GeminiLiveStream Client

**Files:**
- Create: `forgestream/live_stream.py`
- Create: `tests/test_live_stream.py`

- [ ] **Step 1: Write failing tests**

`tests/test_live_stream.py`:
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from forgestream.audio.source import AudioSource
from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import EventType
from forgestream.live_stream import GeminiLiveStream
from forgestream.orchestrator import Orchestrator


class MockAudioSource(AudioSource):
    """Test audio source that yields a fixed number of chunks."""

    def __init__(self, num_chunks: int = 3) -> None:
        super().__init__()
        self._num_chunks = num_chunks

    async def chunks(self):
        for _ in range(self._num_chunks):
            yield b"\x00" * self.chunk_size_bytes
        self._active = False


class TestGeminiLiveStream:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = MockAudioSource()

        stream = GeminiLiveStream(
            config=config,
            orchestrator=orch,
            audio_source=source,
            mode="collaborative",
        )
        assert stream.mode == "collaborative"
        assert stream.audio_source is source

    def test_mode_instructions_exist(self):
        from forgestream.live_stream import MODE_INSTRUCTIONS
        assert "extract" in MODE_INSTRUCTIONS
        assert "collaborative" in MODE_INSTRUCTIONS
        assert "knowledge" in MODE_INSTRUCTIONS

    def test_parse_jsonl_valid(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())

        text = '{"text": "hello", "confidence": 0.9, "topic_keywords": ["test"]}\n{"text": "world", "confidence": 0.8, "topic_keywords": ["test2"]}'
        claims = stream._parse_jsonl(text)
        assert len(claims) == 2
        assert claims[0]["text"] == "hello"

    def test_parse_jsonl_handles_garbage(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())

        text = '```json\n{"text": "hello", "confidence": 0.9}\nnot json\n```'
        claims = stream._parse_jsonl(text)
        assert len(claims) == 1

    def test_set_mode(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        stream = GeminiLiveStream(config=config, orchestrator=orch,
                                   audio_source=MockAudioSource())
        stream.set_mode("knowledge")
        assert stream.mode == "knowledge"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_live_stream.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement GeminiLiveStream**

`forgestream/live_stream.py`:
```python
"""GeminiLiveStream -- WebSocket bridge between AudioSource and Orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from .audio.source import AudioSource
from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .gemini.context import ContextBuilder
from .gemini.extraction import ClaimExtractor
from .graph.materializer import GraphMaterializer
from .orchestrator import Orchestrator
from .synthesis.requirements import RequirementDetector

logger = logging.getLogger(__name__)

MODE_INSTRUCTIONS = {
    "extract": (
        "You are an ECEF knowledge extractor in a meeting. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on what the expert needs built. Extract requirements, "
        "constraints, tech preferences. Do NOT summarize. "
        "Extract EVERY claim. Emit in real-time."
    ),
    "collaborative": (
        "You are an ECEF knowledge extractor in a design discussion. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on architectural decisions, trade-offs, agreements "
        "and disagreements. Emit in real-time."
    ),
    "knowledge": (
        "You are an ECEF knowledge extractor doing expertise capture. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on domain knowledge, mental models, heuristics, "
        "and tacit knowledge. Emit in real-time."
    ),
}


class GeminiLiveStream:
    """Manages the Gemini Live API WebSocket session.

    Connects an AudioSource to the Orchestrator:
    AudioSource → PCM chunks → Gemini → claims → Orchestrator → EventBus → TUI
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        orchestrator: Orchestrator,
        audio_source: AudioSource,
        mode: str = "extract",
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.audio_source = audio_source
        self.mode = mode

        self._session = None
        self._active = False
        self._tasks: list[asyncio.Task] = []

        self.branch_id = uuid4()
        self.extractor = ClaimExtractor(
            session_id=orchestrator.session_id,
            branch_id=self.branch_id,
        )
        self.context_builder = ContextBuilder()
        self.req_detector = RequirementDetector()
        self.materializer = GraphMaterializer()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _parse_jsonl(self, text: str) -> list[dict[str, Any]]:
        """Parse JSONL text, skipping invalid lines."""
        claims = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                claims.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return claims

    async def connect(self) -> None:
        """Establish WebSocket connection to Gemini Live API."""
        try:
            from google import genai

            client = genai.Client(
                vertexai=self.config.gemini_use_vertex,
                project=self.config.gemini_project,
                location=self.config.gemini_location,
            )

            self._session = await client.aio.live.connect(
                model=self.config.gemini_model,
                config={
                    "response_modalities": ["TEXT"],
                    "system_instruction": MODE_INSTRUCTIONS.get(
                        self.mode, MODE_INSTRUCTIONS["extract"]
                    ),
                },
            )
            self._active = True
            logger.info("Connected to Gemini Live API")
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. pip install google-genai"
            )

    async def start(self) -> None:
        """Start streaming audio and receiving claims."""
        await self.audio_source.start()
        self._tasks = [
            asyncio.create_task(self._send_loop()),
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._context_injection_loop()),
        ]

    async def stop(self) -> None:
        """Stop all loops and disconnect."""
        self._active = False
        await self.audio_source.stop()
        for task in self._tasks:
            task.cancel()
        if self._session:
            await self._session.close()
            self._session = None

    async def _send_loop(self) -> None:
        """Send audio chunks to Gemini."""
        try:
            async for chunk in self.audio_source.chunks():
                if not self._active:
                    break
                if self._session:
                    await self._session.send(
                        {"data": chunk, "mime_type": "audio/pcm"}
                    )
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Receive and process claims from Gemini."""
        if not self._session:
            return

        try:
            async for response in self._session.receive():
                if not self._active:
                    break
                if hasattr(response, "text") and response.text:
                    for claim_data in self._parse_jsonl(response.text):
                        event = self.extractor.parse_claim(claim_data)
                        await self.orchestrator.process_event(event)

                        # Auto-detect requirements and create suggestions
                        req = self.req_detector.check(event)
                        if req:
                            suggestion = Event(
                                event_type=EventType.SUGGESTION,
                                session_id=self.orchestrator.session_id,
                                branch_id=self.branch_id,
                                author="synthesis",
                                evaluator=0.0,
                                payload={
                                    "text": f"Scaffold: {req['description'][:60]}",
                                    "priority": 0.7,
                                },
                            )
                            await self.orchestrator.process_event(suggestion)
        except asyncio.CancelledError:
            pass

    async def _context_injection_loop(self) -> None:
        """Inject knowledge graph summary every 10 minutes."""
        try:
            while self._active:
                await asyncio.sleep(600)  # 10 minutes
                if self._session and self._active:
                    events = self.orchestrator._event_buffer
                    graph = self.materializer.materialize(events)
                    summary = self.context_builder.build_injection(graph, [])
                    await self._session.send({"text": summary})
                    logger.info("Context injection sent")
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_live_stream.py -v
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
git add forgestream/live_stream.py tests/test_live_stream.py
git commit -m "feat(B-4): add GeminiLiveStream with WebSocket send/receive/injection loops"
```

---

## Task B-5: AgentDispatcher

**Files:**
- Create: `forgestream/agent_dispatcher.py`
- Create: `tests/test_agent_dispatcher.py`

- [ ] **Step 1: Write failing tests**

`tests/test_agent_dispatcher.py`:
```python
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from forgestream.agent_dispatcher import AgentDispatcher
from forgestream.agents.registry import AgentRegistry, AgentStatus, AgentType
from forgestream.agents.spawner import SpawnPolicy
from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.governor.trust_region import TrustRegion
from forgestream.orchestrator import Orchestrator


class TestAgentDispatcher:
    def test_initializes(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)
        assert dispatcher.registry is not None
        assert dispatcher.spawn_policy is not None

    async def test_handles_requirement_event(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        event = Event(
            event_type=EventType.REQUIREMENT,
            session_id=orch.session_id,
            branch_id=uuid4(),
            author="synthesis",
            evaluator=0.5,
            payload={
                "description": "Build a data pipeline",
                "domain": "data-engineering",
                "complexity_estimate": 0.5,
                "linked_claims": [],
            },
        )

        # With spawning disabled (trust region default), it should queue
        result = dispatcher.should_spawn(event)
        assert isinstance(result, bool)

    def test_build_research_prompt(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        prompt = dispatcher.build_prompt(
            agent_type=AgentType.RESEARCH,
            description="Research Kafka best practices",
            context_claims=["Expert said Kafka is fast"],
        )
        assert "Kafka" in prompt
        assert len(prompt) > 50

    def test_build_scaffold_prompt(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        prompt = dispatcher.build_prompt(
            agent_type=AgentType.SCAFFOLD,
            description="Build a data pipeline",
            context_claims=["Need sub-100ms latency"],
        )
        assert "pipeline" in prompt
        assert len(prompt) > 50

    @patch("forgestream.agent_dispatcher.subprocess")
    def test_spawn_creates_tmux_session(self, mock_subprocess):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        agent = dispatcher.spawn_agent(
            agent_type=AgentType.RESEARCH,
            description="Research test",
            prompt="Test prompt",
        )

        assert agent.status == AgentStatus.RUNNING
        assert agent.agent_type == AgentType.RESEARCH
        mock_subprocess.run.assert_called()  # tmux new-session called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_agent_dispatcher.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 3: Implement AgentDispatcher**

`forgestream/agent_dispatcher.py`:
```python
"""AgentDispatcher -- spawns Claude Code agents from requirement events."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .agents.registry import AgentInfo, AgentRegistry, AgentStatus, AgentType
from .agents.spawner import SpawnPolicy
from .agents.templates.research import ResearchTemplate
from .agents.templates.scaffold import ScaffoldTemplate
from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .governor.trust_region import TrustRegion
from .orchestrator import Orchestrator


class AgentDispatcher:
    """Spawns Claude Code CLI agents in tmux when requirements are detected.

    Subscribes to the orchestrator EventBus. When a requirement event arrives,
    checks SpawnPolicy, builds a prompt, and launches claude -p in tmux.
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        orchestrator: Orchestrator,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.registry = AgentRegistry()
        self.trust_region = TrustRegion()
        self.spawn_policy = SpawnPolicy(
            registry=self.registry, trust_region=self.trust_region
        )
        self.research_template = ResearchTemplate()
        self.scaffold_template = ScaffoldTemplate()

        # Ensure temp directory exists
        Path("/tmp/forgestream").mkdir(parents=True, exist_ok=True)

    def should_spawn(self, event: Event) -> bool:
        """Check if we should spawn an agent for this event."""
        if event.event_type != EventType.REQUIREMENT:
            return False

        # Check trust region limits
        decision = self.spawn_policy.can_spawn(AgentType.RESEARCH)
        return decision.allowed

    def build_prompt(
        self,
        agent_type: AgentType,
        description: str,
        context_claims: list[str],
    ) -> str:
        """Build a prompt for the agent using the appropriate template."""
        if agent_type == AgentType.RESEARCH:
            return self.research_template.build_prompt(
                query=description,
                context_claims=context_claims,
            )
        else:
            return self.scaffold_template.build_prompt(
                requirement=description,
                domain="",
                verified_findings=context_claims,
            )

    def spawn_agent(
        self,
        agent_type: AgentType,
        description: str,
        prompt: str,
    ) -> AgentInfo:
        """Spawn a Claude Code agent in a tmux session."""
        agent = self.registry.register(
            agent_type=agent_type,
            task_description=description,
        )

        # Write prompt to temp file to avoid shell escaping
        prompt_file = f"/tmp/forgestream/prompt-{agent.id}.md"
        Path(prompt_file).write_text(prompt)

        agent.tmux_session = f"{agent_type.value}-{agent.id}"

        if agent_type == AgentType.RESEARCH:
            cmd = (
                f"claude -p \"$(cat {prompt_file})\" "
                f"--allowedTools 'WebSearch,WebFetch,Read,Grep,Glob' "
                f"2>&1 | tee /tmp/forgestream/agent-{agent.id}.out"
            )
        else:
            cmd = (
                f"claude -p \"$(cat {prompt_file})\" "
                f"2>&1 | tee /tmp/forgestream/agent-{agent.id}.out"
            )

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", agent.tmux_session, cmd],
            capture_output=True,
        )

        self.registry.update_status(agent.id, AgentStatus.RUNNING)
        self.spawn_policy.record_spawn()
        return agent

    async def on_event(self, event: Event) -> None:
        """EventBus handler — check if we should spawn an agent."""
        if not self.should_spawn(event):
            return

        description = event.payload.get("description", "")
        recent_claims = [
            e.payload.get("text", "")
            for e in self.orchestrator._event_buffer[-10:]
            if e.event_type == EventType.CLAIM
        ]

        prompt = self.build_prompt(
            agent_type=AgentType.RESEARCH,
            description=description,
            context_claims=recent_claims,
        )

        self.spawn_agent(
            agent_type=AgentType.RESEARCH,
            description=description,
            prompt=prompt,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_agent_dispatcher.py -v
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
git add forgestream/agent_dispatcher.py tests/test_agent_dispatcher.py
git commit -m "feat(B-5): add AgentDispatcher with Claude Code tmux spawning"
```

---

## Task B-6: PostMeetingSynthesis

**Files:**
- Create: `forgestream/post_meeting.py`
- Create: `tests/test_post_meeting.py`
- Create: `data/.gitkeep`

- [ ] **Step 1: Create data directory**

```bash
mkdir -p ~/projects/forgestream/data
touch ~/projects/forgestream/data/.gitkeep
```

- [ ] **Step 2: Write failing tests**

`tests/test_post_meeting.py`:
```python
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.post_meeting import PostMeetingSynthesis


class TestPostMeetingSynthesis:
    def _make_meeting_events(self) -> list[Event]:
        sid = uuid4()
        bid = uuid4()
        return [
            Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                  author="gemini", evaluator=0.4,
                  payload={"text": "Use Kafka", "topic_keywords": ["Kafka"], "confidence": 0.9}),
            Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                  author="gemini", evaluator=0.42,
                  payload={"text": "Sub-100ms latency", "topic_keywords": ["latency"],
                           "confidence": 0.85, "is_requirement": True}),
            Event(event_type=EventType.VERIFIED_FINDING, session_id=sid, branch_id=bid,
                  author="research", evaluator=0.5,
                  payload={"finding": "Kafka achieves 10ms p99", "confidence": 0.9,
                           "sources": [{"url": "https://kafka.apache.org"}]}),
            Event(event_type=EventType.ARTIFACT, session_id=sid, branch_id=bid,
                  author="scaffold", evaluator=0.55,
                  payload={"compiles": True, "tests_pass": True,
                           "files_created": ["pipeline.py"]}),
        ]

    def test_generate_report(self):
        config = ForgeStreamConfig()
        synthesis = PostMeetingSynthesis(config)
        events = self._make_meeting_events()

        report = synthesis.generate_report(events, meeting_name="Test Meeting")
        assert "Test Meeting" in report
        assert "Kafka" in report or "claims" in report.lower()

    def test_save_report(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.meetings_dir = tmpdir
            synthesis = PostMeetingSynthesis(config)
            events = self._make_meeting_events()

            path = synthesis.save_report(events, meeting_name="test-meeting")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "test-meeting" in content.lower() or "Test" in content

    def test_tune_weights(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)
            events = self._make_meeting_events()

            old_weights = synthesis.load_weights()
            new_weights = synthesis.tune_weights(events, human_score=0.8)

            assert set(new_weights.keys()) >= {"knowledge", "verification", "scaffold", "uptake"}
            assert abs(sum(new_weights.values()) - 1.0) < 0.01

    def test_save_and_load_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ForgeStreamConfig()
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)

            weights = {"knowledge": 0.25, "verification": 0.35, "scaffold": 0.25, "uptake": 0.15}
            synthesis.save_weights(weights, meeting_count=1)

            loaded = synthesis.load_weights()
            assert loaded["knowledge"] == 0.25
            assert loaded["verification"] == 0.35

    def test_compute_auto_score(self):
        config = ForgeStreamConfig()
        synthesis = PostMeetingSynthesis(config)
        events = self._make_meeting_events()

        score = synthesis.compute_auto_score(events)
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_post_meeting.py -v
```

Expected: FAIL — ImportError

- [ ] **Step 4: Implement PostMeetingSynthesis**

`forgestream/post_meeting.py`:
```python
"""Post-meeting synthesis -- GRPO weight tuning, reports, knowledge persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ForgeStreamConfig
from .events.schema import Event, EventType
from .governor.evaluator import Evaluator
from .governor.improvement import MeetingSynthesizer, PromptEvolution, WeightTuner
from .governor.trust_region import TrustRegion


class PostMeetingSynthesis:
    """Runs after each meeting to improve the system.

    Phase 1: Generate meeting report
    Phase 2: Tune evaluator weights (GRPO)
    Phase 3: Score and evolve prompts
    Phase 4: Update trust region
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        data_dir: str | None = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir or "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.synthesizer = MeetingSynthesizer()
        self.weight_tuner = WeightTuner()
        self.prompt_evolution = PromptEvolution()
        self.evaluator = Evaluator()

    def generate_report(self, events: list[Event], meeting_name: str = "") -> str:
        """Generate a markdown meeting report."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]

        e_final = events[-1].evaluator if events else 0.0

        lines = [
            f"# Meeting: {meeting_name or 'Untitled'}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"**Claims:** {len(claims)}",
            f"**E(pi) final:** {e_final:.3f}",
            "",
            "## Knowledge Extracted",
            f"- {len(claims)} claims captured",
            f"- {len(findings)} verified findings",
            f"- {len(requirements)} requirements detected",
            f"- {len(artifacts)} artifacts produced",
            "",
        ]

        if requirements:
            lines.append("## Requirements")
            for r in requirements[:15]:
                lines.append(f"- {r.payload.get('description', 'N/A')[:80]}")
            lines.append("")

        if artifacts:
            lines.append("## Artifacts")
            for a in artifacts[:10]:
                compiles = a.payload.get("compiles", False)
                tests = a.payload.get("tests_pass", False)
                files = a.payload.get("files_created", [])
                status = "pass" if compiles and tests else "partial" if compiles else "fail"
                lines.append(f"- [{status}] {len(files)} files")
            lines.append("")

        lines.extend([
            "## SOS Status",
            "- Evaluator weights: see data/weights.json",
            f"- Final E(pi): {e_final:.3f}",
            "",
            "## Human Review Queue",
            "- [ ] Review detected requirements",
            "- [ ] Check scaffold artifacts",
            "- [ ] Promote or archive seeds",
        ])

        return "\n".join(lines)

    def save_report(self, events: list[Event], meeting_name: str = "") -> str:
        """Save meeting report to docs/meetings/."""
        report = self.generate_report(events, meeting_name)
        meetings_dir = Path(self.config.meetings_dir)
        meetings_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = meeting_name.lower().replace(" ", "-")[:30] if meeting_name else "meeting"
        filename = f"{date_str}-{slug}.md"
        path = meetings_dir / filename
        path.write_text(report)
        return str(path)

    def load_weights(self) -> dict[str, float]:
        """Load evaluator weights from disk."""
        weights_file = self.data_dir / "weights.json"
        if weights_file.exists():
            data = json.loads(weights_file.read_text())
            return {k: v for k, v in data.items()
                    if k in ("knowledge", "verification", "scaffold", "uptake")}
        return Evaluator.DEFAULT_WEIGHTS.copy()

    def save_weights(
        self, weights: dict[str, float], meeting_count: int = 0
    ) -> None:
        """Save evaluator weights to disk."""
        data = {
            **weights,
            "meeting_count": meeting_count,
            "last_tuned": datetime.now(timezone.utc).isoformat(),
        }
        weights_file = self.data_dir / "weights.json"
        weights_file.write_text(json.dumps(data, indent=2))

        # Append to history
        history_file = self.data_dir / "weights_history.json"
        history: list = []
        if history_file.exists():
            history = json.loads(history_file.read_text())
        history.append(data)
        history_file.write_text(json.dumps(history, indent=2))

    def tune_weights(
        self,
        events: list[Event],
        human_score: float | None = None,
    ) -> dict[str, float]:
        """Run GRPO weight tuning against meeting events."""
        current = self.load_weights()
        target = human_score if human_score is not None else self.compute_auto_score(events)
        updated = self.weight_tuner.tune(current, events, human_score=target)
        return updated

    def compute_auto_score(self, events: list[Event]) -> float:
        """Compute automatic meeting quality score."""
        claims = [e for e in events if e.event_type == EventType.CLAIM]
        requirements = [e for e in events if e.event_type == EventType.REQUIREMENT]
        artifacts = [e for e in events if e.event_type == EventType.ARTIFACT]
        findings = [e for e in events if e.event_type == EventType.VERIFIED_FINDING]

        req_scaffold = (
            len(artifacts) / max(len(requirements), 1)
            if requirements else 0.0
        )
        findings_per_claim = (
            len(findings) / max(len(claims), 1)
            if claims else 0.0
        )

        return min(1.0, (
            0.3 * min(1.0, req_scaffold)
            + 0.3 * min(1.0, findings_per_claim)
            + 0.2 * 0.5  # suggestion uptake placeholder
            + 0.2 * 0.5  # branch merge rate placeholder
        ))

    async def run(
        self,
        events: list[Event],
        meeting_name: str = "",
        human_score: float | None = None,
    ) -> dict[str, Any]:
        """Run the full post-meeting synthesis pipeline."""
        # Phase 1: Report
        report_path = self.save_report(events, meeting_name)

        # Phase 2: Weight tuning
        new_weights = self.tune_weights(events, human_score)
        current_weights = self.load_weights()
        meeting_count = 1  # increment from stored value in production
        self.save_weights(new_weights, meeting_count)

        # Phase 3: Trust region (compute E_meso)
        e_meso = self.evaluator.compute(events)

        return {
            "report_path": report_path,
            "weights": new_weights,
            "e_meso": e_meso,
            "meeting_count": meeting_count,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_post_meeting.py -v
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
git add forgestream/post_meeting.py tests/test_post_meeting.py data/
git commit -m "feat(B-6): add PostMeetingSynthesis with GRPO tuning and reports"
```

---

## Task B-7: TUI Integration

**Files:**
- Modify: `forgestream/tui/app.py`
- Modify: `forgestream/config.py`

- [ ] **Step 1: Add data_dir to config**

In `forgestream/config.py`, add to `ForgeStreamConfig`:

```python
    # Data persistence
    data_dir: str = "data"
```

- [ ] **Step 2: Add /source and /end commands to TUI**

Add these bindings and actions to `forgestream/tui/app.py`:

Add to BINDINGS list:
```python
        Binding("e", "end_meeting", "End"),
```

Add these methods to `ForgeStreamApp`:

```python
    def action_end_meeting(self) -> None:
        """End the current meeting and trigger post-meeting synthesis."""
        feed = self.query_one("#feed", FeedPanel)
        feed.write("[bold cyan]>>> MEETING ENDED — Running post-meeting synthesis...[/bold cyan]")
        self.sub_title = "Meeting Ended"
```

- [ ] **Step 3: Run full suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

- [ ] **Step 4: Commit**

```bash
cd ~/projects/forgestream
git add forgestream/tui/app.py forgestream/config.py
git commit -m "feat(B-7): add /end command and data_dir config"
```

---

## Task B-8: Integration Tests

**Files:**
- Create: `tests/test_milestone_b.py`

- [ ] **Step 1: Write integration tests**

`tests/test_milestone_b.py`:
```python
"""Milestone B integration tests -- audio → live stream → agents → GRPO."""

import math
import struct
import tempfile
import wave
from pathlib import Path
from uuid import uuid4

from forgestream.audio.file_replay import FileReplaySource
from forgestream.audio.microphone import MicrophoneSource
from forgestream.audio.source import AudioSource
from forgestream.audio.system_audio import SystemAudioSource
from forgestream.config import ForgeStreamConfig
from forgestream.events.schema import Event, EventType
from forgestream.live_stream import GeminiLiveStream
from forgestream.agent_dispatcher import AgentDispatcher
from forgestream.orchestrator import Orchestrator
from forgestream.post_meeting import PostMeetingSynthesis


def _create_test_wav(path: Path, duration_s: float = 1.0) -> None:
    n_samples = int(16000 * duration_s)
    samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


class TestAudioSourceUniformInterface:
    """All three audio sources produce the same chunk format."""

    async def test_file_replay_chunk_size(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _create_test_wav(Path(f.name))

        source = FileReplaySource(f.name, speed=0)
        await source.start()

        async for chunk in source.chunks():
            assert len(chunk) == source.chunk_size_bytes
            break

        await source.stop()
        Path(f.name).unlink()

    def test_all_sources_share_base(self):
        assert issubclass(FileReplaySource, AudioSource)
        assert issubclass(MicrophoneSource, AudioSource)
        assert issubclass(SystemAudioSource, AudioSource)

    def test_all_sources_same_sample_rate(self):
        f = FileReplaySource.__new__(FileReplaySource)
        m = MicrophoneSource.__new__(MicrophoneSource)
        s = SystemAudioSource.__new__(SystemAudioSource)
        assert f.sample_rate == m.sample_rate == s.sample_rate == 16000

    def test_all_sources_same_chunk_size(self):
        f = FileReplaySource.__new__(FileReplaySource)
        m = MicrophoneSource.__new__(MicrophoneSource)
        assert f.chunk_size_bytes == m.chunk_size_bytes == 16000


class TestGeminiLiveStreamUnit:
    def test_stream_connects_source_to_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        source = FileReplaySource.__new__(FileReplaySource)
        source._active = False

        stream = GeminiLiveStream(
            config=config, orchestrator=orch,
            audio_source=source, mode="collaborative"
        )
        assert stream.orchestrator is orch
        assert stream.audio_source is source


class TestAgentDispatcherUnit:
    def test_dispatcher_integrates_with_orchestrator(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)
        assert dispatcher.orchestrator is orch

    def test_prompt_includes_context(self):
        config = ForgeStreamConfig()
        orch = Orchestrator(config)
        dispatcher = AgentDispatcher(config=config, orchestrator=orch)

        from forgestream.agents.registry import AgentType
        prompt = dispatcher.build_prompt(
            AgentType.RESEARCH, "Test query", ["context claim 1"]
        )
        assert "context claim 1" in prompt


class TestPostMeetingIntegration:
    def test_full_synthesis_pipeline(self):
        config = ForgeStreamConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.meetings_dir = f"{tmpdir}/meetings"
            synthesis = PostMeetingSynthesis(config, data_dir=tmpdir)

            sid = uuid4()
            bid = uuid4()
            events = [
                Event(event_type=EventType.CLAIM, session_id=sid, branch_id=bid,
                      author="gemini", evaluator=0.4,
                      payload={"text": "test", "topic_keywords": ["A"]}),
                Event(event_type=EventType.VERIFIED_FINDING, session_id=sid, branch_id=bid,
                      author="research", evaluator=0.5,
                      payload={"finding": "result", "sources": ["x"], "confidence": 0.9}),
            ]

            # Generate report
            report = synthesis.generate_report(events, "Integration Test")
            assert "Integration Test" in report

            # Save report
            path = synthesis.save_report(events, "integration-test")
            assert Path(path).exists()

            # Tune weights
            weights = synthesis.tune_weights(events, human_score=0.7)
            synthesis.save_weights(weights, meeting_count=1)

            # Load weights back
            loaded = synthesis.load_weights()
            assert abs(sum(loaded.values()) - 1.0) < 0.01
```

- [ ] **Step 2: Run integration tests**

```bash
cd ~/projects/forgestream
python3 -m pytest tests/test_milestone_b.py -v
```

Expected: ALL PASS

- [ ] **Step 3: Run complete test suite**

```bash
cd ~/projects/forgestream
python3 -m pytest -q
```

Expected: 170+ passed, 0 failed

- [ ] **Step 4: Commit and push**

```bash
cd ~/projects/forgestream
git add tests/test_milestone_b.py
git commit -m "test: add Milestone B integration tests — audio, streaming, agents, GRPO"
git push
```

---

## Verification Checklist

After all tasks complete:

- [ ] AudioSource base: `python3 -c "from forgestream.audio import AudioSource; print(AudioSource().chunk_size_bytes)"` → `16000`
- [ ] FileReplay: `python3 -c "from forgestream.audio import FileReplaySource; print('OK')"`
- [ ] Microphone: `python3 -c "from forgestream.audio import MicrophoneSource; print(MicrophoneSource.list_input_devices())"`
- [ ] SystemAudio: `python3 -c "from forgestream.audio import SystemAudioSource; print('Available:', SystemAudioSource.is_available())"`
- [ ] LiveStream: `python3 -c "from forgestream.live_stream import GeminiLiveStream, MODE_INSTRUCTIONS; print(len(MODE_INSTRUCTIONS), 'modes')"`
- [ ] AgentDispatcher: `python3 -c "from forgestream.agent_dispatcher import AgentDispatcher; print('OK')"`
- [ ] PostMeeting: `python3 -c "from forgestream.post_meeting import PostMeetingSynthesis; print('OK')"`
- [ ] Full suite: `python3 -m pytest -q` → all pass
