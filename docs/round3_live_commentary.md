# Round 3 Training — Live Commentary

**Session**: 2026-04-05
**Pod**: `0fsqh08bj252sk` — H100 SXM 80GB, $2.69/hr secure cloud (India DC)
**SSH**: `ssh -i ~/.ssh/id_ed25519 root@103.207.149.101 -p 13977`
**Monitor**: `tail -20 /workspace/round3_stdout.log`
**Live MD**: `scp -i ~/.ssh/id_ed25519 -P 13977 root@103.207.149.101:/workspace/grpo_round3/training_live.md .`

---

## Timeline

### 04:13 UTC — Pod created
H100 SXM secure cloud. 81GB GPU, 251GB RAM, 100GB workspace.

### 04:16 UTC — Pre-flight dry run
First run failed: protobuf import name (`protobuf` → `google.protobuf`), Lean not installed.
Fixed both. Second dry run: ALL 6 CHECKS PASSED. 5/5 sample theorems verified in Lean.

### 04:18 UTC — Training launched
nohup via round3_launcher.py. Config:
- Model: DeepSeek-Prover-V2-7B + LoRA
- Group size: 32
- Steps: 200
- Curriculum: 178 theorems (merged from 3 files)
- Checkpoint schedule: 0, 5, 10, 20, 30, 50, 75, 100, 150, 200

### 04:19 UTC — Model download
HuggingFace download: 2 files, ~14GB. Took ~70s.

### 04:20 UTC — Model loaded
Weights loaded in 13s (273 shards at ~20 it/s). 14,275 MiB GPU memory.
LoRA applied. Stage 1 (RS-SFT) directory created.

### 04:20-04:22 UTC — Stage 1 generation starting
GPU at 47% utilization, 184W. Model generating 178 theorems × 200 completions = 35,600 total.
stdout buffer not flushing (nohup issue) but process is alive at 320% CPU.

With KV cache fix (Issue #6), estimated Stage 1 generation: ~4-6 hours.
Without the fix it would have been 30+ days. The fix was critical.

### 04:25 UTC — Stage 1 generation active
Process PID 1187 alive at 234% CPU, 14:48 cumulative CPU time. 14,289 MiB GPU memory.
stdout log buffered (not flushing to disk due to nohup) but process is actively generating.
sft_checkpoint/ directory created — Stage 1 has begun.
35,600 completions being generated. With KV cache fix, estimated ~4-6 hours.
GPU at 47% utilization suggests generation is happening but not maxing out (single-sequence generation).

### Notes on buffering issue
The nohup stdout redirect causes Python to use full buffering instead of line buffering.
The training_live.md is piped through subprocess.PIPE which also buffers.
Actual progress is ahead of what the logs show. Process health confirmed via ps + GPU stats.

### What to watch for
- GPU memory spike to 30-40GB = generation active with activations
- sft_checkpoint/ getting files = Stage 1 SFT starting
- checkpoint_000/ appearing = Stage 2 started
- training_live.md growing = output pipeline flushing

### Cost tracking
Started: 04:13 UTC
Rate: $2.69/hr
Current elapsed: ~12 min
Current cost: ~$0.54

### 04:30 UTC — Deep health check: HEALTHY
Process state: R (running), 290 threads, responding to signals.
GPU: consistent 46-47% utilization, 184W, 14,289 MiB (weights only — normal for no_grad generation).
CPU: 200 ticks/2s = exactly 1 core saturated (Python main thread doing tokenization between GPU calls).
No lean temp files — Stage 1 generates ALL 35,600 completions first, THEN batch-verifies.
No output files yet — all buffered until generation completes.
HF cache: model loaded from memory-mapped safetensors.

Pattern: steady GPU 46% + CPU 100% = textbook single-sequence autoregressive generation.
This will continue for ~3-6 hours. Next status change: lean processes appearing = verification started.

### Cost at 04:30 UTC
Elapsed: 17 min, Cost: ~$0.76

### 06:16 UTC — 2 hours in, steady state
Monitor log shows consistent heartbeats every 15 min since 05:46.
GPU: 14,315 MiB, 46-47%, CPU: 107-109%. No lean files = still generating.
SSH nohup background task timed out (exit 255) — this is just the SSH session disconnecting,
training process unaffected (confirmed via fresh SSH + /proc status).

Estimated: ~2-4 more hours of Stage 1 generation remaining.

### Cost at 06:16 UTC
Elapsed: 2h 3min, Cost: ~$5.52

### 06:45 UTC — Deep diagnostic: CONFIRMED HEALTHY, output buffered
Performed thorough investigation:
- Thread 1187 (main): state=R, wchan=0 — ACTIVELY RUNNING on CPU, not blocked
- 290 threads, 1 running + 49 in futex_wait_queue (PyTorch CUDA thread pool, normal)
- GPU: 41-46% varying over 10s trace — real autoregressive token generation
- Memory bandwidth: 13-15% active — data flowing between CPU and GPU
- Power: 187-191W — active computation
- Zero file I/O — expected, all generation is in-memory (GPU)
- Pipe exists (inode 253840667), Python buffer hasn't flushed to it yet

Root cause of zero visible output: Python fully buffers stdout when piped through subprocess.PIPE.
The training script prints per-theorem progress, but ~100+ print lines are sitting in Python's 8KB buffer.
No GDB/strace available on the pod to force-flush.

Estimate: ~40-50% through Stage 1 generation (10,000-15,000 of 35,600 completions).
Output will bulk-flush when either the buffer fills or Stage 1 generation finishes.
Expected first visible output: ~09:00-10:00 UTC.

### Cost at 06:45 UTC
Elapsed: 2h 32min, Cost: ~$6.81

### Lesson for next run
Launch with `PYTHONUNBUFFERED=1` or `python3 -u` to disable output buffering.

### 09:05 UTC — RELAUNCH: Stage 2 direct, PYTHONUNBUFFERED=1, 80 theorems
Killed previous run after 4h40m (18/178 theorems, buffered output).
Relaunched directly (no launcher) with:
- PYTHONUNBUFFERED=1 — real-time output confirmed working
- --skip-sft --stage 2 — straight to GRPO from base model
- --max-theorems 80 — Level 0-2 only (safe, dense gradient)
- --group-size 32
- Output piped through tee to training_live.md

Model loaded in 4s (cached). LoRA applied. Step 0 baseline collection started.
285 trajectories at ~9s each = ~35 min for step 0.
Per-trajectory output visible in real-time.

### Revised timeline
- Step 0 collection: ~35 min (09:05-09:40)
- GRPO steps 1-5: ~22 min
- Checkpoint 5 collection: ~35 min
- ... pattern continues ...
- Total: ~20 hours
- Expected completion: ~05:00 UTC Apr 6

### Cost tracking
Previous run: 4h40m × $2.69 = $12.58 (sunk)
Current run started: 09:05 UTC
Rate: $2.69/hr

### Monitoring continues...
