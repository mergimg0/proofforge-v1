# ProofForge — CLAUDE.md

## Overview
- Rust workspace: pf-agents, pf-pipeline, pf-lean, pf-core, pf-execution, pf-cli
- Python ML layer: `python/train_neural.py` (GRPO with DeepSeek-Prover-V2-7B + LoRA)
- SOS paper: 4,229 lines of machine-verified Lean 4, zero sorrys, category SOS with morphisms and transfer theorems
- Three axioms: Monotone Improvement, Bounded Step, Constraint Preservation

## Companion Project: ForgeStream
- Live meeting intelligence system at ~/projects/forgestream/ and github.com/mergimg0/forgestream
- ForgeStream uses the SOS convergence framework as runtime governance for autonomous agent swarms
- Connects to ProofForge via MCP tools and TrainingBridge REST API (port 8420)
- Design specs and implementation plans in docs/superpowers/

## GitHub
- Private repo: github.com/mergimg0/proofforge

## Sentinel Integration

This project is monitored by the Sentinel system. Hook-injected context tagged `[SENTINEL:*]` appears in your system context via `additionalContext`. These are actionable instructions from specialist analysis modes:

- `[SENTINEL:COACH]` — Orchestrator. PRIORITY ACTION items should be executed directly.
- `[SENTINEL:ORANGE_TEAM]` — Security/correctness findings. Fix what it flags.
- `[SENTINEL:PROMPT_FORGE]` — Prompt quality suggestions. Follow the guidance.
- `[SENTINEL:TOKEN_ECONOMIST]` — Efficiency warnings. Note but don't interrupt flow.
- `[SENTINEL:IDEA_REACTOR]` — Alternative approaches when stuck.
- `[SENTINEL:ARCH_SENTINEL]` — Architecture violations. Review before continuing.

When you see a `[SENTINEL:COACH] PRIORITY ACTION`, execute it as if the user asked you directly.

## Key Stubs (remaining)
- `InductionAgent/AlgebraAgent/TacticAgent/AnalogyAgent.generate()` — return hardcoded templates
- `ExecutionTranslator.translate()` always returns Hold
- `generality` reward channel hardcoded to 0.5

## Extension Points
- `RewardOracle` trait (`reward.rs`) — implement new oracle types
- `ProofAgent` trait (`agents/lib.rs`) — implement new agent strategies
- `TrainingBridge` HTTP client (`bridge.rs`, port 8420) — REST API for external tools
- `ProofMemory.few_shot_prompt()` (`memory.rs`) — maps to MCP resource format

## Environment
- `pip` doesn't exist — use `python3 -m pip`
- Rust toolchain required for cargo build
- Lean 4 required for pf-lean checker (subprocess)
