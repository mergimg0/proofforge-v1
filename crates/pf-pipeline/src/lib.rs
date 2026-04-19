//! # pf-pipeline: AReaL-style async orchestration
//!
//! Decouples GPU-bound proof generation from CPU-bound Lean 4 type checking.
//! Models AReaL's doubly constraint-lifted SOS:
//!   S_async = constraintLift(constraintLift(S_GRPO, C_stale), C_m2po)
//!
//! Convergence is a free theorem from the SOS framework (proven in AReaL.lean).
//! The 2.77x speedup from AReaL applies directly: proof generation and
//! type checking run on separate worker pools.

pub mod scheduler;
