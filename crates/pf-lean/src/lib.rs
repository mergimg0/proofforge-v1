//! # pf-lean: Lean 4 type checker integration
//!
//! Wraps the Lean 4 compiler as a subprocess to provide the binary reward
//! oracle for ProofForge. Same architectural pattern as DEM's sem.ts
//! wrapping the Sem binary via child_process.execFile.
//!
//! The type checker is the GROUND TRUTH of the reward signal:
//! - Proof type-checks → reward = 1.0
//! - Proof fails → reward = 0.0
//! - No approximation. No hallucination. No learned value function.

pub mod checker;
pub mod oracle;
