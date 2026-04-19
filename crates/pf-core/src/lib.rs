//! # pf-core: Core types and GRPO algorithm for ProofForge
//!
//! This crate defines the fundamental types aligned with the Sacred Object System
//! formalization (Gashi, 2026). The SOS structure is:
//! - Policy space (Π): parameterised proof-generation policies
//! - Evaluator (E): expected proof success rate
//! - Update operator (δ): GRPO step with binary reward oracle
//! - Constraint (C): staleness, M2PO, domain-specific
//!
//! The Lean 4 formalization proves convergence as a free theorem.
//! This Rust implementation is the engineering realization.

pub mod types;
pub mod grpo;
pub mod reward;
pub mod dataset;
pub mod evaluator;
pub mod policy;
pub mod memory;
pub mod bridge;

// Application modules (Priority 3)
pub mod efficiency;  // App 6: Trajectory length as training signal
pub mod phase;       // App 8: Adaptive training controller types
pub mod code_task;   // App 1: Self-bootstrapping code verification
