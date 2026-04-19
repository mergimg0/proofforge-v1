//! Proof experience buffer: the "Lean In" component.
//!
//! Stores verified proofs indexed by theorem characteristics.
//! Used for few-shot context in future proof attempts — successful
//! proofs become training signal for the next generation.
//!
//! This implements Karpathy's "system prompt learning" within the
//! SOS framework: context-space updates (adding proof examples)
//! that operate alongside weight-space updates (GRPO on strategy logits).
//!
//! The experience buffer itself is a constrained autoresearch SOS:
//! - Policy space: the set of prompt contexts (growing buffer)
//! - Evaluator: expected proof success rate with context
//! - Update: add verified proofs to buffer (append-only = monotone)
//! - Constraint: buffer size limit

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A verified proof stored in the experience buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofExample {
    /// The theorem statement
    pub statement: String,
    /// The verified proof
    pub proof: String,
    /// Difficulty tier (1-4)
    pub difficulty: u8,
    /// Which strategy produced this proof
    pub strategy: String,
    /// Training step when this proof was found
    pub found_at_step: usize,
    /// Proof length (for elegance ranking)
    pub proof_length: usize,
    /// Tags for similarity matching (extracted from statement)
    pub tags: Vec<String>,
}

/// The proof experience buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMemory {
    /// All verified proofs, keyed by theorem ID
    proofs: HashMap<String, Vec<ProofExample>>,
    /// Maximum proofs per theorem (keep shortest = most elegant)
    max_per_theorem: usize,
    /// Total proofs stored
    total: usize,
    /// Buffer version (incremented on each addition)
    pub version: u64,
}

impl ProofMemory {
    pub fn new(max_per_theorem: usize) -> Self {
        Self {
            proofs: HashMap::new(),
            max_per_theorem,
            total: 0,
            version: 0,
        }
    }

    /// Add a verified proof to the buffer.
    /// Returns true if this is a NEW proof (not a duplicate).
    ///
    /// This is the "Lean In" operation: verified outputs feed back
    /// as training context. The append-only nature guarantees
    /// monotone improvement of the buffer's information content
    /// (ECEF pattern from the SOS paper, Section 5.4).
    pub fn add(&mut self, theorem_id: &str, example: ProofExample) -> bool {
        let entries = self.proofs.entry(theorem_id.to_string()).or_default();

        // Check for duplicate proofs
        if entries.iter().any(|e| e.proof == example.proof) {
            return false;
        }

        entries.push(example);

        // Keep only the best (shortest) proofs per theorem
        entries.sort_by_key(|e| e.proof_length);
        entries.truncate(self.max_per_theorem);

        self.total = self.proofs.values().map(|v| v.len()).sum();
        self.version += 1;
        true
    }

    /// Retrieve the best proof for a theorem (if we have one).
    pub fn best_proof(&self, theorem_id: &str) -> Option<&ProofExample> {
        self.proofs.get(theorem_id).and_then(|v| v.first())
    }

    /// Retrieve few-shot examples similar to the given statement.
    /// Returns up to `k` proofs, prioritizing same-difficulty and same-tag matches.
    pub fn few_shot_examples(&self, difficulty: u8, tags: &[String], k: usize) -> Vec<&ProofExample> {
        let mut candidates: Vec<(&ProofExample, usize)> = Vec::new();

        for examples in self.proofs.values() {
            for example in examples {
                let mut score = 0;

                // Same difficulty = high relevance
                if example.difficulty == difficulty {
                    score += 10;
                }

                // Tag overlap = relevance
                for tag in tags {
                    if example.tags.contains(tag) {
                        score += 5;
                    }
                }

                // Shorter proofs are better examples
                score += 100_usize.saturating_sub(example.proof_length);

                candidates.push((example, score));
            }
        }

        // Sort by score descending, take top k
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.into_iter().take(k).map(|(e, _)| e).collect()
    }

    /// Build a few-shot prompt section from stored proofs.
    pub fn few_shot_prompt(&self, difficulty: u8, tags: &[String], k: usize) -> String {
        let examples = self.few_shot_examples(difficulty, tags, k);
        if examples.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("Here are some verified proofs for reference:\n\n");
        for (i, ex) in examples.iter().enumerate() {
            prompt.push_str(&format!(
                "Example {}:\n{}\n  {}\n\n",
                i + 1,
                ex.statement,
                ex.proof
            ));
        }
        prompt
    }

    /// Total number of unique theorems with at least one verified proof.
    pub fn theorems_solved(&self) -> usize {
        self.proofs.len()
    }

    /// Total proofs in the buffer.
    pub fn total_proofs(&self) -> usize {
        self.total
    }

    /// Get proofs per difficulty tier.
    pub fn stats_by_difficulty(&self) -> HashMap<u8, usize> {
        let mut stats = HashMap::new();
        for examples in self.proofs.values() {
            for ex in examples {
                *stats.entry(ex.difficulty).or_insert(0) += 1;
            }
        }
        stats
    }
}

impl Default for ProofMemory {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Extract simple tags from a theorem statement for similarity matching.
pub fn extract_tags(statement: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let lower = statement.to_lowercase();

    if lower.contains("nat") || lower.contains("ℕ") { tags.push("nat".into()); }
    if lower.contains("list") { tags.push("list".into()); }
    if lower.contains("prop") { tags.push("prop".into()); }
    if lower.contains("∧") || lower.contains("and") { tags.push("and".into()); }
    if lower.contains("∨") || lower.contains("or") { tags.push("or".into()); }
    if lower.contains("¬") || lower.contains("not") { tags.push("not".into()); }
    if lower.contains("→") || lower.contains("->") { tags.push("implies".into()); }
    if lower.contains("∀") || lower.contains("forall") { tags.push("forall".into()); }
    if lower.contains("∃") || lower.contains("exists") { tags.push("exists".into()); }
    if lower.contains("+") { tags.push("add".into()); }
    if lower.contains("*") || lower.contains("×") { tags.push("mul".into()); }
    if lower.contains("≤") || lower.contains("<=") { tags.push("le".into()); }
    if lower.contains("induction") || lower.contains("succ") { tags.push("induction".into()); }

    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_retrieve() {
        let mut mem = ProofMemory::new(3);
        let ex = ProofExample {
            statement: "theorem t : True := by".into(),
            proof: "trivial".into(),
            difficulty: 1,
            strategy: "Tactic".into(),
            found_at_step: 0,
            proof_length: 7,
            tags: vec!["prop".into()],
        };

        assert!(mem.add("t1", ex.clone()));
        assert!(!mem.add("t1", ex)); // duplicate
        assert_eq!(mem.theorems_solved(), 1);
        assert_eq!(mem.total_proofs(), 1);
    }

    #[test]
    fn test_keeps_shortest() {
        let mut mem = ProofMemory::new(2);

        mem.add("t1", ProofExample {
            statement: "t".into(), proof: "very long proof here".into(),
            difficulty: 1, strategy: "A".into(), found_at_step: 0,
            proof_length: 20, tags: vec![],
        });
        mem.add("t1", ProofExample {
            statement: "t".into(), proof: "short".into(),
            difficulty: 1, strategy: "B".into(), found_at_step: 1,
            proof_length: 5, tags: vec![],
        });
        mem.add("t1", ProofExample {
            statement: "t".into(), proof: "medium length".into(),
            difficulty: 1, strategy: "C".into(), found_at_step: 2,
            proof_length: 13, tags: vec![],
        });

        // Should keep 2 shortest: "short" (5) and "medium length" (13)
        let best = mem.best_proof("t1").unwrap();
        assert_eq!(best.proof, "short");
        assert_eq!(mem.total_proofs(), 2); // max_per_theorem = 2
    }

    #[test]
    fn test_few_shot_prompt() {
        let mut mem = ProofMemory::new(3);
        mem.add("t1", ProofExample {
            statement: "theorem t1 (n : Nat) : n = n := by".into(),
            proof: "rfl".into(), difficulty: 2, strategy: "Tactic".into(),
            found_at_step: 0, proof_length: 3, tags: vec!["nat".into()],
        });
        mem.add("t2", ProofExample {
            statement: "theorem t2 (n : Nat) : 0 + n = n := by".into(),
            proof: "simp".into(), difficulty: 2, strategy: "Tactic".into(),
            found_at_step: 1, proof_length: 4, tags: vec!["nat".into(), "add".into()],
        });

        let prompt = mem.few_shot_prompt(2, &["nat".into()], 2);
        assert!(prompt.contains("rfl"));
        assert!(prompt.contains("simp"));
    }

    #[test]
    fn test_extract_tags() {
        let tags = extract_tags("theorem t (a b : Nat) : a + b = b + a := by");
        assert!(tags.contains(&"nat".into()));
        assert!(tags.contains(&"add".into()));
    }
}
