//! Adaptive proof agent: combines LLM generation with proof memory.
//!
//! This is the "system prompt learning" component: instead of (or alongside)
//! weight updates, the agent adapts its behavior by incorporating successful
//! proofs into its context. Verified proofs become few-shot examples that
//! guide future proof generation.
//!
//! The adaptation loop:
//! 1. Retrieve similar verified proofs from memory (Lean In)
//! 2. Build few-shot prompt with those examples
//! 3. Generate proof attempts with the enriched context
//! 4. Type-check results (binary oracle)
//! 5. Store verified proofs back into memory (Lean Out → Lean In)
//!
//! This is a concrete autoresearch SOS on the space of prompt contexts:
//! - Each context update (adding a proof example) is monotone (more info = better)
//! - The step is bounded (adding finite tokens)
//! - The constraint (valid prompt, within token budget) is preserved

use pf_core::memory::ProofMemory;
use pf_core::types::{AgentId, ProofAttempt};
use crate::ProofAgent;

/// An adaptive LLM agent that uses proof memory for few-shot prompting.
pub struct AdaptiveAgent {
    pub agent_id: AgentId,
    pub api_url: String,
    pub model: String,
    /// Number of few-shot examples to include
    pub num_examples: usize,
    /// Shared reference to proof memory (read-only during generation)
    memory_snapshot: Option<ProofMemorySnapshot>,
}

/// A frozen snapshot of proof memory for use during generation.
#[derive(Debug, Clone)]
pub struct ProofMemorySnapshot {
    pub few_shot_prompts: std::collections::HashMap<u8, String>,
    pub version: u64,
}

impl ProofMemorySnapshot {
    /// Create a snapshot from current proof memory.
    pub fn from_memory(memory: &ProofMemory) -> Self {
        let mut prompts = std::collections::HashMap::new();
        for difficulty in 1..=4 {
            let prompt = memory.few_shot_prompt(difficulty, &[], 4);
            if !prompt.is_empty() {
                prompts.insert(difficulty, prompt);
            }
        }
        Self {
            few_shot_prompts: prompts,
            version: memory.version,
        }
    }
}

impl AdaptiveAgent {
    pub fn new(agent_id: AgentId, api_url: &str, model: &str) -> Self {
        Self {
            agent_id,
            api_url: api_url.to_string(),
            model: model.to_string(),
            num_examples: 4,
            memory_snapshot: None,
        }
    }

    /// Update the agent's memory snapshot (call before each generation round).
    pub fn update_memory(&mut self, memory: &ProofMemory) {
        self.memory_snapshot = Some(ProofMemorySnapshot::from_memory(memory));
    }

    fn build_prompt(&self, statement: &str, difficulty: u8) -> String {
        let mut prompt = String::new();

        // Add few-shot examples from memory if available
        if let Some(snapshot) = &self.memory_snapshot {
            if let Some(examples) = snapshot.few_shot_prompts.get(&difficulty) {
                prompt.push_str(examples);
            }
        }

        prompt.push_str(&format!(
            "Now prove the following. Output ONLY the proof tactic(s), nothing else.\n\n{}\n",
            statement
        ));
        prompt
    }

    fn call_api(&self, prompt: &str) -> Option<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let system = format!(
            "{}\n\nYou have access to verified proof examples. Use similar strategies.",
            ADAPTIVE_SYSTEM_PROMPT
        );

        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": false,
            "options": {
                "temperature": 0.6,
                "num_predict": 256,
                "stop": ["\n\n", "theorem", "-- END"]
            }
        });

        let resp = client
            .post(&format!("{}/api/generate", self.api_url))
            .json(&body)
            .send()
            .ok()?;

        let json: serde_json::Value = resp.json().ok()?;
        json["response"].as_str().map(|s| s.trim().to_string())
    }
}

impl ProofAgent for AdaptiveAgent {
    fn id(&self) -> AgentId {
        self.agent_id
    }

    fn as_adaptive_mut(&mut self) -> Option<&mut AdaptiveAgent> {
        Some(self)
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        // Estimate difficulty from statement length/complexity
        let difficulty = estimate_difficulty(statement);
        let prompt = self.build_prompt(statement, difficulty);

        (0..count)
            .map(|_| {
                let proof = self
                    .call_api(&prompt)
                    .unwrap_or_else(|| "sorry".to_string());

                ProofAttempt {
                    statement: statement.to_string(),
                    proof,
                    agent_id: self.agent_id,
                    policy_version,
                }
            })
            .collect()
    }
}

/// Estimate difficulty from statement complexity.
fn estimate_difficulty(statement: &str) -> u8 {
    let len = statement.len();
    let has_forall = statement.contains('∀') || statement.contains("forall");
    let has_induction_hint = statement.contains("Nat.succ") || statement.contains("List");
    let has_logic = statement.contains('∧') || statement.contains('∨') || statement.contains('¬');

    if len < 40 && !has_forall && !has_logic {
        1
    } else if has_induction_hint {
        3
    } else if has_forall || has_logic {
        if statement.matches('→').count() > 1 || statement.contains("∧") && statement.contains("∨") {
            4
        } else {
            2
        }
    } else {
        2
    }
}

const ADAPTIVE_SYSTEM_PROMPT: &str = r#"You are a Lean 4 proof assistant. Given a theorem statement ending with `:= by`, output ONLY the tactic proof body. No explanation, no imports, no theorem statement — just the tactics.

Use the verified examples above as guidance for proof strategy. Prefer shorter proofs."#;
