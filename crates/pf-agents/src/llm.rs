//! LLM-based proof agent: generates proofs via Ollama or compatible HTTP API.

use pf_core::types::{AgentId, ProofAttempt};
use crate::ProofAgent;

/// An LLM proof agent that calls an Ollama-compatible API.
pub struct LLMAgent {
    pub agent_id: AgentId,
    pub api_url: String,
    pub model: String,
    pub system_prompt: String,
}

impl LLMAgent {
    pub fn new(agent_id: AgentId, api_url: &str, model: &str) -> Self {
        Self {
            agent_id,
            api_url: api_url.to_string(),
            model: model.to_string(),
            system_prompt: LEAN4_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Call the Ollama API synchronously.
    fn call_api(&self, prompt: &str) -> Option<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": false,
            "options": {
                "temperature": 0.7,
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

impl ProofAgent for LLMAgent {
    fn id(&self) -> AgentId {
        self.agent_id
    }

    fn generate(&self, statement: &str, count: usize, policy_version: u64) -> Vec<ProofAttempt> {
        let prompt = format!(
            "Complete the following Lean 4 proof. Output ONLY the proof tactic(s), nothing else.\n\n{}\n",
            statement
        );

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

const LEAN4_SYSTEM_PROMPT: &str = r#"You are a Lean 4 proof assistant. Given a theorem statement ending with `:= by`, output ONLY the tactic proof body. No explanation, no imports, no theorem statement — just the tactics.

Examples:
Input: theorem t : True := by
Output: trivial

Input: theorem t (n : Nat) : 0 + n = n := by
Output: simp

Input: theorem t (p q : Prop) : p → q → p ∧ q := by
Output: intro hp hq; exact ⟨hp, hq⟩

Input: theorem t (a b : Nat) : a + b = b + a := by
Output: omega"#;
