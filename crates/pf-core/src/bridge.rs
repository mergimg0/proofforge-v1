//! Rust ↔ Python training bridge via HTTP.
//!
//! The Rust orchestrator calls the Python training server for:
//! - /generate: get proof attempts from the neural policy
//! - /reward: report binary rewards (from Lean 4 type checker)
//! - /step: execute one GRPO gradient update
//! - /status: check training state
//!
//! This decouples the orchestration (Rust: fast, deterministic)
//! from the gradient computation (Python/PyTorch: GPU-optimized).

use std::time::Duration;

use serde::Deserialize;

/// Configuration for the training server bridge.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub server_url: String,
    pub timeout_secs: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8420".to_string(),
            timeout_secs: 600, // 10 minutes — real backprop on 7B is slow
        }
    }
}

/// Response from /generate endpoint.
#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub proofs: Vec<Vec<String>>,
    pub mock: bool,
}

/// Response from /reward endpoint.
#[derive(Debug, Deserialize)]
pub struct RewardResponse {
    pub advantages: Vec<Vec<f64>>,
    pub batch_reward: f64,
}

/// Response from /step endpoint (training step metrics).
#[derive(Debug, Deserialize)]
pub struct StepResponse {
    pub step: usize,
    pub batch_reward: f64,
    pub loss: f64,
    pub grad_norm: f64,
    #[serde(default)]
    pub mean_kl: f64,
    pub n_updates: usize,
    pub policy_version: u64,
}

/// Response from /evaluate endpoint (eval set proofs for Lean verification).
#[derive(Debug, Deserialize)]
pub struct EvaluateResponse {
    pub proofs: Vec<Vec<String>>,
}

/// Response from /configure endpoint.
#[derive(Debug, Deserialize)]
pub struct ConfigureResponse {
    pub configured: bool,
    pub changes: Vec<String>,
}

/// Response from /status endpoint.
#[derive(Debug, Deserialize)]
pub struct StatusResponse {
    pub step: usize,
    pub batch_reward: f64,
    pub policy_version: u64,
    pub total_proofs_generated: usize,
    pub total_proofs_verified: usize,
    // D7 fix: fields added to Python /status but missing from Rust struct
    pub total_eval_proofs_generated: usize,
    pub batch_reward_history: Vec<f64>,
    pub model_loaded: bool,
}

/// Response from /health endpoint.
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub torch_available: bool,
    pub model_loaded: bool,
    pub device: String,
}

/// The training bridge client.
pub struct TrainingBridge {
    config: BridgeConfig,
    client: reqwest::blocking::Client,
    // D10 fix: separate short-timeout client for health checks
    health_client: reqwest::blocking::Client,
}

impl TrainingBridge {
    pub fn new(config: BridgeConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(1)
            .build()
            .expect("failed to build HTTP client");
        // D10 fix: 5s timeout for health checks (was 600s)
        let health_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(1)
            .build()
            .expect("failed to build health HTTP client");
        Self { config, client, health_client }
    }

    /// Check if the training server is healthy.
    pub fn health(&self) -> Result<HealthResponse, BridgeError> {
        let url = format!("{}/health", self.config.server_url);
        // D10 fix: use short-timeout client for health checks
        let resp = self.health_client
            .get(&url)
            .send()
            .map_err(|e| BridgeError::Connection(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(BridgeError::ServerError(format!("{status}: {body}")));
        }
        resp.json().map_err(|e| BridgeError::Parse(e.to_string()))
    }

    /// Get current training status.
    pub fn status(&self) -> Result<StatusResponse, BridgeError> {
        let url = format!("{}/status", self.config.server_url);
        self.get(&url)
    }

    /// Generate proof attempts from the neural policy.
    pub fn generate(
        &self,
        statements: &[String],
        n: usize,
    ) -> Result<GenerateResponse, BridgeError> {
        let url = format!("{}/generate", self.config.server_url);
        let body = serde_json::json!({
            "statements": statements,
            "n": n,
        });
        self.post(&url, &body)
    }

    /// Report rewards and get GRPO advantages.
    pub fn reward(
        &self,
        statements: &[String],
        proofs: &[Vec<String>],
        rewards: &[Vec<f64>],
    ) -> Result<RewardResponse, BridgeError> {
        self.reward_with_tactic_usage(statements, proofs, rewards, None)
    }

    /// Report rewards with optional tactic usage data for diversity bonus.
    pub fn reward_with_tactic_usage(
        &self,
        statements: &[String],
        proofs: &[Vec<String>],
        rewards: &[Vec<f64>],
        tactic_usage: Option<&std::collections::HashMap<String, f64>>,
    ) -> Result<RewardResponse, BridgeError> {
        let url = format!("{}/reward", self.config.server_url);
        let mut body = serde_json::json!({
            "statements": statements,
            "proofs": proofs,
            "rewards": rewards,
        });
        if let Some(usage) = tactic_usage {
            body["tactic_usage"] = serde_json::json!(usage);
        }
        self.post(&url, &body)
    }

    /// Execute one GRPO training step.
    pub fn step(
        &self,
        statements: &[String],
        proofs: &[Vec<String>],
        rewards: &[Vec<f64>],
        advantages: &[Vec<f64>],
    ) -> Result<StepResponse, BridgeError> {
        let url = format!("{}/step", self.config.server_url);
        let body = serde_json::json!({
            "statements": statements,
            "proofs": proofs,
            "rewards": rewards,
            "advantages": advantages,
        });
        self.post(&url, &body)
    }

    /// Reconfigure training parameters mid-run (e.g., reduce LR on SOS violation).
    pub fn configure(&self, params: &serde_json::Value) -> Result<ConfigureResponse, BridgeError> {
        let url = format!("{}/configure", self.config.server_url);
        self.post(&url, params)
    }

    /// Generate proofs for evaluation (Rust side verifies with Lean).
    pub fn evaluate(
        &self,
        statements: &[String],
        n: usize,
    ) -> Result<EvaluateResponse, BridgeError> {
        let url = format!("{}/evaluate", self.config.server_url);
        let body = serde_json::json!({
            "statements": statements,
            "n": n,
        });
        self.post(&url, &body)
    }

    fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, BridgeError> {
        let resp = self.client
            .get(url)
            .send()
            .map_err(|e| BridgeError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(BridgeError::ServerError(format!("{status}: {body}")));
        }

        resp.json().map_err(|e| BridgeError::Parse(e.to_string()))
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<T, BridgeError> {
        let resp = self.client
            .post(url)
            .json(body)
            .send()
            .map_err(|e| BridgeError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(BridgeError::ServerError(format!("{status}: {body}")));
        }

        resp.json().map_err(|e| BridgeError::Parse(e.to_string()))
    }
}

/// Bridge errors.
#[derive(Debug)]
pub enum BridgeError {
    Connection(String),
    Parse(String),
    ServerError(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Connection(e) => write!(f, "Connection error: {e}"),
            BridgeError::Parse(e) => write!(f, "Parse error: {e}"),
            BridgeError::ServerError(e) => write!(f, "Server error: {e}"),
        }
    }
}
