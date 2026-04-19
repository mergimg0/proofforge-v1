//! App 6: Trajectory length as training signal — efficiency reward types.
//!
//! Extends the binary reward oracle with efficiency-weighted rewards
//! that restore gradient signal on mastered theorems by differentiating
//! short (good) from long (okay) proofs.
//!
//! SOS preservation: the efficiency factor is a monotone function of
//! proof quality (shorter correct proofs always score higher). Incorrect
//! proofs still get 0.0. Therefore E(δ(π)) ≥ E(π) is preserved.

use serde::{Deserialize, Serialize};

/// Configuration for efficiency-weighted reward.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EfficiencyConfig {
    /// α for fully mastered theorems (breadth ≥ 3)
    pub alpha_mastered: f64,
    /// α for moderately mastered theorems (breadth ≥ 2)
    pub alpha_moderate: f64,
    /// α for fragile theorems (breadth < 2) — should be 0
    pub alpha_fragile: f64,
    /// Maximum proof length in tokens for normalization
    pub max_proof_tokens: usize,
    /// Training step at which to switch from binary to efficiency
    pub phase_in_step: usize,
    /// Minimum pass rate before switching to efficiency
    pub phase_in_pass_rate: f64,
    /// Floor: minimum reward for correct proof regardless of length
    pub length_floor: f64,
}

impl Default for EfficiencyConfig {
    fn default() -> Self {
        Self {
            alpha_mastered: 0.5,
            alpha_moderate: 0.2,
            alpha_fragile: 0.0,
            max_proof_tokens: 512,
            phase_in_step: 50,
            phase_in_pass_rate: 0.20,
            length_floor: 0.3,
        }
    }
}

/// Temperature breadth for a single theorem.
/// Measures at how many temperatures the model solves it reliably.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremBreadth {
    /// Theorem identifier
    pub theorem_id: String,
    /// Number of temperatures where pass rate exceeds threshold
    pub breadth: u32,
    /// Per-temperature pass rates
    pub temperature_rates: Vec<(f64, f64)>,
}

/// Compute adaptive α for a theorem based on its temperature breadth.
///
/// breadth ≥ 3 → alpha_mastered (strong efficiency pressure)
/// breadth ≥ 2 → alpha_moderate (moderate pressure)
/// breadth < 2 → alpha_fragile (no pressure, pure binary)
pub fn adaptive_alpha(breadth: u32, config: &EfficiencyConfig) -> f64 {
    if breadth >= 3 {
        config.alpha_mastered
    } else if breadth >= 2 {
        config.alpha_moderate
    } else {
        config.alpha_fragile
    }
}

/// Compute efficiency-weighted reward.
///
/// reward = verified * max(floor, 1.0 - α * length / max_length)
///
/// For unverified proofs: reward = 0.0
/// For verified proofs: reward ∈ [length_floor, 1.0]
pub fn efficiency_reward(
    verified: bool,
    proof_tokens: usize,
    alpha: f64,
    config: &EfficiencyConfig,
) -> f64 {
    if !verified {
        return 0.0;
    }
    if alpha == 0.0 {
        return 1.0;
    }
    let length_ratio = (proof_tokens as f64 / config.max_proof_tokens as f64).min(1.0);
    let raw = 1.0 - alpha * length_ratio;
    raw.max(config.length_floor)
}

/// Whether efficiency reward should be active at this training state.
pub fn efficiency_active(step: usize, pass_rate: f64, config: &EfficiencyConfig) -> bool {
    step >= config.phase_in_step && pass_rate >= config.phase_in_pass_rate
}

/// Trajectory length metrics for monitoring App 6 effectiveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMetrics {
    pub step: usize,
    pub mean_proof_length: f64,
    pub median_proof_length: f64,
    /// Low-T / High-T proof length ratio (should approach 1.0)
    pub trajectory_length_ratio: f64,
    pub pass_rate: f64,
    pub theorems_with_efficiency: usize,
    pub theorems_binary_only: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_alpha() {
        let config = EfficiencyConfig::default();
        assert_eq!(adaptive_alpha(0, &config), 0.0);
        assert_eq!(adaptive_alpha(1, &config), 0.0);
        assert_eq!(adaptive_alpha(2, &config), 0.2);
        assert_eq!(adaptive_alpha(3, &config), 0.5);
        assert_eq!(adaptive_alpha(5, &config), 0.5);
    }

    #[test]
    fn test_efficiency_reward_unverified() {
        let config = EfficiencyConfig::default();
        assert_eq!(efficiency_reward(false, 100, 0.5, &config), 0.0);
    }

    #[test]
    fn test_efficiency_reward_binary_mode() {
        let config = EfficiencyConfig::default();
        // alpha = 0 → pure binary (1.0 for verified)
        assert_eq!(efficiency_reward(true, 256, 0.0, &config), 1.0);
    }

    #[test]
    fn test_efficiency_reward_short_proof() {
        let config = EfficiencyConfig::default();
        // 10 tokens out of 512 max, alpha = 0.5
        let r = efficiency_reward(true, 10, 0.5, &config);
        assert!(r > 0.98, "short proof should get near-perfect reward: {r}");
    }

    #[test]
    fn test_efficiency_reward_long_proof() {
        let config = EfficiencyConfig::default();
        // 512 tokens (max), alpha = 0.5
        let r = efficiency_reward(true, 512, 0.5, &config);
        assert_eq!(r, 0.5, "max-length proof with α=0.5 should get 0.5");
    }

    #[test]
    fn test_efficiency_reward_floor() {
        let config = EfficiencyConfig::default();
        // α=0.9 with max-length proof: 1.0 - 0.9*1.0 = 0.1, below floor of 0.3
        let r = efficiency_reward(true, 512, 0.9, &config);
        assert_eq!(r, config.length_floor, "should be clamped to floor");
    }

    #[test]
    fn test_efficiency_active() {
        let config = EfficiencyConfig::default();
        assert!(!efficiency_active(10, 0.5, &config)); // too early
        assert!(!efficiency_active(60, 0.1, &config)); // pass rate too low
        assert!(efficiency_active(60, 0.3, &config));  // both thresholds met
    }
}
