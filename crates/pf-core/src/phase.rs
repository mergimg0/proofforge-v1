//! App 8: Training phase detection types for the adaptive controller.
//!
//! Defines the five training phases and phase transition types that
//! the adaptive controller monitors. These types mirror the Python
//! controller implementation for cross-language consistency.

use serde::{Deserialize, Serialize};

/// The six phases of GRPO training with binary verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingPhase {
    /// GRPO reshaping distribution. Pass rate may drop. Expected.
    Disruption,
    /// Infrastructure building. Reward climbing, pass rate flat.
    Accumulation,
    /// Infrastructure critical mass. Pass rate rising rapidly. OPTIMAL.
    Breakout,
    /// Theorems stop flickering and become reliably solved.
    /// Retention jumps from ~9% to ~95%. The paper's sharpest finding.
    Consolidation,
    /// Most theorems solved. Gradient signal collapsing.
    Saturation,
    /// Stuck. Reward flat below useful threshold.
    Stagnation,
}

impl TrainingPhase {
    /// Human-readable phase name.
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingPhase::Disruption => "disruption",
            TrainingPhase::Accumulation => "accumulation",
            TrainingPhase::Breakout => "breakout",
            TrainingPhase::Consolidation => "consolidation",
            TrainingPhase::Saturation => "saturation",
            TrainingPhase::Stagnation => "stagnation",
        }
    }
}

/// A detected phase transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub from_phase: TrainingPhase,
    pub to_phase: TrainingPhase,
    pub step: usize,
    pub trigger: String,
    pub reward_mean: f64,
    pub pass_rate: f64,
}

/// Intervention types that the controller can recommend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterventionType {
    /// Do nothing. Log the phase.
    Wait,
    /// Record progress. Estimate breakout ETA.
    Log,
    /// Optimal zone. Don't touch anything.
    NoAction,
    /// Add harder theorems / expand dataset.
    ExpandDataset,
    /// Switch to efficiency reward (App 6).
    SwitchEfficiency,
    /// Increase group_size for richer advantage signal.
    IncreaseGroupSize,
    /// Reduce learning rate.
    ReduceLearningRate,
    /// Enable shaped reward warmup.
    EnableShapedWarmup,
    /// Save checkpoint.
    Checkpoint,
}

/// Configuration for phase detection thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseDetectorConfig {
    /// Reward mean above this + positive trend = disruption → accumulation
    pub accumulation_threshold: f64,
    /// Reward mean above this = accumulation → breakout
    pub breakout_threshold: f64,
    /// Reward mean above this consistently = breakout → saturation
    pub saturation_threshold: f64,
    /// Reward std below this (with high mean) confirms saturation
    pub saturation_std: f64,
    /// Reward mean below this for stagnation_patience steps = stagnation
    pub stagnation_ceiling: f64,
    /// Steps of low reward before declaring stagnation
    pub stagnation_patience: usize,
}

impl Default for PhaseDetectorConfig {
    fn default() -> Self {
        Self {
            accumulation_threshold: 0.10,
            breakout_threshold: 0.25,
            saturation_threshold: 0.80,
            saturation_std: 0.10,
            stagnation_ceiling: 0.15,
            stagnation_patience: 50,
        }
    }
}

/// Rolling metric snapshot for phase detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    pub step: usize,
    pub reward_mean: f64,
    pub reward_trend: f64,
    pub reward_std: f64,
    pub pass_rate_mean: f64,
    pub pass_rate_trend: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_names() {
        assert_eq!(TrainingPhase::Disruption.as_str(), "disruption");
        assert_eq!(TrainingPhase::Breakout.as_str(), "breakout");
        assert_eq!(TrainingPhase::Consolidation.as_str(), "consolidation");
        assert_eq!(TrainingPhase::Saturation.as_str(), "saturation");
    }

    #[test]
    fn test_default_config() {
        let config = PhaseDetectorConfig::default();
        assert!(config.accumulation_threshold < config.breakout_threshold);
        assert!(config.breakout_threshold < config.saturation_threshold);
    }
}
