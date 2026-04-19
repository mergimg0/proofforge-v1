//! Theorem dataset loading and management.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremDataset {
    pub description: String,
    pub version: String,
    pub tiers: Vec<Tier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier {
    pub name: String,
    pub description: String,
    pub theorems: Vec<Theorem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    pub id: String,
    pub statement: String,
    pub known_proof: String,
    pub difficulty: u8,
}

impl TheoremDataset {
    /// Load dataset from a JSON file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let dataset: Self = serde_json::from_str(&contents)?;
        Ok(dataset)
    }

    /// Get all theorems across all tiers.
    pub fn all_theorems(&self) -> Vec<&Theorem> {
        self.tiers.iter().flat_map(|t| t.theorems.iter()).collect()
    }

    /// Get theorems for a specific difficulty level.
    pub fn theorems_by_difficulty(&self, difficulty: u8) -> Vec<&Theorem> {
        self.all_theorems()
            .into_iter()
            .filter(|t| t.difficulty == difficulty)
            .collect()
    }

    /// Total number of theorems.
    pub fn count(&self) -> usize {
        self.tiers.iter().map(|t| t.theorems.len()).sum()
    }
}
