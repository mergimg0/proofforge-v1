//! Lean 4 type checker subprocess wrapper.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use thiserror::Error;
use wait_timeout::ChildExt;

#[derive(Error, Debug)]
pub enum LeanCheckerError {
    #[error("Lean binary not found at {0}")]
    BinaryNotFound(PathBuf),
    #[error("Type check failed: {0}")]
    TypeCheckFailed(String),
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result of checking a Lean 4 source file.
#[derive(Debug)]
pub struct CheckResult {
    pub success: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub duration: Duration,
}

/// Configuration for the Lean 4 checker.
#[derive(Debug, Clone)]
pub struct LeanCheckerConfig {
    /// Path to the Lean 4 binary
    pub lean_path: PathBuf,
    /// Path to the project root (for lake environment)
    pub project_path: PathBuf,
    /// Maximum time per proof check
    pub timeout: Duration,
    /// Additional LEAN_PATH entries (for Mathlib oleans)
    pub extra_lean_paths: Vec<PathBuf>,
}

impl Default for LeanCheckerConfig {
    fn default() -> Self {
        Self {
            lean_path: PathBuf::from("lean"),
            project_path: PathBuf::from("."),
            timeout: Duration::from_secs(60),
            extra_lean_paths: vec![],
        }
    }
}

/// The Lean 4 type checker. Spawns lean as a subprocess to verify proofs.
pub struct LeanChecker {
    config: LeanCheckerConfig,
}

impl LeanChecker {
    pub fn new(config: LeanCheckerConfig) -> Self {
        Self { config }
    }

    /// Check a complete Lean 4 source string.
    ///
    /// Writes the source to a temporary file, runs `lean` on it,
    /// and parses the output for errors. Thread-safe via unique temp files.
    pub fn check_source(&self, source: &str) -> Result<CheckResult, LeanCheckerError> {
        // Thread-safe temp file via tempfile crate
        let mut tmp = tempfile::Builder::new()
            .prefix("pf_check_")
            .suffix(".lean")
            .tempfile()?;
        tmp.write_all(source.as_bytes())?;
        let tmp_path = tmp.path().to_path_buf();

        let result = self.run_lean(&tmp_path);
        // tmp drops here, auto-deleting the file

        result
    }

    /// Check a proof for a given theorem statement.
    ///
    /// Constructs a complete Lean 4 file with imports, the statement,
    /// and the proposed proof, then type-checks it.
    pub fn check_proof(
        &self,
        statement: &str,
        proof: &str,
        imports: &[&str],
    ) -> Result<CheckResult, LeanCheckerError> {
        let mut source = String::new();

        // Add imports
        for import in imports {
            source.push_str(&format!("import {import}\n"));
        }
        source.push('\n');

        // Add the theorem with proof
        source.push_str(statement);
        if !statement.ends_with('\n') {
            source.push('\n');
        }
        source.push_str(proof);
        source.push('\n');

        self.check_source(&source)
    }

    fn run_lean(&self, file: &Path) -> Result<CheckResult, LeanCheckerError> {
        let start = Instant::now();

        let mut cmd = Command::new(&self.config.lean_path);
        cmd.arg(file);

        // Set LEAN_PATH if we have extra paths
        if !self.config.extra_lean_paths.is_empty() {
            let paths: Vec<String> = self
                .config
                .extra_lean_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            cmd.env("LEAN_PATH", paths.join(":"));
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn()?;
        let timeout = self.config.timeout;

        // Real timeout: kill process if it exceeds the deadline
        let status = match child.wait_timeout(timeout) {
            Ok(Some(status)) => status,
            Ok(None) => {
                // Timed out — kill the child
                let _ = child.kill();
                let _ = child.wait();
                return Err(LeanCheckerError::Timeout(timeout));
            }
            Err(e) => return Err(LeanCheckerError::Io(e)),
        };

        let duration = start.elapsed();

        let mut stdout_buf = String::new();
        let mut stderr_buf = String::new();
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout.read_to_string(&mut stdout_buf);
        }
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut stderr_buf);
        }

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in stderr_buf.lines().chain(stdout_buf.lines()) {
            if line.contains("error:") {
                errors.push(line.to_string());
            } else if line.contains("warning:") {
                warnings.push(line.to_string());
            }
        }

        Ok(CheckResult {
            success: status.success() && errors.is_empty(),
            errors,
            warnings,
            duration,
        })
    }
}
