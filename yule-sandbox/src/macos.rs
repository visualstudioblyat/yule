use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

pub struct MacOsSandbox;

impl MacOsSandbox {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MacOsSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for MacOsSandbox {
    fn apply_to_current_process(&self, _config: &SandboxConfig) -> Result<SandboxGuard> {
        Err(YuleError::Sandbox(
            "macos sandbox not yet implemented".into(),
        ))
    }

    fn spawn(&self, _config: &SandboxConfig) -> Result<SandboxedProcess> {
        Err(YuleError::Sandbox(
            "macos sandbox not yet implemented".into(),
        ))
    }
}
