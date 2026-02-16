use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

pub struct LinuxSandbox;

impl LinuxSandbox {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinuxSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for LinuxSandbox {
    fn apply_to_current_process(&self, _config: &SandboxConfig) -> Result<SandboxGuard> {
        Err(YuleError::Sandbox(
            "linux sandbox not yet implemented".into(),
        ))
    }

    fn spawn(&self, _config: &SandboxConfig) -> Result<SandboxedProcess> {
        Err(YuleError::Sandbox(
            "linux sandbox not yet implemented".into(),
        ))
    }
}
