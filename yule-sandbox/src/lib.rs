pub mod policy;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

use yule_core::error::Result;

pub trait Sandbox: Send {
    fn apply_to_current_process(&self, config: &SandboxConfig) -> Result<SandboxGuard>;
    fn spawn(&self, config: &SandboxConfig) -> Result<SandboxedProcess>;
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub model_path: std::path::PathBuf,
    pub allow_gpu: bool,
    pub max_memory_bytes: u64,
    pub allow_network: bool,
}

pub struct SandboxedProcess {
    pub pid: u32,
    pub stdin: std::process::ChildStdin,
    pub stdout: std::process::ChildStdout,
}

pub struct SandboxGuard {
    #[cfg(target_os = "windows")]
    pub(crate) job_handle: *mut std::ffi::c_void,
    #[cfg(not(target_os = "windows"))]
    _marker: (),
}

// job handle is just a kernel handle, safe to send across threads
unsafe impl Send for SandboxGuard {}

impl Drop for SandboxGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            if !self.job_handle.is_null() {
                unsafe {
                    windows::close_job_handle(self.job_handle);
                }
            }
        }
    }
}

pub fn create_sandbox() -> Box<dyn Sandbox> {
    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LinuxSandbox::new())
    }

    #[cfg(target_os = "windows")]
    {
        Box::new(windows::WindowsSandbox::new())
    }

    #[cfg(target_os = "macos")]
    {
        Box::new(macos::MacOsSandbox::new())
    }
}
