use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

use std::ffi::{CStr, CString, c_char};
use std::ptr;

// ---------------------------------------------------------------------------
// FFI: macOS Seatbelt (sandbox_init / sandbox_free_error)
// ---------------------------------------------------------------------------

extern "C" {
    fn sandbox_init(profile: *const c_char, flags: u64, errorbuf: *mut *mut c_char) -> i32;
    fn sandbox_free_error(errorbuf: *mut c_char);
}

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
    fn apply_to_current_process(&self, config: &SandboxConfig) -> Result<SandboxGuard> {
        // 1. memory limit
        apply_rlimit(config.max_memory_bytes)?;

        // 2. seatbelt profile (permanent, no undo)
        let profile = build_seatbelt_profile(config);
        apply_seatbelt(&profile)?;

        tracing::info!(memory_limit = config.max_memory_bytes, "sandbox applied");

        Ok(SandboxGuard { _marker: () })
    }

    fn spawn(&self, _config: &SandboxConfig) -> Result<SandboxedProcess> {
        Err(YuleError::Sandbox(
            "macos sandbox spawn not yet implemented".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Layer 1: memory limit via setrlimit
// ---------------------------------------------------------------------------

fn apply_rlimit(max_memory_bytes: u64) -> Result<()> {
    let limit = libc::rlimit {
        rlim_cur: max_memory_bytes as libc::rlim_t,
        rlim_max: max_memory_bytes as libc::rlim_t,
    };
    let ret = unsafe { libc::setrlimit(libc::RLIMIT_AS, &limit) };
    if ret != 0 {
        return Err(YuleError::Sandbox(format!(
            "setrlimit(RLIMIT_AS) failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    tracing::info!(max_memory_bytes, "rlimit: RLIMIT_AS set");
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 2: Seatbelt profile (SBPL)
// ---------------------------------------------------------------------------

fn build_seatbelt_profile(config: &SandboxConfig) -> String {
    let model_path = config.model_path.to_string_lossy();

    let mut profile = String::with_capacity(2048);

    // header: deny everything by default
    profile.push_str("(version 1)\n");
    profile.push_str("(deny default)\n\n");

    // model file — read only
    profile.push_str(&format!(
        "(allow file-read* (literal \"{model_path}\"))\n\n"
    ));

    // system libraries and frameworks (required for dyld, Metal, etc.)
    profile.push_str("(allow file-read* (subpath \"/usr/lib\"))\n");
    profile.push_str("(allow file-read* (subpath \"/System/Library\"))\n");
    profile.push_str("(allow file-read* (subpath \"/Library/Apple\"))\n");
    profile.push_str("(allow file-read* (subpath \"/usr/share\"))\n");
    profile.push_str("(allow file-read* (subpath \"/private/var/db/dyld\"))\n");
    profile.push_str("(allow file-read* (literal \"/dev/urandom\"))\n");
    profile.push_str("(allow file-read* (literal \"/dev/random\"))\n\n");

    // process basics
    profile.push_str("(allow process-exec*)\n");
    profile.push_str("(allow process-fork)\n");
    profile.push_str("(allow sysctl-read)\n");
    profile.push_str("(allow mach-lookup)\n");
    profile.push_str("(allow signal (target self))\n\n");

    // GPU access (Metal / I/O Kit)
    if config.allow_gpu {
        profile.push_str("(allow iokit-open)\n");
        profile.push_str("(allow file-read* (subpath \"/Library/GPUBundles\"))\n");
        profile.push_str("(allow file-read* (subpath \"/System/Library/Extensions\"))\n\n");
    }

    // networking
    if config.allow_network {
        profile.push_str("(allow network-outbound)\n");
        profile.push_str("(allow network-inbound)\n");
        profile.push_str("(allow network-bind)\n");
        profile.push_str("(allow system-socket)\n\n");
    }

    profile
}

fn apply_seatbelt(profile: &str) -> Result<()> {
    let c_profile = CString::new(profile)
        .map_err(|e| YuleError::Sandbox(format!("seatbelt profile contains null byte: {e}")))?;

    let mut errorbuf: *mut c_char = ptr::null_mut();

    // flags = 0: raw profile string (not a named profile)
    let ret = unsafe { sandbox_init(c_profile.as_ptr(), 0, &mut errorbuf) };

    if ret != 0 {
        let msg = if !errorbuf.is_null() {
            let err = unsafe { CStr::from_ptr(errorbuf) }
                .to_string_lossy()
                .into_owned();
            unsafe { sandbox_free_error(errorbuf) };
            err
        } else {
            "unknown error".to_string()
        };
        return Err(YuleError::Sandbox(format!("sandbox_init failed: {msg}")));
    }

    tracing::info!("seatbelt: profile applied");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(allow_gpu: bool, allow_network: bool) -> SandboxConfig {
        SandboxConfig {
            model_path: "/tmp/test-model.gguf".into(),
            allow_gpu,
            max_memory_bytes: 32 * 1024 * 1024 * 1024,
            allow_network,
        }
    }

    #[test]
    fn seatbelt_profile_contains_model_path() {
        let config = test_config(false, false);
        let profile = build_seatbelt_profile(&config);
        assert!(profile.contains("/tmp/test-model.gguf"));
        assert!(profile.contains("(deny default)"));
        assert!(profile.contains("(version 1)"));
    }

    #[test]
    fn seatbelt_profile_default_denies_network() {
        let config = test_config(false, false);
        let profile = build_seatbelt_profile(&config);
        assert!(!profile.contains("network-outbound"));
        assert!(!profile.contains("network-inbound"));
        assert!(!profile.contains("network-bind"));
    }

    #[test]
    fn seatbelt_profile_with_gpu() {
        let config = test_config(true, false);
        let profile = build_seatbelt_profile(&config);
        assert!(profile.contains("iokit-open"));
        assert!(profile.contains("GPUBundles"));
    }

    #[test]
    fn seatbelt_profile_with_network() {
        let config = test_config(false, true);
        let profile = build_seatbelt_profile(&config);
        assert!(profile.contains("network-outbound"));
        assert!(profile.contains("network-inbound"));
        assert!(profile.contains("network-bind"));
        assert!(profile.contains("system-socket"));
    }
}
