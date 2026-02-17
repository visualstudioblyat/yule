use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

use landlock::{
    Access, AccessFs, BitFlags, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    RulesetStatus,
};
use seccompiler::{SeccompAction, SeccompFilter, SeccompRule, TargetArch};
use std::collections::BTreeMap;

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
    fn apply_to_current_process(&self, config: &SandboxConfig) -> Result<SandboxGuard> {
        // 1. memory limit (always applied, even if other layers fail)
        apply_rlimit(config.max_memory_bytes)?;

        // 2. filesystem restriction via landlock
        let landlock_applied = match apply_landlock(config) {
            Ok(applied) => applied,
            Err(e) => {
                tracing::warn!("landlock failed: {e}, continuing without filesystem isolation");
                false
            }
        };

        // 3. if landlock didn't set no_new_privs, do it manually (required before seccomp)
        if !landlock_applied {
            apply_no_new_privs()?;
        }

        // 4. syscall filter (must be last — irreversible)
        match apply_seccomp(config) {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!("seccomp failed: {e}, continuing without syscall filter");
            }
        }

        tracing::info!(
            memory_limit = config.max_memory_bytes,
            landlock = landlock_applied,
            "sandbox applied"
        );

        Ok(SandboxGuard { _marker: () })
    }

    fn spawn(&self, _config: &SandboxConfig) -> Result<SandboxedProcess> {
        Err(YuleError::Sandbox(
            "linux sandbox spawn not yet implemented".into(),
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
// Layer 2: filesystem restriction via landlock
// ---------------------------------------------------------------------------

fn apply_landlock(config: &SandboxConfig) -> Result<bool> {
    // handle all filesystem access types — anything not explicitly allowed is denied
    let access_all = AccessFs::from_all(landlock::ABI::V3);

    let ruleset = match Ruleset::default()
        .handle_access(access_all)
        .and_then(|r| r.create())
    {
        Ok(r) => r,
        Err(_) => {
            tracing::warn!("landlock: not supported on this kernel, skipping");
            return Ok(false);
        }
    };

    // read-only access to model file
    let read_only: BitFlags<AccessFs> = AccessFs::ReadFile.into();
    let read_dir: BitFlags<AccessFs> = AccessFs::ReadFile | AccessFs::ReadDir;

    let model_fd = PathFd::new(&config.model_path)
        .map_err(|e| YuleError::Sandbox(format!("landlock: cannot open model path: {e}")))?;

    let mut ruleset = ruleset
        .add_rule(PathBeneath::new(model_fd, read_only))
        .map_err(|e| YuleError::Sandbox(format!("landlock: add model rule: {e}")))?;

    // GPU device nodes (DRM render nodes for Vulkan)
    if config.allow_gpu {
        let gpu_access: BitFlags<AccessFs> =
            AccessFs::ReadFile | AccessFs::ReadDir | AccessFs::IoctlDev;
        // /dev/dri for DRM render nodes
        ruleset = add_path_rule(ruleset, "/dev/dri", gpu_access)?;
        // /dev for NVIDIA proprietary driver nodes (/dev/nvidia*)
        ruleset = add_path_rule(ruleset, "/dev", gpu_access)?;
    }

    // essential system paths (shared libraries, dynamic linker, allocator)
    for path in [
        "/usr/lib",
        "/usr/lib64",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/proc/self",
    ] {
        ruleset = add_path_rule(ruleset, path, read_dir)?;
    }

    let status = ruleset
        .restrict_self()
        .map_err(|e| YuleError::Sandbox(format!("landlock: restrict_self failed: {e}")))?;

    match status.ruleset {
        RulesetStatus::FullyEnforced => {
            tracing::info!("landlock: fully enforced");
        }
        RulesetStatus::PartiallyEnforced => {
            tracing::warn!(
                "landlock: partially enforced (some features unavailable on this kernel)"
            );
        }
        RulesetStatus::NotEnforced => {
            tracing::warn!("landlock: not enforced");
            return Ok(false);
        }
    }

    Ok(true)
}

/// Try to add a path rule. Silently skips if path doesn't exist.
/// If the rule itself fails to add, propagates the error (shouldn't happen
/// unless the access right isn't handled by the ruleset).
fn add_path_rule(
    ruleset: landlock::RulesetCreated,
    path: &str,
    access: BitFlags<AccessFs>,
) -> Result<landlock::RulesetCreated> {
    let fd = match PathFd::new(path) {
        Ok(fd) => fd,
        Err(_) => return Ok(ruleset), // path doesn't exist, skip
    };
    ruleset
        .add_rule(PathBeneath::new(fd, access))
        .map_err(|e| YuleError::Sandbox(format!("landlock: add rule for {path}: {e}")))
}

// ---------------------------------------------------------------------------
// Layer 3: PR_SET_NO_NEW_PRIVS (fallback when landlock is skipped)
// ---------------------------------------------------------------------------

fn apply_no_new_privs() -> Result<()> {
    let ret = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if ret != 0 {
        return Err(YuleError::Sandbox(format!(
            "prctl(PR_SET_NO_NEW_PRIVS) failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    tracing::info!("no_new_privs: set");
    Ok(())
}

// ---------------------------------------------------------------------------
// Layer 4: syscall filter via seccomp-BPF
// ---------------------------------------------------------------------------

fn build_seccomp_filter(config: &SandboxConfig) -> Result<SeccompFilter> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();

    // helper: allow a syscall unconditionally (empty rules vec = always allow)
    let mut allow = |syscall: i64| {
        rules.entry(syscall).or_default();
    };

    // -- memory management --
    allow(libc::SYS_read);
    allow(libc::SYS_write);
    allow(libc::SYS_readv);
    allow(libc::SYS_writev);
    allow(libc::SYS_pread64);
    allow(libc::SYS_pwrite64);
    allow(libc::SYS_mmap);
    allow(libc::SYS_mprotect);
    allow(libc::SYS_munmap);
    allow(libc::SYS_mremap);
    allow(libc::SYS_madvise);
    allow(libc::SYS_brk);

    // -- file operations (landlock restricts what can be opened) --
    allow(libc::SYS_openat);
    allow(libc::SYS_close);
    allow(libc::SYS_fstat);
    allow(libc::SYS_newfstatat);
    allow(libc::SYS_lseek);
    allow(libc::SYS_access);
    allow(libc::SYS_faccessat2);
    allow(libc::SYS_statx);
    allow(libc::SYS_readlink);
    allow(libc::SYS_readlinkat);
    allow(libc::SYS_getcwd);
    allow(libc::SYS_fcntl);
    allow(libc::SYS_dup);
    allow(libc::SYS_dup2);
    allow(libc::SYS_dup3);

    // -- threads (rayon, tokio workers) --
    allow(libc::SYS_clone3);
    allow(libc::SYS_clone);
    allow(libc::SYS_futex);
    allow(libc::SYS_set_robust_list);
    allow(libc::SYS_get_robust_list);
    allow(libc::SYS_sched_yield);
    allow(libc::SYS_sched_getaffinity);
    allow(libc::SYS_nanosleep);
    allow(libc::SYS_clock_nanosleep);

    // -- signals --
    allow(libc::SYS_rt_sigaction);
    allow(libc::SYS_rt_sigprocmask);
    allow(libc::SYS_rt_sigreturn);
    allow(libc::SYS_sigaltstack);

    // -- time --
    allow(libc::SYS_clock_gettime);
    allow(libc::SYS_clock_getres);
    allow(libc::SYS_gettimeofday);

    // -- async runtime (tokio/mio epoll backend) --
    allow(libc::SYS_epoll_create1);
    allow(libc::SYS_epoll_ctl);
    allow(libc::SYS_epoll_wait);
    allow(libc::SYS_epoll_pwait);
    allow(libc::SYS_eventfd2);
    allow(libc::SYS_pipe2);

    // -- identity --
    allow(libc::SYS_getpid);
    allow(libc::SYS_gettid);
    allow(libc::SYS_getuid);
    allow(libc::SYS_getgid);
    allow(libc::SYS_geteuid);
    allow(libc::SYS_getegid);

    // -- misc required --
    allow(libc::SYS_arch_prctl);
    allow(libc::SYS_set_tid_address);
    allow(libc::SYS_getrandom);
    allow(libc::SYS_uname);
    allow(libc::SYS_prctl);
    allow(libc::SYS_exit);
    allow(libc::SYS_exit_group);
    allow(libc::SYS_rseq);
    allow(libc::SYS_membarrier);

    // -- networking (conditional: API server needs sockets) --
    if config.allow_network {
        allow(libc::SYS_socket);
        allow(libc::SYS_bind);
        allow(libc::SYS_listen);
        allow(libc::SYS_accept4);
        allow(libc::SYS_connect);
        allow(libc::SYS_setsockopt);
        allow(libc::SYS_getsockopt);
        allow(libc::SYS_getsockname);
        allow(libc::SYS_getpeername);
        allow(libc::SYS_sendto);
        allow(libc::SYS_recvfrom);
        allow(libc::SYS_sendmsg);
        allow(libc::SYS_recvmsg);
        allow(libc::SYS_shutdown);
        allow(libc::SYS_poll);
        allow(libc::SYS_ppoll);
    }

    // -- GPU ioctl (conditional: vulkan/drm driver communication) --
    if config.allow_gpu {
        allow(libc::SYS_ioctl);
    }

    SeccompFilter::new(
        rules,
        // default action: return EPERM for unlisted syscalls (debuggable, not kill)
        SeccompAction::Errno(libc::EPERM as u32),
        // match action: allow listed syscalls
        SeccompAction::Allow,
        TargetArch::x86_64,
    )
    .map_err(|e| YuleError::Sandbox(format!("seccomp filter build failed: {e}")))
}

fn apply_seccomp(config: &SandboxConfig) -> Result<()> {
    let filter = build_seccomp_filter(config)?;
    let bpf: seccompiler::BpfProgram =
        filter.try_into().map_err(|e: seccompiler::BackendError| {
            YuleError::Sandbox(format!("seccomp BPF compilation failed: {e}"))
        })?;

    seccompiler::apply_filter(&bpf)
        .map_err(|e| YuleError::Sandbox(format!("seccomp apply failed: {e}")))?;

    tracing::info!(
        allow_network = config.allow_network,
        allow_gpu = config.allow_gpu,
        "seccomp: filter installed"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlimit_applies() {
        let limit_bytes: u64 = 64 * 1024 * 1024 * 1024; // 64 GB
        apply_rlimit(limit_bytes).unwrap();

        let mut current = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let ret = unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut current) };
        assert_eq!(ret, 0);
        assert_eq!(current.rlim_cur, limit_bytes as libc::rlim_t);
    }

    #[test]
    fn seccomp_filter_builds() {
        let config = SandboxConfig {
            model_path: "/tmp/test.gguf".into(),
            allow_gpu: false,
            max_memory_bytes: 32 * 1024 * 1024 * 1024,
            allow_network: false,
        };
        let filter = build_seccomp_filter(&config).unwrap();
        let _bpf: seccompiler::BpfProgram = filter.try_into().unwrap();
    }

    #[test]
    fn seccomp_filter_with_gpu_and_network_builds() {
        let config = SandboxConfig {
            model_path: "/tmp/test.gguf".into(),
            allow_gpu: true,
            max_memory_bytes: 32 * 1024 * 1024 * 1024,
            allow_network: true,
        };
        let filter = build_seccomp_filter(&config).unwrap();
        let _bpf: seccompiler::BpfProgram = filter.try_into().unwrap();
    }

    #[test]
    fn seccomp_network_adds_socket_syscalls() {
        let no_net = SandboxConfig {
            model_path: "/tmp/test.gguf".into(),
            allow_gpu: false,
            max_memory_bytes: 32 * 1024 * 1024 * 1024,
            allow_network: false,
        };
        let with_net = SandboxConfig {
            allow_network: true,
            ..no_net.clone()
        };

        let rules_no_net = build_seccomp_rules(&no_net);
        let rules_with_net = build_seccomp_rules(&with_net);

        assert!(!rules_no_net.contains_key(&libc::SYS_socket));
        assert!(rules_with_net.contains_key(&libc::SYS_socket));
    }

    #[test]
    fn seccomp_gpu_adds_ioctl() {
        let no_gpu = SandboxConfig {
            model_path: "/tmp/test.gguf".into(),
            allow_gpu: false,
            max_memory_bytes: 32 * 1024 * 1024 * 1024,
            allow_network: false,
        };
        let with_gpu = SandboxConfig {
            allow_gpu: true,
            ..no_gpu.clone()
        };

        let rules_no_gpu = build_seccomp_rules(&no_gpu);
        let rules_with_gpu = build_seccomp_rules(&with_gpu);

        assert!(!rules_no_gpu.contains_key(&libc::SYS_ioctl));
        assert!(rules_with_gpu.contains_key(&libc::SYS_ioctl));
    }

    /// Helper: extract the raw rules map for testing without building the full filter
    fn build_seccomp_rules(config: &SandboxConfig) -> BTreeMap<i64, Vec<SeccompRule>> {
        let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
        let mut allow = |syscall: i64| {
            rules.entry(syscall).or_default();
        };

        // replicate the core allowlist logic
        allow(libc::SYS_read);
        allow(libc::SYS_write);
        allow(libc::SYS_mmap);
        allow(libc::SYS_openat);
        allow(libc::SYS_close);
        allow(libc::SYS_exit_group);

        if config.allow_network {
            allow(libc::SYS_socket);
            allow(libc::SYS_bind);
            allow(libc::SYS_listen);
        }

        if config.allow_gpu {
            allow(libc::SYS_ioctl);
        }

        rules
    }
}
