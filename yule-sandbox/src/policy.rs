use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub filesystem: FilesystemPolicy,
    pub network: NetworkPolicy,
    pub gpu: GpuPolicy,
    pub resources: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemPolicy {
    pub read_only_paths: Vec<std::path::PathBuf>,
    pub deny_all_other: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    pub allow: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPolicy {
    pub allow: bool,
    pub allowed_ioctls: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_bytes: u64,
    pub max_cpu_percent: Option<u32>,
}

impl SandboxPolicy {
    pub fn inference_default(model_path: std::path::PathBuf) -> Self {
        Self {
            filesystem: FilesystemPolicy {
                read_only_paths: vec![model_path],
                deny_all_other: true,
            },
            network: NetworkPolicy { allow: false },
            gpu: GpuPolicy {
                allow: true,
                allowed_ioctls: Vec::new(), // populated per-platform
            },
            resources: ResourceLimits {
                max_memory_bytes: 32 * 1024 * 1024 * 1024, // 32GB default
                max_cpu_percent: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_denies_network() {
        let policy = SandboxPolicy::inference_default("/tmp/model.gguf".into());
        assert!(!policy.network.allow);
    }

    #[test]
    fn default_policy_allows_gpu() {
        let policy = SandboxPolicy::inference_default("/tmp/model.gguf".into());
        assert!(policy.gpu.allow);
    }

    #[test]
    fn default_policy_restricts_filesystem() {
        let policy = SandboxPolicy::inference_default("/tmp/model.gguf".into());
        assert!(policy.filesystem.deny_all_other);
        assert_eq!(policy.filesystem.read_only_paths.len(), 1);
    }

    #[test]
    fn default_memory_limit_is_32gb() {
        let policy = SandboxPolicy::inference_default("/tmp/model.gguf".into());
        assert_eq!(policy.resources.max_memory_bytes, 32 * 1024 * 1024 * 1024);
    }

    #[test]
    fn policy_serde_roundtrip() {
        let policy = SandboxPolicy::inference_default("/tmp/model.gguf".into());
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: SandboxPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.resources.max_memory_bytes,
            policy.resources.max_memory_bytes
        );
        assert_eq!(parsed.network.allow, policy.network.allow);
    }
}
