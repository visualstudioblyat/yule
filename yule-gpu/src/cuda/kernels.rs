//! CUDA kernel manager: compiles .cu sources via nvcc at runtime, loads PTX modules,
//! and provides typed kernel launch helpers.
//!
//! Kernel lifecycle:
//! 1. On `CudaKernelManager::new()`, CUDA C sources are written to a temp directory
//!    and compiled to PTX via `nvcc -ptx`. If nvcc is unavailable, it falls back to
//!    loading pre-compiled PTX from `yule-gpu/kernels/cuda/compiled/`.
//! 2. Each PTX module is loaded into the CudaDevice via `load_ptx()`.
//! 3. Kernel functions are retrieved via `get_func()` and launched with typed wrappers.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use yule_core::error::{Result, YuleError};

/// Embedded CUDA C kernel sources for runtime compilation.
const KERNEL_SOURCES: &[(&str, &str, &[&str])] = &[
    (
        "add",
        include_str!("../../kernels/cuda/add.cu"),
        &["add_kernel"],
    ),
    (
        "silu_mul",
        include_str!("../../kernels/cuda/silu_mul.cu"),
        &["silu_mul_kernel"],
    ),
    (
        "rms_norm",
        include_str!("../../kernels/cuda/rms_norm.cu"),
        &["rms_norm_kernel"],
    ),
    (
        "softmax",
        include_str!("../../kernels/cuda/softmax.cu"),
        &["softmax_kernel"],
    ),
    (
        "rope",
        include_str!("../../kernels/cuda/rope.cu"),
        &["rope_kernel"],
    ),
    (
        "attn_score",
        include_str!("../../kernels/cuda/attn_score.cu"),
        &["attn_score_kernel"],
    ),
    (
        "attn_value",
        include_str!("../../kernels/cuda/attn_value.cu"),
        &["attn_value_kernel"],
    ),
    (
        "embed_lookup",
        include_str!("../../kernels/cuda/embed_lookup.cu"),
        &["embed_lookup_kernel"],
    ),
    (
        "matmul",
        include_str!("../../kernels/cuda/matmul.cu"),
        &["matmul_kernel"],
    ),
    (
        "qmv_q4_0",
        include_str!("../../kernels/cuda/qmv_q4_0.cu"),
        &["qmv_q4_0_kernel"],
    ),
    (
        "qmv_q8_0",
        include_str!("../../kernels/cuda/qmv_q8_0.cu"),
        &["qmv_q8_0_kernel"],
    ),
    (
        "qmv_q4_k",
        include_str!("../../kernels/cuda/qmv_q4_k.cu"),
        &["qmv_q4_k_kernel"],
    ),
    (
        "qmv_q6_k",
        include_str!("../../kernels/cuda/qmv_q6_k.cu"),
        &["qmv_q6_k_kernel"],
    ),
];

pub struct CudaKernelManager {
    device: Arc<CudaDevice>,
}

impl CudaKernelManager {
    /// Initialize the kernel manager: compile CUDA C sources to PTX and load all modules.
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self> {
        let mgr = Self {
            device: Arc::clone(device),
        };
        mgr.load_all_kernels()?;
        Ok(mgr)
    }

    fn load_all_kernels(&self) -> Result<()> {
        // Strategy 1: Try compiling CUDA C sources with nvcc at runtime
        if let Ok(ptx_dir) = self.compile_with_nvcc() {
            tracing::info!("compiled CUDA kernels with nvcc");
            return self.load_ptx_from_dir(&ptx_dir);
        }

        // Strategy 2: Try loading pre-compiled PTX from the source tree
        let precompiled = Self::precompiled_ptx_dir();
        if precompiled.is_dir() {
            tracing::info!(path = %precompiled.display(), "loading pre-compiled CUDA PTX");
            return self.load_ptx_from_dir(&precompiled);
        }

        Err(YuleError::Gpu(
            "CUDA kernels unavailable: nvcc not found and no pre-compiled PTX in \
             yule-gpu/kernels/cuda/compiled/. Install the CUDA toolkit or run \
             yule-gpu/kernels/cuda/compile.sh"
                .into(),
        ))
    }

    /// Path to pre-compiled PTX directory relative to the crate root.
    fn precompiled_ptx_dir() -> PathBuf {
        // Try relative to the executable first, then fall back to the manifest dir
        if let Ok(exe) = std::env::current_exe() {
            let dir = exe
                .parent()
                .unwrap_or(Path::new("."))
                .join("kernels")
                .join("cuda")
                .join("compiled");
            if dir.is_dir() {
                return dir;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("cuda")
            .join("compiled")
    }

    /// Compile embedded CUDA C sources to PTX using nvcc in a temp directory.
    fn compile_with_nvcc(&self) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        // Check if nvcc is available
        let nvcc_check = Command::new("nvcc").arg("--version").output();
        if nvcc_check.is_err() {
            return Err("nvcc not found".into());
        }

        let tmp_dir = std::env::temp_dir().join("yule_cuda_kernels");
        std::fs::create_dir_all(&tmp_dir)?;

        let ptx_dir = tmp_dir.join("ptx");
        std::fs::create_dir_all(&ptx_dir)?;

        for (name, source, _fns) in KERNEL_SOURCES {
            let cu_path = tmp_dir.join(format!("{name}.cu"));
            let ptx_path = ptx_dir.join(format!("{name}.ptx"));

            // Skip if PTX already exists and is newer than this build
            if ptx_path.exists() {
                // Simple cache: if the PTX file exists, use it
                // (recompile by deleting the temp directory)
                continue;
            }

            let mut f = std::fs::File::create(&cu_path)?;
            f.write_all(source.as_bytes())?;

            let output = Command::new("nvcc")
                .args(["-ptx", "-arch=sm_70", "-o"])
                .arg(&ptx_path)
                .arg(&cu_path)
                .output()?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::warn!(kernel = name, err = %stderr, "nvcc compilation failed");
                return Err(format!("nvcc failed for {name}: {stderr}").into());
            }
        }

        Ok(ptx_dir)
    }

    /// Load all PTX files from a directory into the CUDA device.
    fn load_ptx_from_dir(&self, dir: &Path) -> Result<()> {
        for (name, _source, fn_names) in KERNEL_SOURCES {
            let ptx_path = dir.join(format!("{name}.ptx"));
            let ptx_text = std::fs::read_to_string(&ptx_path).map_err(|e| {
                YuleError::Gpu(format!(
                    "failed to read PTX file {}: {e}",
                    ptx_path.display()
                ))
            })?;

            self.device
                .load_ptx(cudarc::driver::Ptx::from_src(ptx_text), name, fn_names)
                .map_err(|e| YuleError::Gpu(format!("failed to load PTX module '{name}': {e}")))?;
        }
        Ok(())
    }

    /// Get a kernel function by module and function name.
    pub fn get_func(&self, module: &str, func: &str) -> Result<CudaFunction> {
        self.device.get_func(module, func).ok_or_else(|| {
            YuleError::Gpu(format!(
                "CUDA kernel '{module}::{func}' not loaded. \
                 Ensure kernels are compiled (run yule-gpu/kernels/cuda/compile.sh)"
            ))
        })
    }

    /// Standard launch config: 256 threads per block, computed grid size.
    pub fn launch_config_1d(total_threads: u32) -> LaunchConfig {
        let block = 256u32;
        let grid = (total_threads + block - 1) / block;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch config for reduction kernels: 1 workgroup, 256 threads, shared memory.
    pub fn launch_config_reduction() -> LaunchConfig {
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4, // 256 floats
        }
    }

    /// Launch config for per-row reduction: N workgroups, 256 threads each.
    pub fn launch_config_per_row(n_rows: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        }
    }
}
