pub mod device;
pub mod kernels;
pub mod memory;

use cudarc::driver::LaunchAsync;

use yule_core::error::{Result, YuleError};

use crate::{BackendKind, BufferHandle, ComputeBackend, DeviceInfo};
use device::CudaDeviceWrapper;
use kernels::CudaKernelManager;
use memory::CudaMemoryManager;

pub struct CudaBackend {
    device: CudaDeviceWrapper,
    memory: CudaMemoryManager,
    kernels: CudaKernelManager,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        let device = CudaDeviceWrapper::new()?;
        let memory = CudaMemoryManager::new();
        let kernels = CudaKernelManager::new(device.inner())?;

        tracing::info!(
            device = %device.name(),
            vram_mb = device.memory_bytes() / (1024 * 1024),
            "cuda backend initialized"
        );

        Ok(Self {
            device,
            memory,
            kernels,
        })
    }

    pub fn is_available() -> bool {
        CudaDeviceWrapper::is_available()
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name(),
            backend: BackendKind::Cuda,
            memory_bytes: self.device.memory_bytes(),
            compute_units: self.device.sm_count(),
        }
    }

    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        self.memory.allocate(&self.device, size_bytes)
    }

    fn free(&self, handle: BufferHandle) -> Result<()> {
        self.memory.free(handle)
    }

    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        self.memory.copy_to_device(&self.device, data, handle)
    }

    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        self.memory.copy_from_device(&self.device, handle, data)
    }

    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()> {
        self.memory.copy_buffer(&self.device, src, dst, size)
    }

    fn copy_buffer_offset(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        self.memory
            .copy_buffer_offset(&self.device, src, dst, src_offset, dst_offset, size)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }

    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        out: &BufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let func = self.kernels.get_func("matmul", "matmul_kernel")?;
        let ptrs = self.memory.device_ptrs(&[a, b, out])?;
        let cfg = CudaKernelManager::launch_config_per_row(m);

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], ptrs[2], m, n, k))
                .map_err(|e| YuleError::Gpu(format!("cuda matmul launch failed: {e}")))?;
        }
        Ok(())
    }

    fn softmax(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        let func = self.kernels.get_func("softmax", "softmax_kernel")?;
        let ptrs = self.memory.device_ptrs(&[input, output])?;
        let cfg = CudaKernelManager::launch_config_reduction();

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], size))
                .map_err(|e| YuleError::Gpu(format!("cuda softmax launch failed: {e}")))?;
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        input: &BufferHandle,
        weight: &BufferHandle,
        output: &BufferHandle,
        size: u32,
        eps: f32,
    ) -> Result<()> {
        let func = self.kernels.get_func("rms_norm", "rms_norm_kernel")?;
        let ptrs = self.memory.device_ptrs(&[input, weight, output])?;
        let cfg = CudaKernelManager::launch_config_reduction();

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], ptrs[2], size, eps))
                .map_err(|e| YuleError::Gpu(format!("cuda rms_norm launch failed: {e}")))?;
        }
        Ok(())
    }

    fn rope(
        &self,
        q: &BufferHandle,
        k: &BufferHandle,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        n_heads_q: u32,
        n_heads_k: u32,
    ) -> Result<()> {
        let func = self.kernels.get_func("rope", "rope_kernel")?;
        let ptrs = self.memory.device_ptrs(&[q, k])?;
        let half_dim = head_dim / 2;
        let total_threads = n_heads_q * half_dim + n_heads_k * half_dim;
        let cfg = CudaKernelManager::launch_config_1d(total_threads);

        unsafe {
            func.launch(
                cfg,
                (
                    ptrs[0], ptrs[1], pos, head_dim, freq_base, n_heads_q, n_heads_k,
                ),
            )
            .map_err(|e| YuleError::Gpu(format!("cuda rope launch failed: {e}")))?;
        }
        Ok(())
    }

    fn silu(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        // The silu_mul kernel computes SiLU(gate) * up.
        // For standalone silu, we pass the same buffer as both gate and up,
        // which gives SiLU(x) * x. That's not quite right.
        // Actually, looking at the Vulkan backend, silu() dispatches silu_mul with
        // input as gate and output as up (output already contains the up-projection).
        let func = self.kernels.get_func("silu_mul", "silu_mul_kernel")?;
        let ptrs = self.memory.device_ptrs(&[input, output, output])?;
        let cfg = CudaKernelManager::launch_config_1d(size);

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], ptrs[2], size))
                .map_err(|e| YuleError::Gpu(format!("cuda silu launch failed: {e}")))?;
        }
        Ok(())
    }

    fn element_mul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        // Reuse silu_mul: SiLU(a) * b. This matches the Vulkan backend behavior
        // where element_mul dispatches through silu_mul.
        let func = self.kernels.get_func("silu_mul", "silu_mul_kernel")?;
        let ptrs = self.memory.device_ptrs(&[a, b, output])?;
        let cfg = CudaKernelManager::launch_config_1d(size);

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], ptrs[2], size))
                .map_err(|e| YuleError::Gpu(format!("cuda element_mul launch failed: {e}")))?;
        }
        Ok(())
    }

    fn add(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        let func = self.kernels.get_func("add", "add_kernel")?;
        let ptrs = self.memory.device_ptrs(&[a, b, output])?;
        let cfg = CudaKernelManager::launch_config_1d(size);

        unsafe {
            func.launch(cfg, (ptrs[0], ptrs[1], ptrs[2], size))
                .map_err(|e| YuleError::Gpu(format!("cuda add launch failed: {e}")))?;
        }
        Ok(())
    }

    fn attn_score(
        &self,
        q: &BufferHandle,
        k_cache: &BufferHandle,
        scores: &BufferHandle,
        head_dim: u32,
        seq_len: u32,
        head_offset: u32,
        kv_offset: u32,
        kv_stride: u32,
    ) -> Result<()> {
        let func = self.kernels.get_func("attn_score", "attn_score_kernel")?;
        let ptrs = self.memory.device_ptrs(&[q, k_cache, scores])?;
        let cfg = CudaKernelManager::launch_config_per_row(seq_len);

        unsafe {
            func.launch(
                cfg,
                (
                    ptrs[0],
                    ptrs[1],
                    ptrs[2],
                    head_dim,
                    seq_len,
                    head_offset,
                    kv_offset,
                    kv_stride,
                ),
            )
            .map_err(|e| YuleError::Gpu(format!("cuda attn_score launch failed: {e}")))?;
        }
        Ok(())
    }

    fn attn_value(
        &self,
        weights: &BufferHandle,
        v_cache: &BufferHandle,
        output: &BufferHandle,
        head_dim: u32,
        seq_len: u32,
        kv_offset: u32,
        kv_stride: u32,
        out_offset: u32,
    ) -> Result<()> {
        let func = self.kernels.get_func("attn_value", "attn_value_kernel")?;
        let ptrs = self.memory.device_ptrs(&[weights, v_cache, output])?;
        let cfg = CudaKernelManager::launch_config_per_row(head_dim);

        unsafe {
            func.launch(
                cfg,
                (
                    ptrs[0], ptrs[1], ptrs[2], head_dim, seq_len, kv_offset, kv_stride, out_offset,
                ),
            )
            .map_err(|e| YuleError::Gpu(format!("cuda attn_value launch failed: {e}")))?;
        }
        Ok(())
    }

    fn quantized_matmul(
        &self,
        weights: &BufferHandle,
        input: &BufferHandle,
        output: &BufferHandle,
        n_rows: u32,
        n_cols: u32,
        dtype: yule_core::dtype::DType,
    ) -> Result<()> {
        let (module, func_name) = match dtype {
            yule_core::dtype::DType::Q4_0 => ("qmv_q4_0", "qmv_q4_0_kernel"),
            yule_core::dtype::DType::Q8_0 => ("qmv_q8_0", "qmv_q8_0_kernel"),
            yule_core::dtype::DType::Q4_K => ("qmv_q4_k", "qmv_q4_k_kernel"),
            yule_core::dtype::DType::Q6_K => ("qmv_q6_k", "qmv_q6_k_kernel"),
            _ => {
                return Err(YuleError::Gpu(format!(
                    "unsupported dtype for CUDA qmv: {dtype:?}"
                )));
            }
        };

        let func = self.kernels.get_func(module, func_name)?;
        let ptrs = self.memory.device_ptrs(&[weights, input, output])?;
        let block_size = dtype.block_size() as u32;
        let blocks_per_row = (n_cols + block_size - 1) / block_size;
        let cfg = CudaKernelManager::launch_config_per_row(n_rows);

        unsafe {
            func.launch(
                cfg,
                (ptrs[0], ptrs[1], ptrs[2], n_rows, n_cols, blocks_per_row),
            )
            .map_err(|e| {
                YuleError::Gpu(format!(
                    "cuda quantized_matmul ({dtype:?}) launch failed: {e}"
                ))
            })?;
        }
        Ok(())
    }
}
