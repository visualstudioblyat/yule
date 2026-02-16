pub mod backend;
pub mod buffer;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "vulkan")]
pub use ash::vk;

use yule_core::error::Result;

#[allow(clippy::too_many_arguments)]
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn device_info(&self) -> DeviceInfo;
    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle>;
    fn free(&self, handle: BufferHandle) -> Result<()>;
    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        out: &BufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()>;
    fn softmax(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()>;
    fn rms_norm(
        &self,
        input: &BufferHandle,
        weight: &BufferHandle,
        output: &BufferHandle,
        size: u32,
        eps: f32,
    ) -> Result<()>;
    fn rope(
        &self,
        q: &BufferHandle,
        k: &BufferHandle,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        n_heads_q: u32,
        n_heads_k: u32,
    ) -> Result<()>;
    fn silu(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()>;
    fn element_mul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()>;
    fn add(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()>;
    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()>;
    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()>;
    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()>;
    fn copy_buffer_offset(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()>;
    fn synchronize(&self) -> Result<()>;

    /// Compute attention scores: scores[pos] = Q[head_offset..] · K_cache[pos*kv_stride+kv_offset..]
    fn attn_score(
        &self,
        _q: &BufferHandle,
        _k_cache: &BufferHandle,
        _scores: &BufferHandle,
        _head_dim: u32,
        _seq_len: u32,
        _head_offset: u32,
        _kv_offset: u32,
        _kv_stride: u32,
    ) -> Result<()> {
        Err(yule_core::error::YuleError::Gpu(
            "attn_score not supported on this backend".into(),
        ))
    }

    /// Compute weighted value aggregation: out[out_offset+d] = sum_pos(weights[pos] * V[pos*kv_stride+kv_offset+d])
    fn attn_value(
        &self,
        _weights: &BufferHandle,
        _v_cache: &BufferHandle,
        _output: &BufferHandle,
        _head_dim: u32,
        _seq_len: u32,
        _kv_offset: u32,
        _kv_stride: u32,
        _out_offset: u32,
    ) -> Result<()> {
        Err(yule_core::error::YuleError::Gpu(
            "attn_value not supported on this backend".into(),
        ))
    }

    /// Fused dequantize + matrix-vector multiply for quantized weights.
    /// GPU backends override this for fused VRAM kernels.
    /// Default impl falls back to regular matmul (assumes pre-dequantized data).
    fn quantized_matmul(
        &self,
        _weights: &BufferHandle,
        _input: &BufferHandle,
        _output: &BufferHandle,
        _n_rows: u32,
        _n_cols: u32,
        _dtype: yule_core::dtype::DType,
    ) -> Result<()> {
        Err(yule_core::error::YuleError::Gpu(
            "quantized_matmul not supported on this backend".into(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: BackendKind,
    pub memory_bytes: u64,
    pub compute_units: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Vulkan,
    Cuda,
    Metal,
}

#[allow(unused_mut)]
pub fn detect_backends() -> Vec<BackendKind> {
    let mut backends = vec![BackendKind::Cpu];

    #[cfg(feature = "vulkan")]
    if vulkan::VulkanBackend::is_available() {
        backends.push(BackendKind::Vulkan);
    }

    backends
}

pub fn create_backend(kind: BackendKind) -> Result<Box<dyn ComputeBackend>> {
    match kind {
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => Ok(Box::new(cpu::CpuBackend::new())),
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => Ok(Box::new(vulkan::VulkanBackend::new()?)),
        _ => Err(yule_core::error::YuleError::Gpu(format!(
            "backend {kind:?} not compiled in — enable the feature flag"
        ))),
    }
}
