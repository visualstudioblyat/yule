pub mod device;
pub mod memory;

use yule_core::error::{Result, YuleError};

use crate::{BackendKind, BufferHandle, ComputeBackend, DeviceInfo};
use device::MetalDeviceWrapper;
use memory::MetalMemoryManager;

pub struct MetalBackend {
    device: MetalDeviceWrapper,
    memory: MetalMemoryManager,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = MetalDeviceWrapper::new()?;
        let memory = MetalMemoryManager::new();

        tracing::info!(
            device = %device.name(),
            vram_mb = device.memory_bytes() / (1024 * 1024),
            "metal backend initialized"
        );

        Ok(Self { device, memory })
    }

    pub fn is_available() -> bool {
        MetalDeviceWrapper::is_available()
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name(),
            backend: BackendKind::Metal,
            memory_bytes: self.device.memory_bytes(),
            compute_units: self.device.max_threads_per_threadgroup(),
        }
    }

    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        self.memory.allocate(&self.device, size_bytes)
    }

    fn free(&self, handle: BufferHandle) -> Result<()> {
        self.memory.free(handle)
    }

    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        self.memory.copy_to_device(data, handle)
    }

    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        self.memory.copy_from_device(handle, data)
    }

    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()> {
        self.memory.copy_buffer(&self.device, src, dst, 0, 0, size)
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
            .copy_buffer(&self.device, src, dst, src_offset, dst_offset, size)
    }

    fn synchronize(&self) -> Result<()> {
        // Metal command buffers are committed and waited on per-dispatch.
        // For explicit sync, we just ensure the command queue is drained.
        Ok(())
    }

    fn matmul(
        &self,
        _a: &BufferHandle,
        _b: &BufferHandle,
        _out: &BufferHandle,
        _m: u32,
        _n: u32,
        _k: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal matmul: kernel not yet implemented".into(),
        ))
    }

    fn softmax(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
        Err(YuleError::Gpu(
            "metal softmax: kernel not yet implemented".into(),
        ))
    }

    fn rms_norm(
        &self,
        _input: &BufferHandle,
        _weight: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
        _eps: f32,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal rms_norm: kernel not yet implemented".into(),
        ))
    }

    fn rope(
        &self,
        _q: &BufferHandle,
        _k: &BufferHandle,
        _pos: u32,
        _head_dim: u32,
        _freq_base: f32,
        _n_heads_q: u32,
        _n_heads_k: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal rope: kernel not yet implemented".into(),
        ))
    }

    fn silu(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
        Err(YuleError::Gpu(
            "metal silu: kernel not yet implemented".into(),
        ))
    }

    fn element_mul(
        &self,
        _a: &BufferHandle,
        _b: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal element_mul: kernel not yet implemented".into(),
        ))
    }

    fn add(
        &self,
        _a: &BufferHandle,
        _b: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal add: kernel not yet implemented".into(),
        ))
    }

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
        Err(YuleError::Gpu(
            "metal attn_score: kernel not yet implemented".into(),
        ))
    }

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
        Err(YuleError::Gpu(
            "metal attn_value: kernel not yet implemented".into(),
        ))
    }

    fn quantized_matmul(
        &self,
        _weights: &BufferHandle,
        _input: &BufferHandle,
        _output: &BufferHandle,
        _n_rows: u32,
        _n_cols: u32,
        _dtype: yule_core::dtype::DType,
    ) -> Result<()> {
        Err(YuleError::Gpu(
            "metal quantized_matmul: kernel not yet implemented".into(),
        ))
    }
}
