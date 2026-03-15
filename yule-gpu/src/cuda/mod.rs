pub mod device;
pub mod memory;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

use yule_core::error::{Result, YuleError};

use crate::{BackendKind, BufferHandle, ComputeBackend, DeviceInfo};
use device::CudaDeviceWrapper;
use memory::CudaMemoryManager;

pub struct CudaBackend {
    device: CudaDeviceWrapper,
    memory: CudaMemoryManager,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        let device = CudaDeviceWrapper::new()?;
        let memory = CudaMemoryManager::new();

        tracing::info!(
            device = %device.name(),
            vram_mb = device.memory_bytes() / (1024 * 1024),
            "cuda backend initialized"
        );

        Ok(Self { device, memory })
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
        _a: &BufferHandle,
        _b: &BufferHandle,
        _out: &BufferHandle,
        _m: u32,
        _n: u32,
        _k: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu("cuda matmul: kernel not yet implemented".into()))
    }

    fn softmax(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
        Err(YuleError::Gpu("cuda softmax: kernel not yet implemented".into()))
    }

    fn rms_norm(
        &self,
        _input: &BufferHandle,
        _weight: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
        _eps: f32,
    ) -> Result<()> {
        Err(YuleError::Gpu("cuda rms_norm: kernel not yet implemented".into()))
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
        Err(YuleError::Gpu("cuda rope: kernel not yet implemented".into()))
    }

    fn silu(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
        Err(YuleError::Gpu("cuda silu: kernel not yet implemented".into()))
    }

    fn element_mul(
        &self,
        _a: &BufferHandle,
        _b: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu("cuda element_mul: kernel not yet implemented".into()))
    }

    fn add(
        &self,
        _a: &BufferHandle,
        _b: &BufferHandle,
        _output: &BufferHandle,
        _size: u32,
    ) -> Result<()> {
        Err(YuleError::Gpu("cuda add: kernel not yet implemented".into()))
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
        Err(YuleError::Gpu("cuda attn_score: kernel not yet implemented".into()))
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
        Err(YuleError::Gpu("cuda attn_value: kernel not yet implemented".into()))
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
        Err(YuleError::Gpu("cuda quantized_matmul: kernel not yet implemented".into()))
    }
}
