pub mod commands;
pub mod device;
pub mod memory;
pub mod pipeline;

use ash::vk;
use yule_core::error::{Result, YuleError};

use crate::{BufferHandle, ComputeBackend, DeviceInfo};
use commands::CommandEngine;
use device::VulkanDevice;
use memory::MemoryManager;
use pipeline::{PipelineManager, ShaderKey};

pub struct VulkanBackend {
    // Drop order matters: commands and pipelines first (they hold Vk handles),
    // then memory (frees buffers+allocations), then vk_device last (destroys device).
    commands: CommandEngine,
    pipelines: PipelineManager,
    memory: MemoryManager,
    vk_device: VulkanDevice,
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
        let vk_device = VulkanDevice::new()?;
        let memory = MemoryManager::new(&vk_device)?;
        let mut pipelines = PipelineManager::new(&vk_device.device)?;
        let commands = CommandEngine::new(&vk_device)?;

        // Load all pre-compiled shaders
        pipelines.register_all_shaders()?;

        tracing::info!("vulkan backend initialized");

        Ok(Self {
            commands,
            pipelines,
            memory,
            vk_device,
        })
    }

    pub fn is_available() -> bool {
        VulkanDevice::is_available()
    }

    /// Write descriptor set bindings for buffers.
    fn write_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        buffers: &[&BufferHandle],
    ) -> Result<()> {
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        for handle in buffers {
            let vk_buf = self.memory.get_buffer(handle)?;
            let size = self.memory.get_size(handle)?;
            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer: vk_buf,
                offset: 0,
                range: size as u64,
            });
        }

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            self.vk_device.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    /// Dispatch a shader with given buffers, push constants, and workgroup counts.
    fn dispatch(
        &self,
        key: ShaderKey,
        buffers: &[&BufferHandle],
        push_constants: &[u8],
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) -> Result<()> {
        let pipeline = self.pipelines.get(key)?;
        let ds = self.pipelines.allocate_descriptor_set(key)?;
        self.write_descriptor_set(ds, buffers)?;

        let cmd = self.commands.begin_command_buffer()?;
        self.commands
            .record_dispatch(cmd, pipeline, ds, push_constants, wg_x, wg_y, wg_z);
        self.commands.submit_and_wait(cmd)?;

        Ok(())
    }

    // --- Batched command buffer API ---

    /// Reset the descriptor pool, freeing all sets from the previous forward pass.
    /// Must be called after GPU work completes and before allocating new sets.
    pub fn reset_descriptors(&self) -> Result<()> {
        self.pipelines.reset_descriptor_pool()
    }

    /// Begin a batched command buffer. All dispatch_batched calls record into it.
    pub fn begin_batch(&self) -> Result<vk::CommandBuffer> {
        self.commands.begin_command_buffer()
    }

    /// Record a compute dispatch into an existing command buffer (no submit).
    pub fn dispatch_batched(
        &self,
        cmd: vk::CommandBuffer,
        key: ShaderKey,
        buffers: &[&BufferHandle],
        push_constants: &[u8],
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) -> Result<()> {
        let pipeline = self.pipelines.get(key)?;
        let ds = self.pipelines.allocate_descriptor_set(key)?;
        self.write_descriptor_set(ds, buffers)?;
        self.commands
            .record_dispatch(cmd, pipeline, ds, push_constants, wg_x, wg_y, wg_z);
        Ok(())
    }

    /// Record a compute → compute memory barrier into the command buffer.
    pub fn barrier(&self, cmd: vk::CommandBuffer) {
        self.commands.record_barrier(cmd);
    }

    /// Record a transfer → compute barrier into the command buffer.
    pub fn transfer_barrier(&self, cmd: vk::CommandBuffer) {
        self.commands.record_transfer_barrier(cmd);
    }

    /// Record a buffer-to-buffer copy into the command buffer.
    pub fn copy_buffer_batched(
        &self,
        cmd: vk::CommandBuffer,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let src_buf = self.memory.get_buffer(src)?;
        let dst_buf = self.memory.get_buffer(dst)?;
        self.commands.record_copy_offset(
            cmd,
            src_buf,
            dst_buf,
            src_offset as u64,
            dst_offset as u64,
            size as u64,
        );
        Ok(())
    }

    /// Submit the batched command buffer and wait for completion.
    pub fn submit_batch(&self, cmd: vk::CommandBuffer) -> Result<()> {
        self.commands.submit_and_wait(cmd)
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn device_info(&self) -> DeviceInfo {
        self.vk_device.device_info()
    }

    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        self.memory.allocate(size_bytes)
    }

    fn free(&self, handle: BufferHandle) -> Result<()> {
        self.memory.free(handle)
    }

    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        self.commands.copy_to_device(&self.memory, data, handle)
    }

    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        self.commands.copy_from_device(&self.memory, handle, data)
    }

    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()> {
        self.commands.copy_buffer(&self.memory, src, dst, size)
    }

    fn copy_buffer_offset(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        self.commands
            .copy_buffer_offset(&self.memory, src, dst, src_offset, dst_offset, size)
    }

    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        out: &BufferHandle,
        m: u32,
        n: u32,
        _k: u32,
    ) -> Result<()> {
        // f32 matmul — not the hot path (quantized_matmul is)
        // For now, push constants: [m, n, k]
        let push = [m, n, _k];
        let push_bytes = bytemuck::cast_slice(&push);
        let wg_x = m; // one workgroup per output row
        self.dispatch(ShaderKey::QmvQ4_0, &[a, b, out], push_bytes, wg_x, 1, 1)
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
        let key = match dtype {
            yule_core::dtype::DType::Q4_0 => ShaderKey::QmvQ4_0,
            yule_core::dtype::DType::Q4_K => ShaderKey::QmvQ4K,
            yule_core::dtype::DType::Q6_K => ShaderKey::QmvQ6K,
            yule_core::dtype::DType::Q8_0 => ShaderKey::QmvQ8_0,
            _ => {
                return Err(YuleError::Gpu(format!(
                    "unsupported dtype for GPU qmv: {dtype:?}"
                )));
            }
        };

        let block_size = dtype.block_size() as u32;
        let blocks_per_row = (n_cols + block_size - 1) / block_size;
        let push = [n_rows, n_cols, blocks_per_row];
        let push_bytes = bytemuck::cast_slice(&push);

        self.dispatch(key, &[weights, input, output], push_bytes, n_rows, 1, 1)
    }

    fn softmax(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        let push = [size];
        let push_bytes = bytemuck::cast_slice(&push);
        self.dispatch(ShaderKey::Softmax, &[input, output], push_bytes, 1, 1, 1)
    }

    fn rms_norm(
        &self,
        input: &BufferHandle,
        weight: &BufferHandle,
        output: &BufferHandle,
        size: u32,
        eps: f32,
    ) -> Result<()> {
        let push = [size, eps.to_bits()];
        let push_bytes = bytemuck::cast_slice(&push);
        self.dispatch(
            ShaderKey::RmsNorm,
            &[input, weight, output],
            push_bytes,
            1,
            1,
            1,
        )
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
        let push = [pos, head_dim, freq_base.to_bits(), n_heads_q, n_heads_k];
        let push_bytes = bytemuck::cast_slice(&push);
        let half_dim = head_dim / 2;
        let total_threads = n_heads_q * half_dim + n_heads_k * half_dim;
        let wg_x = (total_threads + 63) / 64;
        self.dispatch(ShaderKey::Rope, &[q, k, q, k], push_bytes, wg_x, 1, 1)
    }

    fn silu(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        let push = [size];
        let push_bytes = bytemuck::cast_slice(&push);
        self.dispatch(
            ShaderKey::SiluMul,
            &[input, output, output],
            push_bytes,
            (size + 255) / 256,
            1,
            1,
        )
    }

    fn element_mul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        // Reuse silu_mul shader in mul-only mode (TODO: separate shader)
        let push = [size];
        let push_bytes = bytemuck::cast_slice(&push);
        self.dispatch(
            ShaderKey::SiluMul,
            &[a, b, output],
            push_bytes,
            (size + 255) / 256,
            1,
            1,
        )
    }

    fn add(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        let push = [size];
        let push_bytes = bytemuck::cast_slice(&push);
        self.dispatch(
            ShaderKey::Add,
            &[a, b, output],
            push_bytes,
            (size + 255) / 256,
            1,
            1,
        )
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
        let push = [head_dim, seq_len, head_offset, kv_offset, kv_stride];
        let push_bytes = bytemuck::cast_slice(&push);
        // One workgroup per position, scratch buffer reused as 4th binding
        self.dispatch(
            ShaderKey::AttnScore,
            &[q, k_cache, scores, scores],
            push_bytes,
            seq_len,
            1,
            1,
        )
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
        let push = [head_dim, seq_len, kv_offset, kv_stride, out_offset];
        let push_bytes = bytemuck::cast_slice(&push);
        // One workgroup per output dimension
        self.dispatch(
            ShaderKey::AttnValue,
            &[weights, v_cache, output, output],
            push_bytes,
            head_dim,
            1,
            1,
        )
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.vk_device
                .device
                .device_wait_idle()
                .map_err(|e| YuleError::Gpu(format!("device_wait_idle failed: {e}")))?;
        }
        Ok(())
    }
}
