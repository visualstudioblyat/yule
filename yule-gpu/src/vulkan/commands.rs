use ash::vk;
use yule_core::error::{Result, YuleError};

use super::device::VulkanDevice;
use super::memory::MemoryManager;
use super::pipeline::ComputePipeline;
use crate::BufferHandle;

pub struct CommandEngine {
    device: ash::Device,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    queue: vk::Queue,
}

impl CommandEngine {
    pub fn new(vk_dev: &VulkanDevice) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(vk_dev.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            vk_dev
                .device
                .create_command_pool(&pool_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create command pool: {e}")))?
        };

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            vk_dev
                .device
                .create_fence(&fence_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create fence: {e}")))?
        };

        Ok(Self {
            device: vk_dev.device.clone(),
            command_pool,
            fence,
            queue: vk_dev.queue,
        })
    }

    /// Allocate a one-time command buffer.
    pub fn begin_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| YuleError::Gpu(format!("failed to allocate command buffer: {e}")))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| YuleError::Gpu(format!("failed to begin command buffer: {e}")))?;
        }

        Ok(cmd)
    }

    /// Submit command buffer and wait for completion.
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| YuleError::Gpu(format!("failed to end command buffer: {e}")))?;

            let cmd_bufs = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| YuleError::Gpu(format!("failed to reset fence: {e}")))?;

            self.device
                .queue_submit(self.queue, &[submit_info], self.fence)
                .map_err(|e| YuleError::Gpu(format!("failed to submit: {e}")))?;

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| YuleError::Gpu(format!("fence wait failed: {e}")))?;

            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(())
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: &ComputePipeline,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u8],
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        workgroup_count_z: u32,
    ) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[descriptor_set],
                &[],
            );

            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            self.device
                .cmd_dispatch(cmd, workgroup_count_x, workgroup_count_y, workgroup_count_z);
        }
    }

    /// Insert a compute → compute memory barrier.
    pub fn record_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Record a buffer-to-buffer copy.
    pub fn record_copy(&self, cmd: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, size: u64) {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            self.device.cmd_copy_buffer(cmd, src, dst, &[region]);
        }
    }

    /// Record a buffer-to-buffer copy with byte offsets.
    pub fn record_copy_offset(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        let region = vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        unsafe {
            self.device.cmd_copy_buffer(cmd, src, dst, &[region]);
        }
    }

    /// GPU-to-GPU buffer copy (no staging, no CPU).
    pub fn copy_buffer(
        &self,
        memory: &MemoryManager,
        src_handle: &BufferHandle,
        dst_handle: &BufferHandle,
        size: usize,
    ) -> Result<()> {
        let src_buf = memory.get_buffer(src_handle)?;
        let dst_buf = memory.get_buffer(dst_handle)?;
        let cmd = self.begin_command_buffer()?;
        self.record_copy(cmd, src_buf, dst_buf, size as u64);
        self.submit_and_wait(cmd)
    }

    /// GPU-to-GPU buffer copy with byte offsets.
    pub fn copy_buffer_offset(
        &self,
        memory: &MemoryManager,
        src_handle: &BufferHandle,
        dst_handle: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let src_buf = memory.get_buffer(src_handle)?;
        let dst_buf = memory.get_buffer(dst_handle)?;
        let cmd = self.begin_command_buffer()?;
        self.record_copy_offset(
            cmd,
            src_buf,
            dst_buf,
            src_offset as u64,
            dst_offset as u64,
            size as u64,
        );
        self.submit_and_wait(cmd)
    }

    /// Record a transfer → compute barrier.
    pub fn record_transfer_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Execute a one-shot copy of data to a device-local buffer via staging.
    pub fn copy_to_device(
        &self,
        memory: &MemoryManager,
        data: &[u8],
        dst_handle: &BufferHandle,
    ) -> Result<()> {
        let dst_buffer = memory.get_buffer(dst_handle)?;
        let (staging_buf, mut staging_alloc) = memory.allocate_staging(data.len())?;

        // Map staging and write
        let mapped = staging_alloc
            .mapped_slice_mut()
            .ok_or_else(|| YuleError::Gpu("failed to map staging buffer".into()))?;
        mapped[..data.len()].copy_from_slice(data);

        // Record copy
        let cmd = self.begin_command_buffer()?;
        self.record_copy(cmd, staging_buf, dst_buffer, data.len() as u64);
        self.submit_and_wait(cmd)?;

        // Clean up staging
        memory.free_staging(staging_buf, staging_alloc)?;
        Ok(())
    }

    /// Read data from a device-local buffer via staging.
    pub fn copy_from_device(
        &self,
        memory: &MemoryManager,
        src_handle: &BufferHandle,
        data: &mut [u8],
    ) -> Result<()> {
        let src_buffer = memory.get_buffer(src_handle)?;
        let (staging_buf, mut staging_alloc) = memory.allocate_staging(data.len())?;

        // Record copy
        let cmd = self.begin_command_buffer()?;
        self.record_copy(cmd, src_buffer, staging_buf, data.len() as u64);
        self.submit_and_wait(cmd)?;

        // Read from staging
        let mapped = staging_alloc
            .mapped_slice_mut()
            .ok_or_else(|| YuleError::Gpu("failed to map staging buffer".into()))?;
        data.copy_from_slice(&mapped[..data.len()]);

        memory.free_staging(staging_buf, staging_alloc)?;
        Ok(())
    }
}

impl Drop for CommandEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
