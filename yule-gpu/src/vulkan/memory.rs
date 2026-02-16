use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use std::collections::HashMap;
use std::sync::Mutex;
use yule_core::error::{Result, YuleError};

use super::device::VulkanDevice;
use crate::BufferHandle;
use crate::buffer::next_buffer_handle;

pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: usize,
}

pub struct MemoryManager {
    allocator: Mutex<Allocator>,
    buffers: Mutex<HashMap<u64, VkBuffer>>,
    device: ash::Device,
}

impl MemoryManager {
    pub fn new(vk_dev: &VulkanDevice) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: vk_dev.instance.clone(),
            device: vk_dev.device.clone(),
            physical_device: vk_dev.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| YuleError::Gpu(format!("failed to create allocator: {e}")))?;

        Ok(Self {
            allocator: Mutex::new(allocator),
            buffers: Mutex::new(HashMap::new()),
            device: vk_dev.device.clone(),
        })
    }

    /// Allocate a device-local buffer.
    pub fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create buffer: {e}")))?
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "yule_buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| YuleError::Gpu(format!("failed to allocate: {e}")))?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| YuleError::Gpu(format!("failed to bind buffer memory: {e}")))?;
        }

        let handle = next_buffer_handle();
        self.buffers.lock().unwrap().insert(
            handle.0,
            VkBuffer {
                buffer,
                allocation,
                size: size_bytes,
            },
        );

        Ok(handle)
    }

    /// Allocate a host-visible staging buffer.
    pub fn allocate_staging(&self, size_bytes: usize) -> Result<(vk::Buffer, Allocation)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create staging buffer: {e}")))?
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "yule_staging",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| YuleError::Gpu(format!("failed to allocate staging: {e}")))?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| YuleError::Gpu(format!("failed to bind staging memory: {e}")))?;
        }

        Ok((buffer, allocation))
    }

    pub fn free(&self, handle: BufferHandle) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(vk_buf) = buffers.remove(&handle.0) {
            unsafe { self.device.destroy_buffer(vk_buf.buffer, None) };
            self.allocator
                .lock()
                .unwrap()
                .free(vk_buf.allocation)
                .map_err(|e| YuleError::Gpu(format!("failed to free: {e}")))?;
        }
        Ok(())
    }

    pub fn free_staging(&self, buffer: vk::Buffer, allocation: Allocation) -> Result<()> {
        unsafe { self.device.destroy_buffer(buffer, None) };
        self.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .map_err(|e| YuleError::Gpu(format!("failed to free staging: {e}")))?;
        Ok(())
    }

    /// Get the Vulkan buffer for a handle.
    pub fn get_buffer(&self, handle: &BufferHandle) -> Result<vk::Buffer> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.0)
            .map(|b| b.buffer)
            .ok_or_else(|| YuleError::Gpu(format!("buffer handle {} not found", handle.0)))
    }

    /// Get the size of a buffer.
    pub fn get_size(&self, handle: &BufferHandle) -> Result<usize> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.0)
            .map(|b| b.size)
            .ok_or_else(|| YuleError::Gpu(format!("buffer handle {} not found", handle.0)))
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        // Free all remaining buffers
        let mut buffers = self.buffers.lock().unwrap();
        let mut allocator = self.allocator.lock().unwrap();
        for (_, vk_buf) in buffers.drain() {
            unsafe { self.device.destroy_buffer(vk_buf.buffer, None) };
            let _ = allocator.free(vk_buf.allocation);
        }
    }
}
