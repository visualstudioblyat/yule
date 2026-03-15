use std::collections::HashMap;
use std::sync::Mutex;

use metal::Buffer as MTLBuffer;
use yule_core::error::{Result, YuleError};

use super::device::MetalDeviceWrapper;
use crate::BufferHandle;

pub struct MetalMemoryManager {
    buffers: Mutex<HashMap<u64, MTLBuffer>>,
    next_id: Mutex<u64>,
}

impl MetalMemoryManager {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
        }
    }

    pub fn allocate(&self, device: &MetalDeviceWrapper, size_bytes: usize) -> Result<BufferHandle> {
        let buf = device.inner().new_buffer(
            size_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut id = self.next_id.lock().unwrap();
        let handle = BufferHandle(*id);
        *id += 1;

        self.buffers.lock().unwrap().insert(handle.0, buf);
        Ok(handle)
    }

    pub fn free(&self, handle: BufferHandle) -> Result<()> {
        self.buffers
            .lock()
            .unwrap()
            .remove(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("metal free: unknown buffer {}", handle.0)))?;
        Ok(())
    }

    pub fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("metal copy_to: unknown buffer {}", handle.0)))?;

        // Metal shared memory: direct memcpy via contents() pointer
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    pub fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(&handle.0).ok_or_else(|| {
            YuleError::Gpu(format!("metal copy_from: unknown buffer {}", handle.0))
        })?;

        let ptr = buf.contents() as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    pub fn copy_buffer(
        &self,
        device: &MetalDeviceWrapper,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let src_buf = buffers
            .get(&src.0)
            .ok_or_else(|| YuleError::Gpu(format!("metal copy_buf: unknown src {}", src.0)))?;
        let dst_buf = buffers
            .get(&dst.0)
            .ok_or_else(|| YuleError::Gpu(format!("metal copy_buf: unknown dst {}", dst.0)))?;

        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let blit = cmd_buf.new_blit_command_encoder();

        blit.copy_from_buffer(
            src_buf,
            src_offset as u64,
            dst_buf,
            dst_offset as u64,
            size as u64,
        );
        blit.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(())
    }

    pub fn get_buffer(&self, handle: &BufferHandle) -> Result<&MTLBuffer> {
        // This won't work with Mutex — callers should use a scoped lock instead.
        // For now, kernel dispatch will lock and extract what it needs.
        Err(YuleError::Gpu("use scoped buffer access instead".into()))
    }
}
