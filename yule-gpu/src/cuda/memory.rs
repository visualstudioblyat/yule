use std::collections::HashMap;
use std::sync::Mutex;

use cudarc::driver::CudaSlice;
use yule_core::error::{Result, YuleError};

use crate::BufferHandle;
use super::device::CudaDeviceWrapper;

pub struct CudaMemoryManager {
    buffers: Mutex<HashMap<u64, CudaSlice<u8>>>,
    next_id: Mutex<u64>,
}

impl CudaMemoryManager {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
        }
    }

    pub fn allocate(&self, device: &CudaDeviceWrapper, size_bytes: usize) -> Result<BufferHandle> {
        let buf: CudaSlice<u8> = device
            .inner()
            .alloc_zeros(size_bytes)
            .map_err(|e| YuleError::Gpu(format!("cuda alloc failed: {e}")))?;

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
            .ok_or_else(|| YuleError::Gpu(format!("cuda free: unknown buffer {}", handle.0)))?;
        Ok(())
    }

    pub fn copy_to_device(
        &self,
        device: &CudaDeviceWrapper,
        data: &[u8],
        handle: &BufferHandle,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get_mut(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("cuda copy_to: unknown buffer {}", handle.0)))?;

        device
            .inner()
            .htod_copy_into(data.to_vec(), buf)
            .map_err(|e| YuleError::Gpu(format!("cuda htod failed: {e}")))?;
        Ok(())
    }

    pub fn copy_from_device(
        &self,
        device: &CudaDeviceWrapper,
        handle: &BufferHandle,
        data: &mut [u8],
    ) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("cuda copy_from: unknown buffer {}", handle.0)))?;

        let host = device
            .inner()
            .dtoh_sync_copy(buf)
            .map_err(|e| YuleError::Gpu(format!("cuda dtoh failed: {e}")))?;

        let len = data.len().min(host.len());
        data[..len].copy_from_slice(&host[..len]);
        Ok(())
    }

    pub fn copy_buffer(
        &self,
        device: &CudaDeviceWrapper,
        src: &BufferHandle,
        dst: &BufferHandle,
        size: usize,
    ) -> Result<()> {
        self.copy_buffer_offset(device, src, dst, 0, 0, size)
    }

    pub fn copy_buffer_offset(
        &self,
        _device: &CudaDeviceWrapper,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();

        // read src into host buffer first (can't borrow both mutably at once)
        let src_buf = buffers
            .get(&src.0)
            .ok_or_else(|| YuleError::Gpu(format!("cuda copy_buf: unknown src {}", src.0)))?;

        let src_slice = src_buf.slice(src_offset..src_offset + size);

        let dst_buf = buffers
            .get_mut(&dst.0)
            .ok_or_else(|| YuleError::Gpu(format!("cuda copy_buf: unknown dst {}", dst.0)))?;

        let mut dst_slice = dst_buf.slice_mut(dst_offset..dst_offset + size);

        // device-to-device copy via cuMemcpyDtoD
        use cudarc::driver::DeviceRepr;
        unsafe {
            cudarc::driver::sys::lib()
                .cuMemcpyDtoD_v2(
                    *dst_slice.device_ptr() as u64,
                    *src_slice.device_ptr() as u64,
                    size,
                );
        }
        Ok(())
    }
}
