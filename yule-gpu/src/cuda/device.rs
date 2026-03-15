use cudarc::driver::CudaDevice;
use std::sync::Arc;
use yule_core::error::{Result, YuleError};

pub struct CudaDeviceWrapper {
    device: Arc<CudaDevice>,
    device_name: String,
    memory_bytes: u64,
    sm_count: u32,
}

impl CudaDeviceWrapper {
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| YuleError::Gpu(format!("cuda device init failed: {e}")))?;

        let device_name = device
            .name()
            .map_err(|e| YuleError::Gpu(format!("cuda device name query failed: {e}")))?;

        let (free, total) = device
            .mem_info()
            .map_err(|e| YuleError::Gpu(format!("cuda mem_info failed: {e}")))?;

        let sm_count = device
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| YuleError::Gpu(format!("cuda attribute query failed: {e}")))?
            as u32;

        tracing::info!(
            name = %device_name,
            total_mb = total / (1024 * 1024),
            free_mb = free / (1024 * 1024),
            sm_count,
            "cuda device selected"
        );

        Ok(Self {
            device,
            device_name,
            memory_bytes: total as u64,
            sm_count,
        })
    }

    pub fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    pub fn name(&self) -> String {
        self.device_name.clone()
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    pub fn inner(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| YuleError::Gpu(format!("cuda sync failed: {e}")))
    }
}
