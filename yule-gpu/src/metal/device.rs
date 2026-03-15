use metal::Device;
use yule_core::error::{Result, YuleError};

pub struct MetalDeviceWrapper {
    device: Device,
    device_name: String,
}

impl MetalDeviceWrapper {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| YuleError::Gpu("no Metal GPU device found".into()))?;

        let device_name = device.name().to_string();

        tracing::info!(
            name = %device_name,
            unified_memory = device.has_unified_memory(),
            "metal device selected"
        );

        Ok(Self {
            device,
            device_name,
        })
    }

    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    pub fn name(&self) -> String {
        self.device_name.clone()
    }

    pub fn memory_bytes(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }

    pub fn max_threads_per_threadgroup(&self) -> u32 {
        self.device.max_threads_per_threadgroup().width as u32
    }

    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    pub fn inner(&self) -> &Device {
        &self.device
    }

    pub fn new_command_queue(&self) -> metal::CommandQueue {
        self.device.new_command_queue()
    }
}
