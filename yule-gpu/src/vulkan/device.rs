use crate::BackendKind;
use crate::DeviceInfo;
use ash::vk;
use yule_core::error::{Result, YuleError};

pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub subgroup_size: u32,
    pub max_workgroup_size: u32,
    pub max_shared_memory: u32,
}

impl VulkanDevice {
    pub fn new() -> Result<Self> {
        // Load Vulkan at runtime (no link-time dependency)
        let entry = unsafe {
            ash::Entry::load().map_err(|e| YuleError::Gpu(format!("failed to load Vulkan: {e}")))?
        };

        // Create instance
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"yule")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"yule-gpu")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 1, 0)); // Vulkan 1.1 minimum

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create Vulkan instance: {e}")))?
        };

        // Select physical device — prefer discrete GPU
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| YuleError::Gpu(format!("failed to enumerate devices: {e}")))?
        };

        if physical_devices.is_empty() {
            return Err(YuleError::Gpu("no Vulkan devices found".into()));
        }

        let (physical_device, properties) = select_device(&instance, &physical_devices)?;

        let device_name = unsafe {
            std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        tracing::info!("vulkan device: {device_name}");

        // Find compute queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| YuleError::Gpu("no compute queue family found".into()))?
            as u32;

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let queue_create_infos = [queue_create_info];
        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create logical device: {e}")))?
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Query device limits
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Subgroup size (Vulkan 1.1)
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let subgroup_size = subgroup_props.subgroup_size;

        let max_workgroup_size = properties.limits.max_compute_work_group_invocations;
        let max_shared_memory = properties.limits.max_compute_shared_memory_size;

        tracing::info!(
            "  subgroup: {subgroup_size}, max_wg: {max_workgroup_size}, shared_mem: {} KiB",
            max_shared_memory / 1024
        );

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            properties,
            memory_properties,
            subgroup_size,
            max_workgroup_size,
            max_shared_memory,
        })
    }

    /// Quick probe: can we create a Vulkan instance and find a device?
    pub fn is_available() -> bool {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(_) => return false,
        };

        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(i) => i,
            Err(_) => return false,
        };

        let has_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map(|d| !d.is_empty())
                .unwrap_or(false)
        };

        unsafe { instance.destroy_instance(None) };
        has_devices
    }

    pub fn device_info(&self) -> DeviceInfo {
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(self.properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        // Find total device-local memory
        let mut total_vram = 0u64;
        for i in 0..self.memory_properties.memory_heap_count as usize {
            let heap = self.memory_properties.memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_vram += heap.size;
            }
        }

        DeviceInfo {
            name: device_name,
            backend: BackendKind::Vulkan,
            memory_bytes: total_vram,
            compute_units: 0, // not easily queryable in Vulkan core
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn select_device(
    instance: &ash::Instance,
    devices: &[vk::PhysicalDevice],
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
    // Prefer discrete GPU, then integrated, then any
    let mut best = None;
    let mut best_score = 0u32;

    for &pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };

        // Must have compute queue
        let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let has_compute = qf
            .iter()
            .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));

        if has_compute && score > best_score {
            best_score = score;
            best = Some((pd, props));
        }
    }

    best.ok_or_else(|| YuleError::Gpu("no suitable Vulkan device found".into()))
}
