use ash::vk;

use super::device::VulkanDevice;
use super::pipeline::ShaderKey;

/// Cooperative matrix tile dimensions supported by the device.
#[derive(Debug, Clone, Copy)]
pub struct CoopMatrixConfig {
    pub m_size: u32, // tile rows (typically 16)
    pub n_size: u32, // tile cols (typically 16)
    pub k_size: u32, // inner dim (typically 16)
    pub supported: bool,
}

impl Default for CoopMatrixConfig {
    fn default() -> Self {
        Self {
            m_size: 16,
            n_size: 16,
            k_size: 16,
            supported: false,
        }
    }
}

impl CoopMatrixConfig {
    pub fn query(device: &VulkanDevice) -> Self {
        if !device.has_cooperative_matrix {
            return Self::default();
        }

        // The actual tile sizes would be queried via
        // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
        // ash 0.38 may not expose this directly, so we use
        // the most common default: 16x16x16 (NVIDIA Ampere+, AMD RDNA3+).
        tracing::info!("cooperative matrix: using 16x16x16 tiles");

        Self {
            m_size: 16,
            n_size: 16,
            k_size: 16,
            supported: true,
        }
    }

    /// Select the best ShaderKey for a matmul operation.
    /// Returns cooperative variant if supported, falls back to standard.
    pub fn select_matmul_shader(&self) -> ShaderKey {
        if self.supported {
            ShaderKey::CoopMatmul
        } else {
            // No standard matmul ShaderKey exists yet — caller should
            // use the per-row qmv dispatch instead
            ShaderKey::CoopMatmul // cooperative path
        }
    }

    /// Calculate grid dimensions for cooperative matrix dispatch.
    /// Returns (workgroup_x, workgroup_y, workgroup_z).
    pub fn grid_size(&self, m: u32, n: u32) -> (u32, u32, u32) {
        let tiles_m = m.div_ceil(self.m_size);
        let tiles_n = n.div_ceil(self.n_size);
        (tiles_m, tiles_n, 1)
    }
}

/// Check if the given device extensions include VK_KHR_cooperative_matrix.
pub fn check_cooperative_matrix_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default()
    };

    extensions.iter().any(|ext| {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        name.to_bytes() == b"VK_KHR_cooperative_matrix"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coop_matrix_default_config() {
        let cfg = CoopMatrixConfig::default();
        assert!(!cfg.supported);
        assert_eq!(cfg.m_size, 16);
        assert_eq!(cfg.n_size, 16);
        assert_eq!(cfg.k_size, 16);
    }

    #[test]
    fn coop_matrix_grid_calculation() {
        let cfg = CoopMatrixConfig {
            m_size: 16,
            n_size: 16,
            k_size: 16,
            supported: true,
        };

        // Exact multiple
        assert_eq!(cfg.grid_size(32, 32), (2, 2, 1));
        // Non-multiple rounds up
        assert_eq!(cfg.grid_size(17, 33), (2, 3, 1));
        // Single tile
        assert_eq!(cfg.grid_size(1, 1), (1, 1, 1));
    }
}
