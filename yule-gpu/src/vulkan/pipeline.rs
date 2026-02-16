use ash::vk;
use std::collections::HashMap;
use yule_core::error::{Result, YuleError};

/// Identifies a compute shader variant.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ShaderKey {
    Add,
    SiluMul,
    RmsNorm,
    Rope,
    Softmax,
    EmbedLookup,
    AttnScore,
    AttnValue,
    QmvQ4_0,
    QmvQ4K,
    QmvQ6K,
    QmvQ8_0,
}

pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub struct PipelineManager {
    device: ash::Device,
    pipelines: HashMap<ShaderKey, ComputePipeline>,
    descriptor_pool: vk::DescriptorPool,
}

/// Descriptor layout config for a shader.
pub struct ShaderLayout {
    pub n_buffers: u32,
    pub push_constant_size: u32,
}

impl PipelineManager {
    pub fn new(device: &ash::Device) -> Result<Self> {
        // Create a large descriptor pool
        // Batched forward pass records all layers into one command buffer.
        // Each layer needs ~107 descriptor sets (32-head attention dominates).
        // 22 layers × 107 ≈ 2,354 sets. Size for up to 80 layers (70B models).
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 32768,
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(8192)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create descriptor pool: {e}")))?
        };

        Ok(Self {
            device: device.clone(),
            pipelines: HashMap::new(),
            descriptor_pool,
        })
    }

    /// Register a shader from pre-compiled SPIR-V bytes.
    pub fn register_shader(
        &mut self,
        key: ShaderKey,
        spirv: &[u8],
        layout: ShaderLayout,
    ) -> Result<()> {
        // Ensure SPIR-V is aligned to u32
        assert!(spirv.len() % 4 == 0, "SPIR-V must be 4-byte aligned");
        let spirv_u32: &[u32] =
            unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) };

        let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create shader module: {e}")))?
        };

        // Descriptor set layout
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..layout.n_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(|e| {
                    YuleError::Gpu(format!("failed to create descriptor set layout: {e}"))
                })?
        };

        // Pipeline layout (push constants)
        let push_constant_ranges = if layout.push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: layout.push_constant_size,
            }]
        } else {
            vec![]
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| YuleError::Gpu(format!("failed to create pipeline layout: {e}")))?
        };

        // Compute pipeline
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| YuleError::Gpu(format!("failed to create compute pipeline: {e:?}")))?
                [0]
        };

        // Clean up shader module (pipeline owns the compiled code)
        unsafe { self.device.destroy_shader_module(shader_module, None) };

        self.pipelines.insert(
            key,
            ComputePipeline {
                pipeline,
                layout: pipeline_layout,
                descriptor_set_layout,
            },
        );

        Ok(())
    }

    /// Reset the descriptor pool, freeing all allocated descriptor sets.
    /// Safe to call after the GPU has finished executing all commands that use them.
    pub fn reset_descriptor_pool(&self) -> Result<()> {
        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| YuleError::Gpu(format!("failed to reset descriptor pool: {e}")))?;
        }
        Ok(())
    }

    /// Get a pipeline by key.
    pub fn get(&self, key: ShaderKey) -> Result<&ComputePipeline> {
        self.pipelines
            .get(&key)
            .ok_or_else(|| YuleError::Gpu(format!("pipeline {key:?} not registered")))
    }

    /// Allocate a descriptor set for a pipeline.
    pub fn allocate_descriptor_set(&self, key: ShaderKey) -> Result<vk::DescriptorSet> {
        let pipeline = self.get(key)?;
        let layouts = [pipeline.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| YuleError::Gpu(format!("failed to allocate descriptor set: {e}")))?
        };

        Ok(sets[0])
    }

    /// Register all standard shaders from embedded SPIR-V.
    pub fn register_all_shaders(&mut self) -> Result<()> {
        // Each shader: (key, spirv_bytes, n_buffers, push_constant_size)
        let shaders: Vec<(ShaderKey, &[u8], u32, u32)> = vec![
            (
                ShaderKey::Add,
                include_bytes!("../../shaders/compiled/add.spv"),
                3,
                4,
            ),
            (
                ShaderKey::SiluMul,
                include_bytes!("../../shaders/compiled/silu_mul.spv"),
                3,
                4,
            ),
            (
                ShaderKey::RmsNorm,
                include_bytes!("../../shaders/compiled/rms_norm.spv"),
                3,
                8,
            ),
            (
                ShaderKey::Rope,
                include_bytes!("../../shaders/compiled/rope.spv"),
                4,
                20,
            ),
            (
                ShaderKey::Softmax,
                include_bytes!("../../shaders/compiled/softmax.spv"),
                2,
                4,
            ),
            (
                ShaderKey::EmbedLookup,
                include_bytes!("../../shaders/compiled/embed_lookup.spv"),
                3,
                8,
            ),
            (
                ShaderKey::AttnScore,
                include_bytes!("../../shaders/compiled/attn_score.spv"),
                4,
                20,
            ),
            (
                ShaderKey::AttnValue,
                include_bytes!("../../shaders/compiled/attn_value.spv"),
                4,
                20,
            ),
            (
                ShaderKey::QmvQ4_0,
                include_bytes!("../../shaders/compiled/qmv_q4_0.spv"),
                3,
                12,
            ),
            (
                ShaderKey::QmvQ4K,
                include_bytes!("../../shaders/compiled/qmv_q4_k.spv"),
                3,
                12,
            ),
            (
                ShaderKey::QmvQ6K,
                include_bytes!("../../shaders/compiled/qmv_q6_k.spv"),
                3,
                12,
            ),
            (
                ShaderKey::QmvQ8_0,
                include_bytes!("../../shaders/compiled/qmv_q8_0.spv"),
                3,
                12,
            ),
        ];

        for (key, spirv, n_buffers, push_size) in shaders {
            self.register_shader(
                key,
                spirv,
                ShaderLayout {
                    n_buffers,
                    push_constant_size: push_size,
                },
            )?;
        }

        Ok(())
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        unsafe {
            for (_, cp) in self.pipelines.drain() {
                self.device.destroy_pipeline(cp.pipeline, None);
                self.device.destroy_pipeline_layout(cp.layout, None);
                self.device
                    .destroy_descriptor_set_layout(cp.descriptor_set_layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
