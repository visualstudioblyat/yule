use metal::{CommandQueue, ComputePipelineState, Device};
use std::collections::HashMap;
use yule_core::error::{Result, YuleError};

const MSL_SOURCE: &str = include_str!("../../kernels/metal/kernels.metal");

pub struct MetalPipelineManager {
    pipelines: HashMap<&'static str, ComputePipelineState>,
    command_queue: CommandQueue,
}

impl MetalPipelineManager {
    pub fn new(device: &Device) -> Result<Self> {
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| YuleError::Gpu(format!("MSL compilation failed: {e}")))?;

        let kernel_names: [&str; 12] = [
            "add_kernel",
            "silu_mul_kernel",
            "rms_norm_kernel",
            "softmax_kernel",
            "rope_kernel",
            "embed_lookup_kernel",
            "attn_score_kernel",
            "attn_value_kernel",
            "qmv_q4_0_kernel",
            "qmv_q8_0_kernel",
            "qmv_q4_k_kernel",
            "qmv_q6_k_kernel",
        ];

        let mut pipelines = HashMap::new();
        for name in &kernel_names {
            let func = library
                .get_function(name, None)
                .map_err(|e| YuleError::Gpu(format!("MSL function '{name}' not found: {e}")))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| YuleError::Gpu(format!("pipeline for '{name}' failed: {e}")))?;
            pipelines.insert(*name, pipeline);
        }

        let command_queue = device.new_command_queue();

        Ok(Self {
            pipelines,
            command_queue,
        })
    }

    pub fn get_pipeline(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipelines
            .get(name)
            .ok_or_else(|| YuleError::Gpu(format!("pipeline '{name}' not found")))
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }
}
