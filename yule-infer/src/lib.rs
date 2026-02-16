pub mod attention;
#[cfg(feature = "vulkan")]
pub mod gpu_runner;
pub mod kv_cache;
pub mod model_runner;
pub mod sampler;
pub mod weight_loader;

use yule_core::error::Result;
use yule_core::model::LoadedModel;
use yule_gpu::ComputeBackend;

#[allow(dead_code)]
pub struct InferenceEngine {
    backend: Box<dyn ComputeBackend>,
    config: InferenceConfig,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_context_len: u32,
    pub batch_size: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_context_len: 4096,
            batch_size: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub sampling: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub min_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            min_p: 0.05,
        }
    }
}

impl InferenceEngine {
    pub fn new(backend: Box<dyn ComputeBackend>, config: InferenceConfig) -> Self {
        Self { backend, config }
    }

    pub fn load_model(&mut self, _model: &LoadedModel, _weights: &[u8]) -> Result<()> {
        // TODO: load weight tensors into compute backend buffers
        todo!("model loading into compute backend")
    }

    pub fn generate(&self, _request: &GenerateRequest) -> Result<Vec<u32>> {
        // TODO: autoregressive generation loop
        // 1. prefill: process all input tokens
        // 2. decode: generate one token at a time
        // 3. apply sampling
        // 4. check for EOS
        todo!("token generation")
    }
}
