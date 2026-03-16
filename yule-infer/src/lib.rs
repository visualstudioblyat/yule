pub mod attention;
#[cfg(feature = "vulkan")]
pub mod gpu_runner;
pub mod kv_cache;
pub mod mamba;
pub mod model_runner;
pub mod rwkv;
pub mod sampler;
pub mod speculative;
pub mod weight_loader;

use yule_core::error::Result;
use yule_core::model::LoadedModel;
use yule_gpu::ComputeBackend;

use crate::model_runner::ModelRunner;

pub struct InferenceEngine {
    backend: Box<dyn ComputeBackend>,
    config: InferenceConfig,
    // runner MUST be declared before weight_data so it gets dropped first
    // (runner borrows from weight_data; Rust drops fields in declaration order)
    runner: Option<Box<dyn ModelRunner>>,
    weight_data: Option<Vec<u8>>,
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
    pub eos_token: Option<u32>,
    pub speculative: Option<crate::speculative::SpeculativeConfig>,
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
    pub fn device_info(&self) -> yule_gpu::DeviceInfo {
        self.backend.device_info()
    }

    pub fn new(backend: Box<dyn ComputeBackend>, config: InferenceConfig) -> Self {
        Self {
            backend,
            config,
            runner: None,
            weight_data: None,
        }
    }

    pub fn load_model(&mut self, _model: &LoadedModel, weights: &[u8]) -> Result<()> {
        use crate::model_runner::TransformerRunner;
        use crate::weight_loader::{TransformerWeights, WeightStore};
        use yule_core::gguf::GgufParser;

        // Own the weight data
        self.weight_data = Some(weights.to_vec());

        // SAFETY: weight_data lives in self and runner is dropped before weight_data
        // (field declaration order guarantees drop order in Rust)
        let static_ref: &'static [u8] = unsafe {
            std::mem::transmute::<&[u8], &'static [u8]>(
                self.weight_data.as_ref().unwrap().as_slice(),
            )
        };

        // Parse GGUF
        let parser = GgufParser::new();
        let gguf = parser.parse_bytes(static_ref, static_ref.len() as u64)?;

        // Build weight store and runner
        let store = WeightStore::from_gguf(&gguf, static_ref)?;
        let weights = TransformerWeights::new(store);

        // Create runner based on backend
        let runner: Box<dyn ModelRunner> = {
            #[cfg(feature = "vulkan")]
            {
                use yule_gpu::BackendKind;
                if self.backend.device_info().backend == BackendKind::Vulkan {
                    Box::new(crate::gpu_runner::GpuTransformerRunner::new(weights)?)
                } else {
                    Box::new(TransformerRunner::new(weights)?)
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Box::new(TransformerRunner::new(weights)?)
            }
        };

        self.runner = Some(runner);
        Ok(())
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> Result<Vec<u32>> {
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| yule_core::error::YuleError::Inference("model not loaded".into()))?;

        runner.reset();

        // Validate input
        if request.tokens.is_empty() {
            return Err(yule_core::error::YuleError::Inference(
                "empty input tokens".into(),
            ));
        }
        let total_len = request.tokens.len() as u32 + request.max_new_tokens;
        if total_len > self.config.max_context_len {
            return Err(yule_core::error::YuleError::Inference(format!(
                "requested {} tokens exceeds max context length {}",
                total_len, self.config.max_context_len
            )));
        }

        // Prefill input tokens
        let mut logits = runner.prefill(&request.tokens)?;

        // Create sampler
        let sampler = crate::sampler::Sampler::new(request.sampling.clone());

        // Decode loop — track all tokens for repetition penalty
        let mut all_tokens = request.tokens.clone();
        let mut output_tokens = Vec::with_capacity(request.max_new_tokens as usize);

        for _ in 0..request.max_new_tokens {
            let token = sampler.sample_with_history(&logits, &all_tokens)?;

            // EOS check
            if let Some(eos) = request.eos_token {
                if token == eos {
                    break;
                }
            }

            output_tokens.push(token);
            all_tokens.push(token);
            logits = runner.decode_step(token)?;
        }

        Ok(output_tokens)
    }
}
