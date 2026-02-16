use crate::tensor::TensorInfo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Mistral,
    Phi,
    Qwen,
    Gemma,
    Mixtral,
    Mamba,
    Unknown(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: Architecture,
    pub name: Option<String>,
    pub parameters: u64,
    pub context_length: u32,
    pub embedding_dim: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub layer_count: u32,
    pub vocab_size: u32,
    pub rope_freq_base: Option<f64>,
    pub rope_scaling: Option<String>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub norm_eps: Option<f32>,
    pub sliding_window: Option<u32>,
    pub partial_rotary_dim: Option<u32>,
    pub logit_softcap: Option<f32>,
    pub attn_logit_softcap: Option<f32>,
}

#[derive(Debug)]
pub struct LoadedModel {
    pub metadata: ModelMetadata,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
    pub format: ModelFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}
