use std::collections::HashMap;
use yule_core::error::{Result, YuleError};
use yule_core::gguf::GgufFile;
use yule_core::model::ModelMetadata;
use yule_core::tensor::TensorInfo;

pub struct WeightStore<'a> {
    pub meta: ModelMetadata,
    tensors: HashMap<String, TensorRef<'a>>,
}

struct TensorRef<'a> {
    pub info: TensorInfo,
    pub data: &'a [u8],
}

impl<'a> WeightStore<'a> {
    pub fn from_gguf(gguf: &GgufFile, file_data: &'a [u8]) -> Result<Self> {
        let loaded = gguf.to_loaded_model()?;
        let mut tensors = HashMap::with_capacity(loaded.tensors.len());

        for tensor in &loaded.tensors {
            let data = gguf.tensor_data(tensor, file_data)?;
            tensors.insert(
                tensor.name.clone(),
                TensorRef {
                    info: tensor.clone(),
                    data,
                },
            );
        }

        Ok(Self {
            meta: loaded.metadata,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        self.tensors.get(name).map(|t| (&t.info, t.data))
    }

    pub fn require(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        self.get(name)
            .ok_or_else(|| YuleError::Inference(format!("missing tensor: {name}")))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

/// Unified weight accessor for all Llama-family architectures.
/// Tensor naming is identical across Llama, Mistral, Phi, Qwen, Gemma in GGUF.
pub struct TransformerWeights<'a> {
    pub store: WeightStore<'a>,
}

pub type LlamaWeights<'a> = TransformerWeights<'a>;

impl<'a> TransformerWeights<'a> {
    pub fn new(store: WeightStore<'a>) -> Self {
        Self { store }
    }

    pub fn token_embd(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store.require("token_embd.weight")
    }

    pub fn output_norm(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store.require("output_norm.weight")
    }

    pub fn output(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require("output.weight")
            .or_else(|_| self.store.require("token_embd.weight"))
    }

    pub fn attn_norm(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_norm.weight"))
    }
    pub fn attn_q(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_q.weight"))
    }
    pub fn attn_k(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_k.weight"))
    }
    pub fn attn_v(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_v.weight"))
    }
    pub fn attn_output(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.attn_output.weight"))
    }
    pub fn ffn_norm(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ffn_norm.weight"))
    }
    pub fn ffn_gate(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ffn_gate.weight"))
    }
    pub fn ffn_up(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ffn_up.weight"))
    }
    pub fn ffn_down(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ffn_down.weight"))
    }

    // optional tensors for arch-specific features

    pub fn attn_q_bias(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store.get(&format!("blk.{layer}.attn_q.bias"))
    }
    pub fn attn_k_bias(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store.get(&format!("blk.{layer}.attn_k.bias"))
    }
    pub fn attn_v_bias(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store.get(&format!("blk.{layer}.attn_v.bias"))
    }
    pub fn attn_post_norm(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store
            .get(&format!("blk.{layer}.attn_post_norm.weight"))
    }
    pub fn ffn_post_norm(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store.get(&format!("blk.{layer}.ffn_post_norm.weight"))
    }
}
