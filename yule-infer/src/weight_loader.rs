use std::collections::HashMap;
use yule_core::error::{Result, YuleError};
use yule_core::gguf::GgufFile;
use yule_core::model::ModelMetadata;
use yule_core::safetensors::SafetensorsFile;
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

    pub fn from_safetensors(
        st: &SafetensorsFile,
        meta: ModelMetadata,
        file_data: &'a [u8],
    ) -> Result<Self> {
        let mut tensors = HashMap::with_capacity(st.tensors.len());

        for tensor in &st.tensors {
            let end = tensor.offset as usize + tensor.size_bytes as usize;
            if end > file_data.len() {
                return Err(YuleError::Inference(format!(
                    "tensor '{}' out of bounds",
                    tensor.name
                )));
            }
            let data = &file_data[tensor.offset as usize..end];
            let gguf_name = translate_hf_name(&tensor.name);
            tensors.insert(
                gguf_name,
                TensorRef {
                    info: tensor.clone(),
                    data,
                },
            );
        }

        Ok(Self { meta, tensors })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

fn translate_hf_name(hf_name: &str) -> String {
    // Global tensors
    match hf_name {
        "model.embed_tokens.weight" | "transformer.wte.weight" => {
            return "token_embd.weight".to_string()
        }
        "model.norm.weight" | "transformer.ln_f.weight" => {
            return "output_norm.weight".to_string()
        }
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }

    // Layer tensors: model.layers.{N}.xxx -> blk.{N}.yyy
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let mapped = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
                "post_feedforward_layernorm.weight" => "ffn_post_norm.weight",
                "post_self_attn_layernorm.weight" => "attn_post_norm.weight",
                other => return format!("blk.{layer_num}.{other}"),
            };
            return format!("blk.{layer_num}.{mapped}");
        }
    }

    // Fallback: return as-is
    hf_name.to_string()
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
