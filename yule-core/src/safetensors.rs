use crate::dtype::DType;
use crate::error::{Result, YuleError};
use crate::model::{Architecture, LoadedModel, ModelFormat, ModelMetadata};
use crate::tensor::TensorInfo;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

const MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024; // 100MB header cap

/// Intermediate representation of a parsed SafeTensors file.
#[derive(Debug)]
pub struct SafetensorsFile {
    pub tensors: Vec<TensorInfo>,
    pub metadata: HashMap<String, String>,
    pub header_size: u64,
    pub file_size: u64,
}

/// HuggingFace config.json structure.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    pub model_type: Option<String>,
    pub hidden_size: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub vocab_size: Option<u32>,
    pub max_position_embeddings: Option<u32>,
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: Option<f32>,
    pub sliding_window: Option<u32>,
    pub num_local_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub intermediate_size: Option<u32>,
}

/// Raw tensor entry from the JSON header.
#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl SafetensorsFile {
    /// Convert this parsed file into a LoadedModel using HuggingFace config metadata.
    pub fn to_loaded_model(&self, config: &HfConfig) -> Result<LoadedModel> {
        let architecture = match config.model_type.as_deref() {
            Some("llama") => Architecture::Llama,
            Some("mistral") => Architecture::Mistral,
            Some("phi") | Some("phi3") => Architecture::Phi,
            Some("qwen2") => Architecture::Qwen,
            Some("gemma") | Some("gemma2") => Architecture::Gemma,
            Some("mixtral") => Architecture::Mixtral,
            Some("mamba") => Architecture::Mamba,
            Some("rwkv") | Some("rwkv5") | Some("rwkv6") => Architecture::Rwkv,
            Some(other) => Architecture::Unknown(other.to_string()),
            None => Architecture::Unknown("unknown".to_string()),
        };

        // Estimate parameter count from tensor sizes
        let parameters: u64 = self.tensors.iter().map(|t| t.num_elements()).sum();

        let metadata = ModelMetadata {
            architecture,
            name: self.metadata.get("model_name").cloned(),
            parameters,
            context_length: config.max_position_embeddings.unwrap_or(0),
            embedding_dim: config.hidden_size.unwrap_or(0),
            head_count: config.num_attention_heads.unwrap_or(0),
            head_count_kv: config.num_key_value_heads.unwrap_or(0),
            layer_count: config.num_hidden_layers.unwrap_or(0),
            vocab_size: config.vocab_size.unwrap_or(0),
            rope_freq_base: config.rope_theta,
            rope_scaling: None,
            expert_count: config.num_local_experts,
            expert_used_count: config.num_experts_per_tok,
            norm_eps: config.rms_norm_eps,
            sliding_window: config.sliding_window,
            partial_rotary_dim: None,
            logit_softcap: None,
            attn_logit_softcap: None,
        };

        Ok(LoadedModel {
            metadata,
            tensors: self.tensors.clone(),
            file_size: self.file_size,
            format: ModelFormat::Safetensors,
        })
    }
}

pub struct SafetensorsParser;

impl SafetensorsParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a SafeTensors file from disk.
    pub fn parse(&self, path: &Path) -> Result<SafetensorsFile> {
        let mut file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        // Read 8-byte header length
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf);

        Self::validate_header_size(header_len, file_size)?;

        // Read header JSON
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact(&mut header_buf)?;

        self.parse_header(&header_buf, header_len, file_size)
    }

    /// Parse a SafeTensors file from a byte slice (for testing).
    pub fn parse_bytes(&self, data: &[u8]) -> Result<SafetensorsFile> {
        if data.len() < 8 {
            return Err(YuleError::Parse(
                "safetensors file too small: need at least 8 bytes".into(),
            ));
        }

        let header_len = u64::from_le_bytes(data[..8].try_into().unwrap());
        let file_size = data.len() as u64;

        Self::validate_header_size(header_len, file_size)?;

        let header_end = 8 + header_len as usize;
        if header_end > data.len() {
            return Err(YuleError::Parse("header extends beyond file".into()));
        }

        let header_buf = &data[8..header_end];
        self.parse_header(header_buf, header_len, file_size)
    }

    fn validate_header_size(header_len: u64, file_size: u64) -> Result<()> {
        if header_len > MAX_HEADER_SIZE {
            return Err(YuleError::AllocationTooLarge {
                requested: header_len,
                max: MAX_HEADER_SIZE,
            });
        }
        if header_len + 8 > file_size {
            return Err(YuleError::Parse(format!(
                "header length ({header_len}) + 8 exceeds file size ({file_size})"
            )));
        }
        Ok(())
    }

    fn parse_header(
        &self,
        header_buf: &[u8],
        header_len: u64,
        file_size: u64,
    ) -> Result<SafetensorsFile> {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header_buf)?;

        let mut tensors = Vec::new();
        let mut metadata = HashMap::new();
        let data_base_offset = 8 + header_len;

        for (key, value) in &raw {
            if key == "__metadata__" {
                // Parse metadata as string->string map
                if let Some(obj) = value.as_object() {
                    for (mk, mv) in obj {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }

            let entry: RawTensorEntry = serde_json::from_value(value.clone())
                .map_err(|e| YuleError::Parse(format!("invalid tensor entry for '{key}': {e}")))?;

            let dtype = DType::from_safetensors_dtype(&entry.dtype).ok_or_else(|| {
                YuleError::Parse(format!(
                    "unrecognized dtype '{}' for tensor '{key}'",
                    entry.dtype
                ))
            })?;

            let [start, end] = entry.data_offsets;
            if end < start {
                return Err(YuleError::Parse(format!(
                    "tensor '{key}' has invalid data_offsets: end ({end}) < start ({start})"
                )));
            }

            let size_bytes = end - start;
            let absolute_offset = data_base_offset + start;

            // Check tensor data is within file bounds
            let absolute_end =
                data_base_offset
                    .checked_add(end)
                    .ok_or_else(|| YuleError::TensorOutOfBounds {
                        name: key.clone(),
                        offset: absolute_offset,
                        file_size,
                    })?;
            if absolute_end > file_size {
                return Err(YuleError::TensorOutOfBounds {
                    name: key.clone(),
                    offset: absolute_offset,
                    file_size,
                });
            }

            tensors.push(TensorInfo {
                name: key.clone(),
                dtype,
                shape: entry.shape,
                offset: absolute_offset,
                size_bytes,
            });
        }

        // Sort tensors by offset for overlap checking
        tensors.sort_by_key(|t| t.offset);

        // Validate no overlapping tensor data
        for i in 1..tensors.len() {
            let prev = &tensors[i - 1];
            let curr = &tensors[i];
            let prev_end = prev.offset + prev.size_bytes;
            if prev_end > curr.offset {
                return Err(YuleError::Parse(format!(
                    "overlapping tensor data: '{}' [{}..{}] overlaps with '{}' [{}..{}]",
                    prev.name,
                    prev.offset,
                    prev_end,
                    curr.name,
                    curr.offset,
                    curr.offset + curr.size_bytes
                )));
            }
        }

        Ok(SafetensorsFile {
            tensors,
            metadata,
            header_size: header_len,
            file_size,
        })
    }
}

impl Default for SafetensorsParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid safetensors file in memory.
    fn build_safetensors(header_json: &str, tensor_data: &[u8]) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(tensor_data);
        buf
    }

    #[test]
    fn test_parse_minimal_safetensors() {
        // A single F32 tensor of shape [2, 2] = 4 elements = 16 bytes
        let header = r#"{"test_tensor":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let tensor_data = vec![0u8; 16];
        let data = build_safetensors(header, &tensor_data);

        let parser = SafetensorsParser::new();
        let result = parser.parse_bytes(&data).unwrap();

        assert_eq!(result.tensors.len(), 1);
        assert_eq!(result.tensors[0].name, "test_tensor");
        assert_eq!(result.tensors[0].dtype, DType::F32);
        assert_eq!(result.tensors[0].shape, vec![2, 2]);
        assert_eq!(result.tensors[0].size_bytes, 16);
        assert_eq!(result.file_size, data.len() as u64);
    }

    #[test]
    fn test_reject_oversized_header() {
        // Craft a file that claims header_len > MAX_HEADER_SIZE
        let fake_len: u64 = MAX_HEADER_SIZE + 1;
        let mut data = Vec::new();
        data.extend_from_slice(&fake_len.to_le_bytes());
        // Add enough bytes so file_size check doesn't fail first
        data.resize(8 + fake_len as usize + 1, 0);

        let parser = SafetensorsParser::new();
        let result = parser.parse_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("allocation too large"), "got: {err}");
    }

    #[test]
    fn test_reject_overlapping_offsets() {
        // Two tensors whose data regions overlap
        let header = r#"{
            "tensor_a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
            "tensor_b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}
        }"#;
        let tensor_data = vec![0u8; 24];
        let data = build_safetensors(header, &tensor_data);

        let parser = SafetensorsParser::new();
        let result = parser.parse_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("overlapping"), "got: {err}");
    }

    #[test]
    fn test_dtype_mapping() {
        let cases = [
            ("F64", Some(DType::F64)),
            ("F32", Some(DType::F32)),
            ("F16", Some(DType::F16)),
            ("BF16", Some(DType::BF16)),
            ("I64", Some(DType::I64)),
            ("I32", Some(DType::I32)),
            ("I16", Some(DType::I16)),
            ("I8", Some(DType::I8)),
            ("Q4_0", None),
            ("INVALID", None),
        ];
        for (s, expected) in cases {
            assert_eq!(DType::from_safetensors_dtype(s), expected, "dtype: {s}");
        }
    }

    #[test]
    fn test_hf_config_parsing() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type.as_deref(), Some("llama"));
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.num_hidden_layers, Some(32));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.max_position_embeddings, Some(4096));
        assert_eq!(config.rope_theta, Some(10000.0));

        // Test conversion to loaded model
        let header = r#"{"w":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]}}"#;
        let data = build_safetensors(header, &[0u8; 32]);
        let parser = SafetensorsParser::new();
        let sf = parser.parse_bytes(&data).unwrap();
        let model = sf.to_loaded_model(&config).unwrap();
        assert!(matches!(model.metadata.architecture, Architecture::Llama));
        assert_eq!(model.metadata.embedding_dim, 4096);
        assert_eq!(model.metadata.layer_count, 32);
        assert_eq!(model.format, ModelFormat::Safetensors);
    }
}
