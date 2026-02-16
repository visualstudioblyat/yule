use crate::dtype::DType;
use crate::error::{Result, YuleError};
use crate::model::{Architecture, LoadedModel, ModelFormat, ModelMetadata};
use crate::tensor::TensorInfo;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const GGUF_VERSION_3: u32 = 3;
const MAX_STRING_LEN: u64 = 1024 * 1024; // 1MB — prevents DoS via CVE-2025-66960
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_KV_COUNT: u64 = 100_000;
const MAX_ARRAY_LEN: u64 = 10_000_000; // 10M elements
const MAX_DIMENSIONS: u32 = 8; // no tensor should have more than 8 dims
const HEADER_MIN_SIZE: usize = 4 + 4 + 8 + 8; // magic + version + tensor_count + kv_count

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint8(v) => Some(*v as u64),
            GgufValue::Uint16(v) => Some(*v as u64),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::Float64(v) => Some(*v),
            GgufValue::Float32(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Safe cursor into a byte slice — all reads are bounds-checked.
/// This is the core defense against parser exploits.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(YuleError::Parse(format!(
                "unexpected EOF at offset {}: need {n} bytes, have {}",
                self.pos,
                self.remaining()
            )))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        self.ensure(2)?;
        let v = i16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.ensure(4)?;
        let v = i32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.ensure(8)?;
        let v = i64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    /// Read a GGUF string: u64 length prefix + UTF-8 bytes.
    /// Length is bounds-checked against MAX_STRING_LEN before allocation.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_LEN {
            return Err(YuleError::StringTooLong {
                len,
                max: MAX_STRING_LEN,
            });
        }
        let len = len as usize;
        self.ensure(len)?;
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| YuleError::Parse(format!("invalid UTF-8 in string: {e}")))
    }
}

pub struct GgufParser;

impl GgufParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a GGUF file from disk using memory-mapped I/O.
    pub fn parse_file(&self, path: &Path) -> Result<GgufFile> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        if file_size < HEADER_MIN_SIZE as u64 {
            return Err(YuleError::Parse("file too small to be valid GGUF".into()));
        }

        // SAFETY: we open the file read-only and hold it open for the duration.
        // SIGBUS risk if file is truncated externally — we accept this and handle
        // via bounds-checked reads in the Cursor.
        let mmap = unsafe { Mmap::map(&file)? };

        self.parse_bytes(&mmap, file_size)
    }

    /// Parse GGUF from a byte slice. Core parsing logic — all reads go through Cursor.
    pub fn parse_bytes(&self, data: &[u8], file_size: u64) -> Result<GgufFile> {
        let mut cursor = Cursor::new(data);

        // 1. magic number
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(YuleError::InvalidMagic {
                expected: GGUF_MAGIC,
                got: magic,
            });
        }

        // 2. version
        let version = cursor.read_u32()?;
        if version != GGUF_VERSION_3 {
            return Err(YuleError::UnsupportedVersion(version));
        }

        // 3. tensor count — bounds-checked before we allocate anything
        let tensor_count = cursor.read_u64()?;
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(YuleError::Parse(format!(
                "tensor count {tensor_count} exceeds maximum {MAX_TENSOR_COUNT}"
            )));
        }

        // 4. metadata KV count
        let kv_count = cursor.read_u64()?;
        if kv_count > MAX_KV_COUNT {
            return Err(YuleError::Parse(format!(
                "KV count {kv_count} exceeds maximum {MAX_KV_COUNT}"
            )));
        }

        // 5. read metadata KV pairs
        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for i in 0..kv_count {
            let key = cursor
                .read_string()
                .map_err(|e| YuleError::Parse(format!("failed to read KV key {i}: {e}")))?;
            let value = Self::read_value(&mut cursor).map_err(|e| {
                YuleError::Parse(format!("failed to read KV value for '{key}': {e}"))
            })?;
            metadata.insert(key, value);
        }

        // 6. read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = cursor
                .read_string()
                .map_err(|e| YuleError::Parse(format!("failed to read tensor {i} name: {e}")))?;

            let n_dims = cursor.read_u32()?;
            if n_dims > MAX_DIMENSIONS {
                return Err(YuleError::Parse(format!(
                    "tensor '{name}' has {n_dims} dimensions, max is {MAX_DIMENSIONS}"
                )));
            }

            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64()?);
            }

            let dtype_raw = cursor.read_u32()?;
            let dtype = gguf_dtype_to_dtype(dtype_raw)?;

            let offset = cursor.read_u64()?;

            // compute tensor size in bytes
            let num_elements: u64 = shape.iter().copied().try_fold(1u64, |acc, dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| YuleError::InvalidTensorShape {
                        name: name.clone(),
                        reason: "shape overflow".into(),
                    })
            })?;

            let size_bytes = if dtype.is_quantized() {
                let block_size = dtype.block_size() as u64;
                let num_blocks = num_elements.div_ceil(block_size);
                num_blocks * dtype.size_of_block() as u64
            } else {
                num_elements * dtype.size_of_block() as u64
            };

            let info = TensorInfo {
                name,
                dtype,
                shape,
                offset,
                size_bytes,
            };

            tensors.push(info);
        }

        // 7. the data section starts at an aligned offset after header
        // GGUF aligns tensor data to the nearest multiple of the alignment value
        // (default 32 bytes)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;

        let header_end = cursor.pos as u64;
        let data_offset = header_end.div_ceil(alignment) * alignment;

        // 8. validate all tensor offsets are within file bounds
        for tensor in &tensors {
            let absolute_offset = data_offset.checked_add(tensor.offset).ok_or_else(|| {
                YuleError::TensorOutOfBounds {
                    name: tensor.name.clone(),
                    offset: tensor.offset,
                    file_size,
                }
            })?;
            let end = absolute_offset
                .checked_add(tensor.size_bytes)
                .ok_or_else(|| YuleError::TensorOutOfBounds {
                    name: tensor.name.clone(),
                    offset: tensor.offset,
                    file_size,
                })?;
            if end > file_size {
                return Err(YuleError::TensorOutOfBounds {
                    name: tensor.name.clone(),
                    offset: absolute_offset,
                    file_size,
                });
            }
        }

        // 9. check for overlapping tensors (potential exploit vector)
        let mut sorted: Vec<(u64, u64, &str)> = tensors
            .iter()
            .map(|t| (t.offset, t.size_bytes, t.name.as_str()))
            .collect();
        sorted.sort_by_key(|&(offset, _, _)| offset);

        for window in sorted.windows(2) {
            let (offset_a, size_a, name_a) = window[0];
            let (offset_b, _, name_b) = window[1];
            if offset_a + size_a > offset_b {
                return Err(YuleError::Parse(format!(
                    "overlapping tensors: '{name_a}' [{offset_a}..{}] overlaps '{name_b}' [{offset_b}..]",
                    offset_a + size_a
                )));
            }
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
            file_size,
            alignment,
        })
    }

    /// Read a typed value from the cursor.
    fn read_value(cursor: &mut Cursor) -> Result<GgufValue> {
        let value_type = cursor.read_u32()?;
        Self::read_typed_value(cursor, value_type)
    }

    fn read_typed_value(cursor: &mut Cursor, value_type: u32) -> Result<GgufValue> {
        match value_type {
            GGUF_TYPE_UINT8 => Ok(GgufValue::Uint8(cursor.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(GgufValue::Int8(cursor.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(GgufValue::Uint16(cursor.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(GgufValue::Int16(cursor.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(GgufValue::Uint32(cursor.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(GgufValue::Int32(cursor.read_i32()?)),
            GGUF_TYPE_FLOAT32 => Ok(GgufValue::Float32(cursor.read_f32()?)),
            GGUF_TYPE_BOOL => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
            GGUF_TYPE_STRING => Ok(GgufValue::String(cursor.read_string()?)),
            GGUF_TYPE_ARRAY => {
                let elem_type = cursor.read_u32()?;
                let len = cursor.read_u64()?;
                if len > MAX_ARRAY_LEN {
                    return Err(YuleError::Parse(format!(
                        "array length {len} exceeds maximum {MAX_ARRAY_LEN}"
                    )));
                }
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(Self::read_typed_value(cursor, elem_type)?);
                }
                Ok(GgufValue::Array(values))
            }
            GGUF_TYPE_UINT64 => Ok(GgufValue::Uint64(cursor.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(GgufValue::Int64(cursor.read_i64()?)),
            GGUF_TYPE_FLOAT64 => Ok(GgufValue::Float64(cursor.read_f64()?)),
            other => Err(YuleError::Parse(format!(
                "unknown GGUF value type: {other}"
            ))),
        }
    }
}

impl Default for GgufParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed GGUF file — metadata + tensor layout + data offset.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub file_size: u64,
    pub alignment: u64,
}

impl GgufFile {
    /// Extract structured model metadata from GGUF key-value pairs.
    pub fn to_loaded_model(&self) -> Result<LoadedModel> {
        let arch_str = self.get_str("general.architecture").unwrap_or("unknown");
        let architecture = match arch_str {
            "llama" => Architecture::Llama,
            "mistral" => Architecture::Mistral,
            "phi" | "phi2" | "phi3" => Architecture::Phi,
            "qwen" | "qwen2" => Architecture::Qwen,
            "gemma" | "gemma2" => Architecture::Gemma,
            "mixtral" => Architecture::Mixtral,
            "mamba" => Architecture::Mamba,
            other => Architecture::Unknown(other.to_string()),
        };

        // architecture-prefixed keys (e.g., "llama.context_length")
        let prefix = arch_str;

        let context_length = self
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(4096);

        let embedding_dim = self
            .get_u32(&format!("{prefix}.embedding_length"))
            .unwrap_or(0);

        let head_count = self
            .get_u32(&format!("{prefix}.attention.head_count"))
            .unwrap_or(0);

        let head_count_kv = self
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(head_count);

        let layer_count = self.get_u32(&format!("{prefix}.block_count")).unwrap_or(0);

        let vocab_size = self
            .get_u32("tokenizer.ggml.vocab_size")
            .or_else(|| {
                // fallback: count tokens in the token list
                self.metadata.get("tokenizer.ggml.tokens").and_then(|v| {
                    if let GgufValue::Array(arr) = v {
                        Some(arr.len() as u32)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(0);

        let rope_freq_base = self.get_f64(&format!("{prefix}.rope.freq_base"));
        let rope_scaling = self
            .get_str(&format!("{prefix}.rope.scaling.type"))
            .map(String::from);

        let expert_count = self.get_u32(&format!("{prefix}.expert_count"));
        let expert_used_count = self.get_u32(&format!("{prefix}.expert_used_count"));

        let norm_eps = self
            .get_f64(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .map(|v| v as f32);
        let sliding_window = self.get_u32(&format!("{prefix}.attention.sliding_window"));
        let partial_rotary_dim = self.get_u32(&format!("{prefix}.rope.dimension_count"));
        let logit_softcap = self
            .get_f64(&format!("{prefix}.attn_logit_softcapping"))
            .map(|v| v as f32);
        let attn_logit_softcap = self
            .get_f64(&format!("{prefix}.attention.attn_logit_softcapping"))
            .map(|v| v as f32);

        // estimate parameter count from tensor sizes
        let parameters: u64 = self.tensors.iter().map(|t| t.num_elements()).sum();

        let name = self.get_str("general.name").map(String::from);

        Ok(LoadedModel {
            metadata: ModelMetadata {
                architecture,
                name,
                parameters,
                context_length,
                embedding_dim,
                head_count,
                head_count_kv,
                layer_count,
                vocab_size,
                rope_freq_base,
                rope_scaling,
                expert_count,
                expert_used_count,
                norm_eps,
                sliding_window,
                partial_rotary_dim,
                logit_softcap,
                attn_logit_softcap,
            },
            tensors: self.tensors.clone(),
            file_size: self.file_size,
            format: ModelFormat::Gguf,
        })
    }

    /// Get tensor data slice from the original mapped file data.
    pub fn tensor_data<'a>(&self, tensor: &TensorInfo, file_data: &'a [u8]) -> Result<&'a [u8]> {
        let start = self.data_offset + tensor.offset;
        let end = start + tensor.size_bytes;
        if end as usize > file_data.len() {
            return Err(YuleError::TensorOutOfBounds {
                name: tensor.name.clone(),
                offset: start,
                file_size: file_data.len() as u64,
            });
        }
        Ok(&file_data[start as usize..end as usize])
    }

    fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    fn get_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(|v| v.as_f64())
    }
}

fn gguf_dtype_to_dtype(gguf_type: u32) -> Result<DType> {
    DType::from_gguf_type_id(gguf_type)
        .ok_or_else(|| YuleError::Parse(format!("unknown GGUF dtype: {gguf_type}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        // magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // version
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        // tensor count
        buf.extend_from_slice(&0u64.to_le_bytes());
        // kv count
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = build_minimal_gguf();
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_ok());
        let gguf = result.unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 0);
        assert_eq!(gguf.metadata.len(), 0);
    }

    #[test]
    fn test_reject_bad_magic() {
        let mut data = build_minimal_gguf();
        data[0] = 0xFF; // corrupt magic
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            YuleError::InvalidMagic { .. }
        ));
    }

    #[test]
    fn test_reject_bad_version() {
        let mut data = build_minimal_gguf();
        data[4..8].copy_from_slice(&99u32.to_le_bytes()); // bad version
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            YuleError::UnsupportedVersion(99)
        ));
    }

    #[test]
    fn test_reject_excessive_tensor_count() {
        let mut data = build_minimal_gguf();
        // overwrite tensor count with absurd value
        data[8..16].copy_from_slice(&(MAX_TENSOR_COUNT + 1).to_le_bytes());
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_excessive_kv_count() {
        let mut data = build_minimal_gguf();
        data[16..24].copy_from_slice(&(MAX_KV_COUNT + 1).to_le_bytes());
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_file() {
        let data = vec![0u8; 8]; // too small
        let parser = GgufParser::new();
        let result = parser.parse_bytes(&data, data.len() as u64);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_with_metadata() {
        let mut buf = Vec::new();
        // magic + version
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        // tensor count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // kv count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV: key = "general.architecture", type = string, value = "llama"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes()); // type
        let val = b"llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        let parser = GgufParser::new();
        let result = parser.parse_bytes(&buf, buf.len() as u64);
        assert!(result.is_ok());
        let gguf = result.unwrap();
        assert_eq!(gguf.get_str("general.architecture"), Some("llama"));
    }

    #[test]
    fn test_parse_with_tensor() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        // tensor count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // kv count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());

        // tensor info
        let name = b"output.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // pad to alignment + add fake tensor data
        let header_end = buf.len() as u64;
        let data_offset = header_end.div_ceil(32) * 32;
        let padding = (data_offset - header_end) as usize;
        buf.extend(vec![0u8; padding]);
        // 4096 * 4096 * 4 bytes = 67108864 bytes of F32 data
        buf.extend(vec![0u8; 4096 * 4096 * 4]);

        let parser = GgufParser::new();
        let result = parser.parse_bytes(&buf, buf.len() as u64);
        assert!(result.is_ok());
        let gguf = result.unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "output.weight");
        assert_eq!(gguf.tensors[0].shape, vec![4096, 4096]);
        assert_eq!(gguf.tensors[0].dtype, DType::F32);
        assert_eq!(gguf.tensors[0].num_elements(), 4096 * 4096);
    }

    #[test]
    fn test_reject_string_too_long() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv count

        // key with absurd length — the DoS vector from CVE-2025-66960
        buf.extend_from_slice(&(MAX_STRING_LEN + 1).to_le_bytes());
        // don't even need the actual bytes — parser should reject before reading

        let parser = GgufParser::new();
        let result = parser.parse_bytes(&buf, buf.len() as u64);
        assert!(result.is_err());
    }
}
