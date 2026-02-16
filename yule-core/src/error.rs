use thiserror::Error;

#[derive(Error, Debug)]
pub enum YuleError {
    #[error("parse error: {0}")]
    Parse(String),

    #[error("invalid magic number: expected {expected:#010x}, got {got:#010x}")]
    InvalidMagic { expected: u32, got: u32 },

    #[error("unsupported format version: {0}")]
    UnsupportedVersion(u32),

    #[error("tensor '{name}' has invalid shape: {reason}")]
    InvalidTensorShape { name: String, reason: String },

    #[error("tensor '{name}' offset {offset} exceeds file bounds {file_size}")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        file_size: u64,
    },

    #[error("metadata key '{key}' has unexpected type: expected {expected}, got {got}")]
    MetadataTypeMismatch {
        key: String,
        expected: String,
        got: String,
    },

    #[error("allocation too large: requested {requested} bytes, max {max} bytes")]
    AllocationTooLarge { requested: u64, max: u64 },

    #[error("string too long: {len} bytes, max {max} bytes")]
    StringTooLong { len: u64, max: u64 },

    #[error("verification failed: {0}")]
    Verification(String),

    #[error("sandbox error: {0}")]
    Sandbox(String),

    #[error("gpu error: {0}")]
    Gpu(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("api error: {0}")]
    Api(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, YuleError>;
