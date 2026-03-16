use std::path::{Path, PathBuf};
use yule_core::gguf::{GgufFile, GgufParser};
use yule_core::model::LoadedModel;
use yule_core::tokenizer::BpeTokenizer;
use yule_infer::model_runner::{ModelRunner, TransformerRunner};
use yule_infer::weight_loader::{TransformerWeights, WeightStore};

pub struct LoadedTestModel {
    #[allow(dead_code)]
    pub path: PathBuf,
    pub gguf: GgufFile,
    pub model_info: LoadedModel,
    pub tokenizer: BpeTokenizer,
    // Holds the data that the runner borrows from
    _weight_data: Vec<u8>,
}

impl LoadedTestModel {
    /// Access the raw file bytes (needed for tensor data extraction).
    pub fn file_data(&self) -> &[u8] {
        &self._weight_data
    }
}

pub struct TestRunner {
    pub runner: Box<dyn ModelRunner>,
}

/// Resolve the model path from an optional CLI argument.
///
/// 1. If `--model-path` was given, use that directly.
/// 2. Otherwise, look in the default registry cache directory.
/// 3. If nothing is found, return an error with download instructions.
pub fn resolve_model_path(model_path_arg: Option<&str>) -> anyhow::Result<PathBuf> {
    // If an explicit path was provided, use it directly
    if let Some(p) = model_path_arg {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }

        // Try registry resolution for model references like "TheBloke/TinyLlama..."
        let cache_dir = yule_registry::Registry::default_cache_dir();
        let hf_token = std::env::var("HF_TOKEN").ok();
        if let Ok(registry) = yule_registry::Registry::new(cache_dir, hf_token) {
            if let Ok(Some(resolved)) = registry.resolve_local(p) {
                return Ok(resolved);
            }
        }

        anyhow::bail!("model not found at: {p}");
    }

    // No explicit path — search the cache directory for any .gguf file
    let cache_dir = yule_registry::Registry::default_cache_dir();
    if cache_dir.is_dir() {
        // Look for any GGUF file in the cache
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    // Check subdirectories (publisher/repo structure)
                    if let Ok(sub_entries) = std::fs::read_dir(&entry_path) {
                        for sub in sub_entries.flatten() {
                            let sub_path = sub.path();
                            if sub_path.is_dir() {
                                if let Ok(files) = std::fs::read_dir(&sub_path) {
                                    for file in files.flatten() {
                                        let file_path = file.path();
                                        if file_path.extension().is_some_and(|ext| ext == "gguf") {
                                            return Ok(file_path);
                                        }
                                    }
                                }
                            } else if sub_path.extension().is_some_and(|ext| ext == "gguf") {
                                return Ok(sub_path);
                            }
                        }
                    }
                } else if entry_path.extension().is_some_and(|ext| ext == "gguf") {
                    return Ok(entry_path);
                }
            }
        }
    }

    anyhow::bail!(
        "Model not found. Download it with:\n  \
         yule pull TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF\n\
         Or pass --model-path <path>"
    );
}

/// Load the model file: parse GGUF, build tokenizer, extract model info.
pub fn load_model(path: &Path) -> anyhow::Result<LoadedTestModel> {
    // 1. Read the entire file into a Vec<u8> (needed for lifetime management)
    let data = std::fs::read(path)?;

    // 2. Parse GGUF from the bytes
    let parser = GgufParser::new();
    let gguf = parser.parse_bytes(&data, data.len() as u64)?;

    // 3. Build tokenizer from GGUF metadata
    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;

    // 4. Extract model info
    let model_info = gguf.to_loaded_model()?;

    Ok(LoadedTestModel {
        path: path.to_path_buf(),
        gguf,
        model_info,
        tokenizer,
        _weight_data: data,
    })
}

/// Create a model runner from a GGUF file on disk.
///
/// Uses the same unsafe transmute pattern as `InferenceEngine::load_model`
/// to extend the lifetime of owned weight data so the runner can borrow it.
pub fn create_runner(path: &Path) -> anyhow::Result<TestRunner> {
    // Own the weight data in a Box so it has a stable address
    let weight_data: Vec<u8> = std::fs::read(path)?;
    let weight_data = Box::new(weight_data);

    // SAFETY: weight_data is owned by the Box which we leak into a 'static reference.
    // The runner holds references into this data. The data will live for the duration
    // of the program (we intentionally leak it to satisfy lifetime requirements,
    // matching the pattern used in InferenceEngine::load_model).
    let static_ref: &'static [u8] =
        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(weight_data.as_slice()) };
    // Leak the box so the data lives forever (validation is short-lived anyway)
    std::mem::forget(weight_data);

    // Parse GGUF from the static ref
    let parser = GgufParser::new();
    let gguf = parser.parse_bytes(static_ref, static_ref.len() as u64)?;

    // Build weight store and runner (CPU backend only for validation)
    let store = WeightStore::from_gguf(&gguf, static_ref)?;
    let weights = TransformerWeights::new(store);
    let runner: Box<dyn ModelRunner> = Box::new(TransformerRunner::new(weights)?);

    Ok(TestRunner { runner })
}
