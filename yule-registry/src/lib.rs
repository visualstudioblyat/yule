pub mod cache;
pub mod hf_api;
pub mod pull;
pub mod quant;
pub mod resolve;

use anyhow::{Context, Result};
use std::path::PathBuf;

use cache::{CacheEntry, ModelCache};
use hf_api::HfApiClient;
use pull::ModelPuller;
use resolve::{ModelResolver, ParsedModelRef};

pub struct Registry {
    cache: ModelCache,
    hf_client: HfApiClient,
    puller: ModelPuller,
}

/// A model that was successfully pulled and cached.
#[derive(Debug)]
pub struct PulledModel {
    pub path: PathBuf,
    pub publisher: String,
    pub repo: String,
    pub filename: String,
    pub size_bytes: u64,
    pub merkle_root: String,
}

/// Summary of a cached model for display.
#[derive(Debug)]
pub struct CachedModel {
    pub publisher: String,
    pub repo: String,
    pub filename: String,
    pub size_bytes: u64,
    pub merkle_root: Option<String>,
}

impl Registry {
    pub fn new(cache_dir: PathBuf, hf_token: Option<String>) -> Result<Self> {
        let hf_client =
            HfApiClient::new(hf_token.clone()).context("failed to create HuggingFace client")?;
        let puller_client =
            HfApiClient::new(hf_token).context("failed to create download client")?;

        Ok(Self {
            cache: ModelCache::new(cache_dir),
            hf_client,
            puller: ModelPuller::new(puller_client),
        })
    }

    pub fn default_cache_dir() -> PathBuf {
        dirs_path().join("models")
    }

    /// Resolve a model reference to a local file path (cache lookup only, no download).
    pub fn resolve_local(&self, model_ref: &str) -> Result<Option<PathBuf>> {
        let parsed = ModelResolver::parse_model_ref(model_ref)?;
        match parsed {
            ParsedModelRef::LocalFile(p) => {
                let path = PathBuf::from(&p);
                if path.exists() {
                    Ok(Some(path))
                } else {
                    Ok(None)
                }
            }
            ParsedModelRef::Remote {
                publisher, name, ..
            } => {
                // Try exact filename match in cache, then any GGUF
                Ok(self.cache.find_any(&publisher, &name))
            }
        }
    }

    /// Full pull pipeline: list repo → select quant → download → verify → cache.
    pub async fn pull(&self, model_ref: &str) -> Result<PulledModel> {
        let parsed = ModelResolver::parse_model_ref(model_ref)?;
        let (publisher, repo, requested_quant) = match parsed {
            ParsedModelRef::Remote {
                publisher,
                name,
                quantization,
            } => (publisher, name, quantization),
            ParsedModelRef::LocalFile(p) => {
                anyhow::bail!("cannot pull a local file path: {p}");
            }
        };

        eprintln!("fetching file list for {publisher}/{repo}...");
        let files = self
            .hf_client
            .list_repo_files(&publisher, &repo)
            .await
            .context("failed to list repo files")?;

        let gguf_files = quant::filter_gguf_files(&files);
        if gguf_files.is_empty() {
            anyhow::bail!("no GGUF files found in {publisher}/{repo}");
        }

        eprintln!("found {} GGUF file(s)", gguf_files.len());
        for f in &gguf_files {
            let label = quant::extract_quant_label(&f.path).unwrap_or_default();
            eprintln!(
                "  {} ({}) {}",
                f.path,
                format_bytes(f.size),
                if label.is_empty() {
                    String::new()
                } else {
                    format!("[{label}]")
                }
            );
        }

        let selected = quant::select_gguf_file(&gguf_files, requested_quant.as_deref())
            .context("failed to select a GGUF file")?;

        let quant_label = quant::extract_quant_label(&selected.path).unwrap_or_default();
        eprintln!(
            "\nselected: {} ({}){}",
            selected.path,
            format_bytes(selected.size),
            if quant_label.is_empty() {
                String::new()
            } else {
                format!(" [{quant_label}]")
            }
        );

        // Check cache
        if let Some(cached) = self.cache.get(&publisher, &repo, &selected.path) {
            eprintln!("already cached at {}", cached.display());
            // Still verify to report merkle root
            let merkle = compute_merkle_root(&cached)?;
            return Ok(PulledModel {
                path: cached,
                publisher,
                repo,
                filename: selected.path.clone(),
                size_bytes: selected.size,
                merkle_root: merkle,
            });
        }

        // Download
        self.cache.ensure_dir(&publisher, &repo)?;
        let dest = self.cache.model_path(&publisher, &repo, &selected.path);
        let url = HfApiClient::download_url(&publisher, &repo, &selected.path);

        eprintln!("\ndownloading {}...", selected.path);
        self.puller
            .download(&url, &dest)
            .await
            .context("download failed")?;

        // Verify: parse GGUF + compute merkle root
        eprintln!("verifying...");
        let merkle = compute_merkle_root(&dest)?;
        eprintln!("merkle root: {merkle}");

        // Write cache metadata
        let now = chrono_now();
        self.cache.write_metadata(
            &publisher,
            &repo,
            &CacheEntry {
                publisher: publisher.clone(),
                repo: repo.clone(),
                filename: selected.path.clone(),
                size_bytes: selected.size,
                merkle_root: Some(merkle.clone()),
                download_url: url,
                downloaded_at: now,
            },
        )?;

        eprintln!("cached at {}", dest.display());

        Ok(PulledModel {
            path: dest,
            publisher,
            repo,
            filename: selected.path.clone(),
            size_bytes: selected.size,
            merkle_root: merkle,
        })
    }

    /// List all cached models.
    pub fn list_cached(&self) -> Result<Vec<CachedModel>> {
        let entries = self.cache.list_all()?;
        Ok(entries
            .into_iter()
            .map(|e| CachedModel {
                publisher: e.publisher,
                repo: e.repo,
                filename: e.filename,
                size_bytes: e.size_bytes,
                merkle_root: e.merkle_root,
            })
            .collect())
    }
}

/// Parse a GGUF file and compute its merkle root over tensor data.
fn compute_merkle_root(path: &std::path::Path) -> Result<String> {
    let parser = yule_core::gguf::GgufParser::new();
    let gguf = parser
        .parse_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to parse GGUF for verification")?;

    let mmap = yule_core::mmap::mmap_model(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to mmap model for verification")?;

    if (gguf.data_offset as usize) > mmap.len() {
        anyhow::bail!("data offset exceeds file size — corrupt GGUF?");
    }

    let tensor_data = &mmap[gguf.data_offset as usize..];
    let tree = yule_verify::merkle::MerkleTree::new();
    let root = tree.build(tensor_data);

    Ok(root.hash.iter().map(|b| format!("{b:02x}")).collect())
}

/// Simple ISO-8601 timestamp without pulling in chrono.
fn chrono_now() -> String {
    // Use std::time::SystemTime for a rough timestamp
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate: good enough for a cache sidecar
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hour = (secs % 86400) / 3600;
    let min = (secs % 3600) / 60;
    let sec = secs % 60;
    format!("{years:04}-{months:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn dirs_path() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".yule")
    }

    #[cfg(target_os = "macos")]
    {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".yule")
    }

    #[cfg(target_os = "windows")]
    {
        PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("yule")
    }
}

fn format_bytes(n: u64) -> String {
    if n >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", n as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if n >= 1024 * 1024 {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_dir_not_empty() {
        let dir = Registry::default_cache_dir();
        assert!(dir.to_str().unwrap().contains("models"));
    }

    #[test]
    fn chrono_now_format() {
        let ts = chrono_now();
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }
}
