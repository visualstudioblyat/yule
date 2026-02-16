use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Cache layout: `{base_dir}/{publisher}/{repo}/{filename}`
/// Each repo dir has a `cache.json` sidecar with metadata.
pub struct ModelCache {
    base_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub publisher: String,
    pub repo: String,
    pub filename: String,
    pub size_bytes: u64,
    pub merkle_root: Option<String>,
    pub download_url: String,
    pub downloaded_at: String,
}

impl ModelCache {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Get the cached path for a specific file, if it exists.
    pub fn get(&self, publisher: &str, repo: &str, filename: &str) -> Option<PathBuf> {
        let path = self.base_dir.join(publisher).join(repo).join(filename);
        if path.exists() { Some(path) } else { None }
    }

    /// Find any cached GGUF file in a repo directory.
    pub fn find_any(&self, publisher: &str, repo: &str) -> Option<PathBuf> {
        let dir = self.base_dir.join(publisher).join(repo);
        if !dir.is_dir() {
            return None;
        }
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                return Some(path);
            }
        }
        None
    }

    /// Get the file path where a model should be stored.
    pub fn model_path(&self, publisher: &str, repo: &str, filename: &str) -> PathBuf {
        self.base_dir.join(publisher).join(repo).join(filename)
    }

    /// Create the directory tree for a publisher/repo.
    pub fn ensure_dir(&self, publisher: &str, repo: &str) -> Result<()> {
        let dir = self.base_dir.join(publisher).join(repo);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create cache dir: {}", dir.display()))?;
        Ok(())
    }

    /// Write cache metadata sidecar for a repo.
    pub fn write_metadata(&self, publisher: &str, repo: &str, entry: &CacheEntry) -> Result<()> {
        let meta_path = self.base_dir.join(publisher).join(repo).join("cache.json");
        let json =
            serde_json::to_string_pretty(entry).context("failed to serialize cache metadata")?;
        std::fs::write(&meta_path, json)
            .with_context(|| format!("failed to write {}", meta_path.display()))?;
        Ok(())
    }

    /// List all cached models.
    pub fn list_all(&self) -> Result<Vec<CacheEntry>> {
        let mut entries = Vec::new();
        if !self.base_dir.exists() {
            return Ok(entries);
        }
        Self::walk_cache(&self.base_dir, &mut entries)?;
        Ok(entries)
    }

    fn walk_cache(dir: &Path, entries: &mut Vec<CacheEntry>) -> Result<()> {
        let read_dir =
            std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?;

        for item in read_dir.flatten() {
            let path = item.path();
            if path.is_dir() {
                Self::walk_cache(&path, entries)?;
            } else if path.file_name().is_some_and(|n| n == "cache.json") {
                let content = std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(&content) {
                    entries.push(entry);
                }
            }
        }
        Ok(())
    }

    /// Evict a cached model file.
    pub fn evict(&self, publisher: &str, repo: &str, filename: &str) -> Result<()> {
        let path = self.base_dir.join(publisher).join(repo).join(filename);
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to delete {}", path.display()))?;
        }
        // Also remove the sidecar
        let meta_path = self.base_dir.join(publisher).join(repo).join("cache.json");
        if meta_path.exists() {
            let _ = std::fs::remove_file(&meta_path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_path_construction() {
        let cache = ModelCache::new(PathBuf::from("/tmp/yule/models"));
        let path = cache.model_path(
            "bartowski",
            "Llama-3.2-1B-Instruct-GGUF",
            "model-Q4_K_M.gguf",
        );
        assert_eq!(
            path,
            PathBuf::from(
                "/tmp/yule/models/bartowski/Llama-3.2-1B-Instruct-GGUF/model-Q4_K_M.gguf"
            )
        );
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let cache = ModelCache::new(PathBuf::from("/tmp/yule/nonexistent_test_dir_xyz"));
        assert!(cache.get("foo", "bar", "baz.gguf").is_none());
    }

    #[test]
    fn find_any_nonexistent_returns_none() {
        let cache = ModelCache::new(PathBuf::from("/tmp/yule/nonexistent_test_dir_xyz"));
        assert!(cache.find_any("foo", "bar").is_none());
    }

    #[test]
    fn list_empty_cache() {
        let cache = ModelCache::new(PathBuf::from("/tmp/yule/nonexistent_test_dir_xyz"));
        let entries = cache.list_all().unwrap();
        assert!(entries.is_empty());
    }
}
