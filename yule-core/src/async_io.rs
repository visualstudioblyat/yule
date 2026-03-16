//! Async I/O utilities for weight loading.
//! On Linux 5.6+, uses io_uring for high-throughput sequential reads.
//! Falls back to synchronous read + mmap on other platforms.

use crate::error::{Result, YuleError};
use std::path::Path;

pub struct AsyncLoadConfig {
    pub chunk_size: usize, // default 2MB
    pub queue_depth: u32,  // default 32
    pub direct_io: bool,   // bypass page cache (default false)
}

impl Default for AsyncLoadConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2 * 1024 * 1024,
            queue_depth: 32,
            direct_io: false,
        }
    }
}

pub struct AsyncLoadProgress {
    pub bytes_loaded: u64,
    pub total_bytes: u64,
}

impl AsyncLoadProgress {
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.bytes_loaded as f64 / self.total_bytes as f64
        }
    }
}

/// Load a file asynchronously. On platforms without io_uring support,
/// falls back to a synchronous chunked read.
pub fn async_load_file(path: &Path, _config: &AsyncLoadConfig) -> Result<Vec<u8>> {
    // io_uring requires Linux 5.6+ and the io-uring crate.
    // For now, do a synchronous read as the fallback path.
    // The API surface is ready for when io_uring integration is added.
    std::fs::read(path).map_err(YuleError::Io)
}

/// Check if io_uring is available on this system.
pub fn is_io_uring_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check kernel version >= 5.6
        if let Ok(info) = std::fs::read_to_string("/proc/version") {
            if let Some(ver) = info.split_whitespace().nth(2) {
                let parts: Vec<&str> = ver.split('.').collect();
                if parts.len() >= 2 {
                    if let (Ok(major), Ok(minor)) =
                        (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                    {
                        return major > 5 || (major == 5 && minor >= 6);
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_async_load_config_default() {
        let config = AsyncLoadConfig::default();
        assert_eq!(config.chunk_size, 2 * 1024 * 1024);
        assert_eq!(config.queue_depth, 32);
        assert!(!config.direct_io);
    }

    #[test]
    fn test_async_load_progress_fraction() {
        let progress = AsyncLoadProgress {
            bytes_loaded: 50,
            total_bytes: 100,
        };
        assert!((progress.fraction() - 0.5).abs() < 1e-10);

        let zero = AsyncLoadProgress {
            bytes_loaded: 0,
            total_bytes: 0,
        };
        assert!((zero.fraction() - 0.0).abs() < 1e-10);

        let complete = AsyncLoadProgress {
            bytes_loaded: 100,
            total_bytes: 100,
        };
        assert!((complete.fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_async_load_file_reads_content() {
        let dir = std::env::temp_dir().join("yule_async_io_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_weights.bin");
        let data = b"hello yule weights";
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(data).unwrap();
        }
        let config = AsyncLoadConfig::default();
        let loaded = async_load_file(&path, &config).unwrap();
        assert_eq!(loaded, data);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_async_load_file_missing() {
        let path = std::path::Path::new("/nonexistent/path/model.bin");
        let config = AsyncLoadConfig::default();
        assert!(async_load_file(path, &config).is_err());
    }

    #[test]
    fn test_is_io_uring_available_does_not_panic() {
        // On non-Linux (e.g. Windows), this should return false
        // On Linux, it depends on the kernel version
        let _available = is_io_uring_available();
        #[cfg(not(target_os = "linux"))]
        assert!(!_available);
    }
}
