use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// Mmap a model file with page pre-faulting when available, falling back to regular mmap.
/// Pre-faulting eliminates soft page faults during inference; on Linux with THP enabled,
/// the kernel promotes contiguous 4KB pages into 2MB transparent huge pages automatically.
pub fn mmap_model(path: &Path) -> Result<memmap2::Mmap> {
    let file = File::open(path)?;

    // try pre-faulting (populate) mmap first — reduces page faults during inference
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    if let Ok(m) = try_mmap_populate(&file) {
        tracing::debug!("mmap: pre-faulted ({}MB)", m.len() / (1024 * 1024));
        return Ok(m);
    }

    // regular mmap fallback — always works
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    tracing::debug!("mmap: standard pages ({}MB)", mmap.len() / (1024 * 1024));
    Ok(mmap)
}

/// Pre-fault all pages on map. On Linux with transparent huge pages enabled,
/// this also triggers THP promotion for the contiguous mapped region.
/// On Windows, this is MAP_POPULATE-equivalent via memmap2's populate().
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn try_mmap_populate(file: &File) -> std::result::Result<memmap2::Mmap, std::io::Error> {
    unsafe { memmap2::MmapOptions::new().populate().map(file) }
}
