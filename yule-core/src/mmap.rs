use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// Mmap a model file with the best available strategy:
/// 1. Linux: explicit 2MB huge pages (MAP_HUGETLB) — lowest TLB miss rate
/// 2. Pre-faulted populate — eliminates soft page faults, THP may promote
/// 3. Regular mmap fallback — always works
pub fn mmap_model(path: &Path) -> Result<memmap2::Mmap> {
    let file = File::open(path)?;

    // 1. try explicit 2MB huge pages (Linux only, requires nr_hugepages > 0)
    #[cfg(target_os = "linux")]
    if let Ok(m) = try_mmap_huge_pages(&file) {
        tracing::debug!("mmap: 2MB huge pages ({}MB)", m.len() / (1024 * 1024));
        return Ok(m);
    }

    // 2. try pre-faulting (populate) — reduces page faults during inference
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    if let Ok(m) = try_mmap_populate(&file) {
        tracing::debug!("mmap: pre-faulted ({}MB)", m.len() / (1024 * 1024));
        return Ok(m);
    }

    // 3. regular mmap fallback — always works
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    tracing::debug!("mmap: standard pages ({}MB)", mmap.len() / (1024 * 1024));
    Ok(mmap)
}

/// Explicit 2MB huge pages via MAP_HUGETLB. Requires the system to have huge
/// pages reserved (`echo N > /proc/sys/vm/nr_hugepages`) or the process to have
/// CAP_IPC_LOCK. Falls back gracefully on ENOMEM.
#[cfg(target_os = "linux")]
fn try_mmap_huge_pages(file: &File) -> std::result::Result<memmap2::Mmap, std::io::Error> {
    // huge(Some(21)) = MAP_HUGETLB | MAP_HUGE_2MB (21 = log2(2MB))
    unsafe {
        memmap2::MmapOptions::new()
            .huge(Some(21))
            .populate()
            .map(file)
    }
}

/// Pre-fault all pages on map. On Linux with transparent huge pages enabled,
/// this also triggers THP promotion for the contiguous mapped region.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
fn try_mmap_populate(file: &File) -> std::result::Result<memmap2::Mmap, std::io::Error> {
    unsafe { memmap2::MmapOptions::new().populate().map(file) }
}
