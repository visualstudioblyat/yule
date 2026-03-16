use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// Options controlling how model files are memory-mapped.
///
/// The defaults are tuned for inference workloads:
///   - `huge_pages`: true on Linux (try 2MB MAP_HUGETLB), false elsewhere
///   - `prefault`: false (don't pre-fault pages; use `true` to eliminate soft faults)
///   - `sequential`: true (hint sequential access via madvise)
pub struct MmapOptions {
    /// Try 2MB huge pages (MAP_HUGETLB on Linux).
    /// Falls back gracefully if huge pages are not available.
    pub huge_pages: bool,
    /// Pre-fault pages into memory on map (MAP_POPULATE / MADV_WILLNEED).
    pub prefault: bool,
    /// Hint sequential access pattern (MADV_SEQUENTIAL).
    pub sequential: bool,
}

impl Default for MmapOptions {
    fn default() -> Self {
        Self {
            huge_pages: cfg!(target_os = "linux"),
            prefault: false,
            sequential: true,
        }
    }
}

/// Mmap a model file with the best available strategy (using default options):
/// 1. Linux: explicit 2MB huge pages (MAP_HUGETLB) — lowest TLB miss rate
/// 2. Pre-faulted populate — eliminates soft page faults, THP may promote
/// 3. Regular mmap fallback — always works
pub fn mmap_model(path: &Path) -> Result<memmap2::Mmap> {
    mmap_model_with(path, &MmapOptions::default())
}

/// Mmap a model file with explicit options controlling huge pages, prefaulting,
/// and access-pattern hints.
pub fn mmap_model_with(path: &Path, opts: &MmapOptions) -> Result<memmap2::Mmap> {
    let file = File::open(path)?;

    // 1. Try explicit huge pages if requested
    if opts.huge_pages {
        if let Some(m) = try_huge_pages(&file) {
            tracing::debug!("mmap: huge pages ({}MB)", m.len() / (1024 * 1024));
            apply_post_map_hints(&m, opts);
            return Ok(m);
        }
    }

    // 2. Try pre-faulted (populate) mapping
    if opts.prefault {
        if let Ok(m) = try_mmap_populate(&file) {
            tracing::debug!("mmap: pre-faulted ({}MB)", m.len() / (1024 * 1024));
            apply_post_map_hints(&m, opts);
            return Ok(m);
        }
    }

    // 3. Regular mmap fallback — always works
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    tracing::debug!("mmap: standard pages ({}MB)", mmap.len() / (1024 * 1024));
    apply_post_map_hints(&mmap, opts);
    Ok(mmap)
}

// ─── Platform-specific huge page helpers ────────────────────────────

/// Try to map with huge pages. Returns `Some(mmap)` on success, `None` on
/// failure or unsupported platform.
#[cfg(target_os = "linux")]
fn try_huge_pages(file: &File) -> Option<memmap2::Mmap> {
    // Try explicit 2MB huge pages (MAP_HUGETLB | MAP_HUGE_2MB).
    // Requires `echo N > /proc/sys/vm/nr_hugepages` or CAP_IPC_LOCK.
    match unsafe { memmap2::MmapOptions::new().huge(Some(21)).populate().map(file) } {
        Ok(m) => {
            tracing::debug!("mmap: explicit MAP_HUGETLB succeeded");
            Some(m)
        }
        Err(e) => {
            tracing::debug!("mmap: MAP_HUGETLB failed ({}), trying THP madvise", e);
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn try_huge_pages(_file: &File) -> Option<memmap2::Mmap> {
    // macOS does not support MAP_HUGETLB. Huge page promotion is handled
    // entirely through madvise hints applied after mapping.
    tracing::debug!("mmap: macOS has no MAP_HUGETLB; relying on madvise hints");
    None
}

#[cfg(target_os = "windows")]
fn try_huge_pages(_file: &File) -> Option<memmap2::Mmap> {
    // Windows large pages require SeLockMemoryPrivilege and FILE_MAP_LARGE_PAGES.
    // memmap2 doesn't expose these flags directly. We fall back to regular
    // mapping with PrefetchVirtualMemory hints applied post-map.
    tracing::debug!("mmap: Windows large pages not directly supported via memmap2; using fallback");
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn try_huge_pages(_file: &File) -> Option<memmap2::Mmap> {
    None
}

/// Pre-fault all pages on map. On Linux with transparent huge pages enabled,
/// this also triggers THP promotion for the contiguous mapped region.
fn try_mmap_populate(file: &File) -> std::result::Result<memmap2::Mmap, std::io::Error> {
    unsafe { memmap2::MmapOptions::new().populate().map(file) }
}

// ─── Post-map advisory hints ───────────────────────────────────────

/// Apply madvise / platform hints after successful mapping.
#[allow(unused_variables)]
fn apply_post_map_hints(mmap: &memmap2::Mmap, opts: &MmapOptions) {
    #[cfg(target_os = "linux")]
    apply_linux_hints(mmap, opts);

    #[cfg(target_os = "macos")]
    apply_macos_hints(mmap, opts);

    #[cfg(target_os = "windows")]
    apply_windows_hints(mmap, opts);
}

// ─── Linux madvise hints ───────────────────────────────────────────

#[cfg(target_os = "linux")]
fn apply_linux_hints(mmap: &memmap2::Mmap, opts: &MmapOptions) {
    use std::ffi::c_void;

    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }

    let ptr = mmap.as_ptr() as *mut c_void;
    let len = mmap.len();

    // Request transparent huge pages — softer than MAP_HUGETLB, works without
    // explicit nr_hugepages reservation. MADV_HUGEPAGE = 14 on Linux.
    if opts.huge_pages {
        const MADV_HUGEPAGE: i32 = 14;
        let ret = unsafe { madvise(ptr, len, MADV_HUGEPAGE) };
        if ret == 0 {
            tracing::debug!("mmap: MADV_HUGEPAGE applied (transparent huge pages)");
        } else {
            tracing::debug!("mmap: MADV_HUGEPAGE not available, continuing without");
        }
    }

    // Sequential access hint for kernel read-ahead.
    if opts.sequential {
        const MADV_SEQUENTIAL: i32 = 2;
        let ret = unsafe { madvise(ptr, len, MADV_SEQUENTIAL) };
        if ret == 0 {
            tracing::debug!("mmap: MADV_SEQUENTIAL applied");
        } else {
            tracing::debug!("mmap: MADV_SEQUENTIAL hint failed");
        }
    }

    // Pre-fault via MADV_WILLNEED if prefault was requested but we didn't
    // use populate (e.g., fell through to regular mmap).
    if opts.prefault {
        const MADV_WILLNEED: i32 = 3;
        let ret = unsafe { madvise(ptr, len, MADV_WILLNEED) };
        if ret == 0 {
            tracing::debug!("mmap: MADV_WILLNEED applied (pre-faulting)");
        } else {
            tracing::debug!("mmap: MADV_WILLNEED hint failed");
        }
    }
}

// ─── macOS madvise hints ───────────────────────────────────────────

#[cfg(target_os = "macos")]
fn apply_macos_hints(mmap: &memmap2::Mmap, opts: &MmapOptions) {
    use std::ffi::c_void;

    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }

    let ptr = mmap.as_ptr() as *mut c_void;
    let len = mmap.len();

    // Sequential access hint.
    if opts.sequential {
        const MADV_SEQUENTIAL: i32 = 2;
        let ret = unsafe { madvise(ptr, len, MADV_SEQUENTIAL) };
        if ret == 0 {
            tracing::debug!("mmap: MADV_SEQUENTIAL applied");
        } else {
            tracing::debug!("mmap: MADV_SEQUENTIAL hint failed");
        }
    }

    // Pre-fault pages via MADV_WILLNEED.
    if opts.prefault {
        const MADV_WILLNEED: i32 = 1;
        let ret = unsafe { madvise(ptr, len, MADV_WILLNEED) };
        if ret == 0 {
            tracing::debug!("mmap: MADV_WILLNEED applied (pre-faulting pages)");
        } else {
            tracing::debug!("mmap: MADV_WILLNEED hint failed");
        }
    }
}

// ─── Windows hints ─────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn apply_windows_hints(mmap: &memmap2::Mmap, opts: &MmapOptions) {
    if opts.prefault || opts.sequential {
        prefetch_virtual_memory_win(mmap);
    }
}

/// Try to use PrefetchVirtualMemory (Windows 8+) to pre-fault mapped pages.
/// Dynamically loaded to avoid hard dependency on newer Windows versions.
#[cfg(target_os = "windows")]
fn prefetch_virtual_memory_win(mmap: &memmap2::Mmap) {
    use std::ffi::c_void;
    use std::mem;

    #[repr(C)]
    struct MemoryRangeEntry {
        virtual_address: *mut c_void,
        number_of_bytes: usize,
    }

    type PrefetchFn = unsafe extern "system" fn(
        h_process: isize,
        number_of_entries: usize,
        virtual_addresses: *const MemoryRangeEntry,
        flags: u32,
    ) -> i32;

    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn LoadLibraryA(name: *const u8) -> isize;
        fn GetProcAddress(module: isize, name: *const u8) -> *const c_void;
    }

    unsafe {
        let kernel32 = LoadLibraryA(b"kernel32.dll\0".as_ptr());
        if kernel32 == 0 {
            tracing::debug!("mmap: could not load kernel32 for PrefetchVirtualMemory");
            return;
        }
        let proc = GetProcAddress(kernel32, b"PrefetchVirtualMemory\0".as_ptr());
        if proc.is_null() {
            tracing::debug!("mmap: PrefetchVirtualMemory not available on this Windows version");
            return;
        }
        let prefetch: PrefetchFn = mem::transmute(proc);
        let entry = MemoryRangeEntry {
            virtual_address: mmap.as_ptr() as *mut c_void,
            number_of_bytes: mmap.len(),
        };
        let ret = prefetch(GetCurrentProcess(), 1, &entry, 0);
        if ret != 0 {
            tracing::debug!(
                "mmap: PrefetchVirtualMemory applied ({}MB)",
                mmap.len() / (1024 * 1024)
            );
        } else {
            tracing::debug!("mmap: PrefetchVirtualMemory failed");
        }
    }
}
