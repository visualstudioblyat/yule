use crate::BufferHandle;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

pub fn next_buffer_handle() -> BufferHandle {
    BufferHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
}

#[allow(dead_code)]
pub struct BufferPool {
    allocated: HashMap<u64, AllocatedBuffer>,
    total_bytes: u64,
    max_bytes: u64,
}

#[allow(dead_code)]
struct AllocatedBuffer {
    size: usize,
    in_use: bool,
}

impl BufferPool {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            allocated: HashMap::new(),
            total_bytes: 0,
            max_bytes,
        }
    }

    pub fn total_allocated(&self) -> u64 {
        self.total_bytes
    }

    pub fn max_capacity(&self) -> u64 {
        self.max_bytes
    }
}
