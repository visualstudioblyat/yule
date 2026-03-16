use crate::BufferHandle;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

pub fn next_buffer_handle() -> BufferHandle {
    BufferHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
}

pub struct BufferPool {
    allocated: HashMap<u64, AllocatedBuffer>,
    free_list: Vec<(u64, usize)>, // (handle_id, size) — reusable buffers
    total_bytes: u64,
    max_bytes: u64,
}

struct AllocatedBuffer {
    size: usize,
    in_use: bool,
}

impl BufferPool {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            allocated: HashMap::new(),
            free_list: Vec::new(),
            total_bytes: 0,
            max_bytes,
        }
    }

    pub fn acquire(&mut self, size: usize) -> Option<BufferHandle> {
        // Try to find a free buffer of the right size (exact match or slightly larger)
        let best = self
            .free_list
            .iter()
            .enumerate()
            .filter(|(_, (_, s))| *s >= size && *s <= size + size / 4) // within 25% overalloc
            .min_by_key(|(_, (_, s))| *s)
            .map(|(idx, (id, _))| (idx, *id));

        if let Some((idx, id)) = best {
            self.free_list.swap_remove(idx);
            if let Some(buf) = self.allocated.get_mut(&id) {
                buf.in_use = true;
            }
            return Some(BufferHandle(id));
        }

        // Check if we have capacity for a new allocation
        if self.total_bytes + size as u64 > self.max_bytes {
            return None; // over budget
        }

        let handle = next_buffer_handle();
        self.allocated.insert(
            handle.0,
            AllocatedBuffer {
                size,
                in_use: true,
            },
        );
        self.total_bytes += size as u64;
        Some(handle)
    }

    pub fn release(&mut self, handle: &BufferHandle) {
        if let Some(buf) = self.allocated.get_mut(&handle.0) {
            buf.in_use = false;
            self.free_list.push((handle.0, buf.size));
        }
    }

    pub fn total_allocated(&self) -> u64 {
        self.total_bytes
    }

    pub fn max_capacity(&self) -> u64 {
        self.max_bytes
    }

    pub fn in_use_count(&self) -> usize {
        self.allocated.values().filter(|b| b.in_use).count()
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_pool_acquire_release() {
        let mut pool = BufferPool::new(1024);

        let h1 = pool.acquire(256).unwrap();
        assert_eq!(pool.in_use_count(), 1);
        assert_eq!(pool.total_allocated(), 256);

        let h2 = pool.acquire(256).unwrap();
        assert_eq!(pool.in_use_count(), 2);

        pool.release(&h1);
        assert_eq!(pool.in_use_count(), 1);
        assert_eq!(pool.free_count(), 1);

        // Reacquire should reuse the freed buffer
        let h3 = pool.acquire(256).unwrap();
        assert_eq!(h3.0, h1.0); // same handle reused
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.in_use_count(), 2);

        pool.release(&h2);
        pool.release(&h3);
    }

    #[test]
    fn buffer_pool_over_budget() {
        let mut pool = BufferPool::new(100);
        let _h1 = pool.acquire(80).unwrap();
        assert!(pool.acquire(30).is_none()); // would exceed 100
    }
}
