use yule_core::error::{Result, YuleError};
use yule_gpu::{BufferHandle, ComputeBackend};

pub struct KvCache {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub current_len: u32,
    key_buffers: Vec<BufferHandle>,
    value_buffers: Vec<BufferHandle>,
}

impl KvCache {
    pub fn allocate(
        backend: &dyn ComputeBackend,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> Result<Self> {
        let kv_len = max_seq_len as usize * num_kv_heads as usize * head_dim as usize;
        let buf_size = kv_len * 4; // f32

        let mut key_buffers = Vec::with_capacity(num_layers as usize);
        let mut value_buffers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            key_buffers.push(backend.allocate(buf_size)?);
            value_buffers.push(backend.allocate(buf_size)?);
        }

        Ok(Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            key_buffers,
            value_buffers,
        })
    }

    pub fn key_buffer(&self, layer: u32) -> Result<&BufferHandle> {
        self.key_buffers
            .get(layer as usize)
            .ok_or_else(|| YuleError::Inference(format!("KV cache layer {layer} out of range")))
    }

    pub fn value_buffer(&self, layer: u32) -> Result<&BufferHandle> {
        self.value_buffers
            .get(layer as usize)
            .ok_or_else(|| YuleError::Inference(format!("KV cache layer {layer} out of range")))
    }

    pub fn write_kv(
        &mut self,
        backend: &dyn ComputeBackend,
        layer: u32,
        pos: u32,
        k_data: &BufferHandle,
        v_data: &BufferHandle,
    ) -> Result<()> {
        if pos >= self.max_seq_len {
            return Err(YuleError::Inference("KV cache position exceeds max".into()));
        }

        let kv_stride = self.num_kv_heads as usize * self.head_dim as usize;
        let byte_offset = pos as usize * kv_stride * 4;
        let byte_size = kv_stride * 4;

        backend.copy_buffer_offset(k_data, self.key_buffer(layer)?, 0, byte_offset, byte_size)?;
        backend.copy_buffer_offset(v_data, self.value_buffer(layer)?, 0, byte_offset, byte_size)?;

        if pos + 1 > self.current_len {
            self.current_len = pos + 1;
        }
        Ok(())
    }

    pub fn size_bytes(&self) -> u64 {
        let per_layer =
            2 * self.num_kv_heads as u64 * self.head_dim as u64 * self.max_seq_len as u64 * 4;
        per_layer * self.num_layers as u64
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.max_seq_len.saturating_sub(self.current_len)
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

pub struct PagedKvCache {
    pub page_size: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_pages: u32,
    page_table: Vec<Vec<Option<BufferHandle>>>,
    #[allow(dead_code)] // populated by future page fault handler
    logical_to_physical: Vec<Vec<Option<u32>>>,
}

impl PagedKvCache {
    pub fn new(
        page_size: u32,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_pages: u32,
    ) -> Self {
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            num_pages: max_pages,
            page_table: vec![Vec::new(); num_layers as usize],
            logical_to_physical: vec![Vec::new(); num_layers as usize],
        }
    }

    pub fn allocate_page(
        &mut self,
        backend: &dyn ComputeBackend,
        layer: u32,
    ) -> Result<u32> {
        let page_elements =
            self.page_size as usize * self.num_kv_heads as usize * self.head_dim as usize;
        let buf_size = page_elements * 4 * 2; // K + V interleaved

        let handle = backend.allocate(buf_size)?;
        let page_idx = self.page_table[layer as usize].len() as u32;
        self.page_table[layer as usize].push(Some(handle));
        Ok(page_idx)
    }

    pub fn tokens_per_page(&self) -> u32 {
        self.page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_size_calculation() {
        // 2 layers, 4 kv heads, 64 head dim, 1024 max seq
        let cache = KvCache {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 1024,
            current_len: 0,
            key_buffers: Vec::new(),
            value_buffers: Vec::new(),
        };
        // per layer: 2 * 4 * 64 * 1024 * 4 = 2MB
        // 2 layers: 4MB = 4194304
        assert_eq!(cache.size_bytes(), 4_194_304);
    }

    #[test]
    fn kv_cache_remaining() {
        let mut cache = KvCache {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_seq_len: 100,
            current_len: 0,
            key_buffers: Vec::new(),
            value_buffers: Vec::new(),
        };
        assert_eq!(cache.remaining_tokens(), 100);
        cache.current_len = 60;
        assert_eq!(cache.remaining_tokens(), 40);
        cache.clear();
        assert_eq!(cache.remaining_tokens(), 100);
    }
}
