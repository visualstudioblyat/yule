use yule_gpu::BufferHandle;

#[allow(dead_code)]
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
    pub fn new(num_layers: u32, num_kv_heads: u32, head_dim: u32, max_seq_len: u32) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            key_buffers: Vec::new(),
            value_buffers: Vec::new(),
        }
    }

    pub fn size_bytes(&self) -> u64 {
        // per layer: 2 * num_kv_heads * head_dim * max_seq_len * sizeof(f16)
        let per_layer =
            2 * self.num_kv_heads as u64 * self.head_dim as u64 * self.max_seq_len as u64 * 2; // f16 = 2 bytes
        per_layer * self.num_layers as u64
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.max_seq_len.saturating_sub(self.current_len)
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

// paged KV cache for memory-efficient long contexts
#[allow(dead_code)]
pub struct PagedKvCache {
    pub page_size: u32, // tokens per page
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    page_table: Vec<Vec<Option<BufferHandle>>>, // [layer][page_idx] -> buffer
}

impl PagedKvCache {
    pub fn new(page_size: u32, num_layers: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            page_table: vec![Vec::new(); num_layers as usize],
        }
    }
}
