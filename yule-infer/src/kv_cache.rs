use std::collections::HashMap;

use yule_core::error::{Result, YuleError};
use yule_gpu::{BufferHandle, ComputeBackend};

pub type PageId = u32;

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

/// INT8-quantized KV cache that uses per-head absmax scaling for 4x memory reduction.
///
/// Instead of storing K/V in f32, this cache stores them as i8 with a per-head
/// scale factor, achieving approximately 4x memory savings at the cost of
/// small quantization error (typically < 1% for normal value ranges).
pub struct QuantizedKvCache {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub current_len: u32,
    /// INT8 quantized key storage: [layer][max_seq * kv_heads * head_dim]
    key_data: Vec<Vec<i8>>,
    /// INT8 quantized value storage: [layer][max_seq * kv_heads * head_dim]
    value_data: Vec<Vec<i8>>,
    /// Per-head, per-position scale factors for keys: [layer][max_seq * kv_heads]
    key_scales: Vec<Vec<f32>>,
    /// Per-head, per-position scale factors for values: [layer][max_seq * kv_heads]
    value_scales: Vec<Vec<f32>>,
}

impl QuantizedKvCache {
    /// Create a new quantized KV cache with the given dimensions.
    pub fn new(num_layers: u32, num_kv_heads: u32, head_dim: u32, max_seq_len: u32) -> Self {
        let kv_len = max_seq_len as usize * num_kv_heads as usize * head_dim as usize;
        let scale_len = max_seq_len as usize * num_kv_heads as usize;

        let mut key_data = Vec::with_capacity(num_layers as usize);
        let mut value_data = Vec::with_capacity(num_layers as usize);
        let mut key_scales = Vec::with_capacity(num_layers as usize);
        let mut value_scales = Vec::with_capacity(num_layers as usize);

        for _ in 0..num_layers {
            key_data.push(vec![0i8; kv_len]);
            value_data.push(vec![0i8; kv_len]);
            key_scales.push(vec![0.0f32; scale_len]);
            value_scales.push(vec![0.0f32; scale_len]);
        }

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            key_data,
            value_data,
            key_scales,
            value_scales,
        }
    }

    /// Write K/V at the given position, quantizing f32 to int8 with per-head absmax scaling.
    ///
    /// `k` and `v` must each have `num_kv_heads * head_dim` elements.
    pub fn write_kv(&mut self, layer: u32, pos: u32, k: &[f32], v: &[f32]) {
        let heads = self.num_kv_heads as usize;
        let dim = self.head_dim as usize;
        let stride = heads * dim;
        assert_eq!(k.len(), stride, "k length must be num_kv_heads * head_dim");
        assert_eq!(v.len(), stride, "v length must be num_kv_heads * head_dim");
        assert!(
            (pos as usize) < self.max_seq_len as usize,
            "position exceeds max_seq_len"
        );

        let layer = layer as usize;
        let data_offset = pos as usize * stride;
        let scale_offset = pos as usize * heads;

        // Quantize each head independently
        for h in 0..heads {
            let head_start = h * dim;
            let head_end = head_start + dim;

            // Key quantization
            let k_head = &k[head_start..head_end];
            let k_absmax = k_head.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let k_scale = if k_absmax > 0.0 {
                k_absmax / 127.0
            } else {
                1.0
            };
            self.key_scales[layer][scale_offset + h] = k_scale;
            for (d, &kv) in k_head.iter().enumerate() {
                let quantized = (kv / k_scale).round().clamp(-127.0, 127.0) as i8;
                self.key_data[layer][data_offset + head_start + d] = quantized;
            }

            // Value quantization
            let v_head = &v[head_start..head_end];
            let v_absmax = v_head.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let v_scale = if v_absmax > 0.0 {
                v_absmax / 127.0
            } else {
                1.0
            };
            self.value_scales[layer][scale_offset + h] = v_scale;
            for (d, &vv) in v_head.iter().enumerate() {
                let quantized = (vv / v_scale).round().clamp(-127.0, 127.0) as i8;
                self.value_data[layer][data_offset + head_start + d] = quantized;
            }
        }

        if pos + 1 > self.current_len {
            self.current_len = pos + 1;
        }
    }

    /// Read K at the given position, dequantizing int8 to f32.
    /// Returns `num_kv_heads * head_dim` f32 values.
    pub fn read_k(&self, layer: u32, pos: u32) -> Vec<f32> {
        let heads = self.num_kv_heads as usize;
        let dim = self.head_dim as usize;
        let stride = heads * dim;
        let data_offset = pos as usize * stride;
        let scale_offset = pos as usize * heads;
        let layer = layer as usize;

        let mut result = vec![0.0f32; stride];
        for h in 0..heads {
            let scale = self.key_scales[layer][scale_offset + h];
            let head_start = h * dim;
            for d in 0..dim {
                result[head_start + d] =
                    self.key_data[layer][data_offset + head_start + d] as f32 * scale;
            }
        }
        result
    }

    /// Read V at the given position, dequantizing int8 to f32.
    /// Returns `num_kv_heads * head_dim` f32 values.
    pub fn read_v(&self, layer: u32, pos: u32) -> Vec<f32> {
        let heads = self.num_kv_heads as usize;
        let dim = self.head_dim as usize;
        let stride = heads * dim;
        let data_offset = pos as usize * stride;
        let scale_offset = pos as usize * heads;
        let layer = layer as usize;

        let mut result = vec![0.0f32; stride];
        for h in 0..heads {
            let scale = self.value_scales[layer][scale_offset + h];
            let head_start = h * dim;
            for d in 0..dim {
                result[head_start + d] =
                    self.value_data[layer][data_offset + head_start + d] as f32 * scale;
            }
        }
        result
    }

    /// Size in bytes of the quantized KV cache.
    /// This is approximately 4x smaller than the equivalent f32 KV cache.
    pub fn size_bytes(&self) -> u64 {
        let kv_len = self.max_seq_len as u64 * self.num_kv_heads as u64 * self.head_dim as u64;
        let scale_len = self.max_seq_len as u64 * self.num_kv_heads as u64;
        // Per layer: 2 * (kv_len * 1 byte for i8 + scale_len * 4 bytes for f32)
        let per_layer = 2 * (kv_len + scale_len * 4);
        per_layer * self.num_layers as u64
    }

    /// Clear all cached KV data.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Number of remaining token positions available.
    pub fn remaining_tokens(&self) -> u32 {
        self.max_seq_len.saturating_sub(self.current_len)
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

    pub fn allocate_page(&mut self, backend: &dyn ComputeBackend, layer: u32) -> Result<u32> {
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

pub struct SequenceState {
    pub seq_id: u64,
    /// `page_tables[layer][logical_page_idx]` maps to a physical `PageId`.
    pub page_tables: Vec<Vec<PageId>>,
    pub token_count: u32,
    pub max_tokens: u32,
}

pub struct PagedKvManager {
    page_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_layers: u32,
    free_pages: Vec<PageId>,
    /// `pages[layer][page_id]` -> `(K buffer, V buffer)` for that page.
    pages: Vec<HashMap<PageId, (BufferHandle, BufferHandle)>>,
    #[allow(dead_code)] // reserved for dynamic page allocation beyond initial pool
    next_page_id: PageId,
    #[allow(dead_code)] // used for capacity reporting
    total_pages: u32,
}

impl PagedKvManager {
    pub fn new(
        backend: &dyn ComputeBackend,
        page_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        num_layers: u32,
        max_pages: u32,
    ) -> Result<Self> {
        let page_elements = page_size as usize * num_kv_heads as usize * head_dim as usize;
        let buf_size = page_elements * 4; // f32

        let mut pages = Vec::with_capacity(num_layers as usize);
        let mut free_pages = Vec::with_capacity(max_pages as usize);

        for _ in 0..num_layers {
            let mut layer_pages = HashMap::new();
            for page_id in 0..max_pages {
                let k_buf = backend.allocate(buf_size)?;
                let v_buf = backend.allocate(buf_size)?;
                layer_pages.insert(page_id, (k_buf, v_buf));
            }
            pages.push(layer_pages);
        }

        // All pages start free
        for page_id in (0..max_pages).rev() {
            free_pages.push(page_id);
        }

        Ok(Self {
            page_size,
            num_kv_heads,
            head_dim,
            num_layers,
            free_pages,
            pages,
            next_page_id: max_pages,
            total_pages: max_pages,
        })
    }

    /// Allocate a new sequence with empty page tables.
    pub fn create_sequence(&mut self, seq_id: u64, max_tokens: u32) -> Result<SequenceState> {
        Ok(SequenceState {
            seq_id,
            page_tables: vec![Vec::new(); self.num_layers as usize],
            token_count: 0,
            max_tokens,
        })
    }

    /// Allocate a physical page from the free pool, returning its `PageId`.
    fn alloc_page(&mut self) -> Result<PageId> {
        self.free_pages
            .pop()
            .ok_or_else(|| YuleError::Inference("paged KV cache exhausted: no free pages".into()))
    }

    /// Append a token to a sequence, allocating new pages as needed.
    pub fn append_token(
        &mut self,
        backend: &dyn ComputeBackend,
        seq: &mut SequenceState,
        layer: u32,
        k_data: &BufferHandle,
        v_data: &BufferHandle,
    ) -> Result<()> {
        if seq.token_count >= seq.max_tokens {
            return Err(YuleError::Inference(format!(
                "sequence {} reached max tokens {}",
                seq.seq_id, seq.max_tokens
            )));
        }

        // Use the current position BEFORE any increment for consistent page allocation
        // across all layers. token_count represents the next write position, and it's
        // only incremented after the last layer writes.
        let pos = if layer == 0 {
            seq.token_count
        } else {
            // For non-zero layers, we need the position that was used for layer 0.
            // Since layer 0 already incremented, subtract 1.
            seq.token_count - 1
        };

        let logical_page = pos / self.page_size;
        let offset_in_page = pos % self.page_size;
        let layer_idx = layer as usize;

        // Allocate a new page if needed
        if logical_page as usize >= seq.page_tables[layer_idx].len() {
            let page_id = self.alloc_page()?;
            seq.page_tables[layer_idx].push(page_id);
        }

        let phys_page = seq.page_tables[layer_idx][logical_page as usize];
        let page_bufs_w = &self.pages[layer_idx][&phys_page];

        let kv_stride = self.num_kv_heads as usize * self.head_dim as usize;
        let byte_offset = offset_in_page as usize * kv_stride * 4;
        let byte_size = kv_stride * 4;

        backend.copy_buffer_offset(k_data, &page_bufs_w.0, 0, byte_offset, byte_size)?;
        backend.copy_buffer_offset(v_data, &page_bufs_w.1, 0, byte_offset, byte_size)?;

        // Only bump token_count when writing layer 0
        if layer == 0 {
            seq.token_count += 1;
        }

        Ok(())
    }

    /// Get the K/V buffers and byte offset for a given sequence position.
    pub fn get_kv_at(
        &self,
        seq: &SequenceState,
        layer: u32,
        position: u32,
    ) -> Result<(&BufferHandle, &BufferHandle, usize)> {
        let logical_page = position / self.page_size;
        let offset_in_page = position % self.page_size;
        let layer_idx = layer as usize;

        let phys_page = *seq
            .page_tables
            .get(layer_idx)
            .and_then(|pt| pt.get(logical_page as usize))
            .ok_or_else(|| {
                YuleError::Inference(format!(
                    "no page for layer {layer} position {position} in sequence {}",
                    seq.seq_id
                ))
            })?;

        let page_bufs = self.pages[layer_idx]
            .get(&phys_page)
            .ok_or_else(|| YuleError::Inference(format!("physical page {phys_page} not found")))?;

        let kv_stride = self.num_kv_heads as usize * self.head_dim as usize;
        let byte_offset = offset_in_page as usize * kv_stride * 4;

        Ok((&page_bufs.0, &page_bufs.1, byte_offset))
    }

    /// Free all pages for a completed sequence, returning them to the pool.
    pub fn free_sequence(&mut self, seq: &SequenceState) {
        for layer_pt in &seq.page_tables {
            for &page_id in layer_pt {
                // Only return to free pool if not already free (dedup for shared pages)
                if !self.free_pages.contains(&page_id) {
                    self.free_pages.push(page_id);
                }
            }
        }
    }

    /// Get number of free pages remaining.
    pub fn free_page_count(&self) -> u32 {
        self.free_pages.len() as u32
    }

    /// Share prefix pages between sequences (copy-on-write).
    ///
    /// The target sequence's page tables are set to reference the same physical
    /// pages as the source for the prefix region. No data is copied.
    pub fn share_prefix(
        &mut self,
        source: &SequenceState,
        target: &mut SequenceState,
        prefix_tokens: u32,
    ) -> Result<()> {
        let prefix_pages = prefix_tokens.div_ceil(self.page_size);

        for layer_idx in 0..self.num_layers as usize {
            if source.page_tables[layer_idx].len() < prefix_pages as usize {
                return Err(YuleError::Inference(format!(
                    "source sequence {} has only {} pages in layer {}, need {prefix_pages}",
                    source.seq_id,
                    source.page_tables[layer_idx].len(),
                    layer_idx,
                )));
            }
            // Share the physical page ids (copy-on-write semantics)
            target.page_tables[layer_idx] =
                source.page_tables[layer_idx][..prefix_pages as usize].to_vec();
        }

        target.token_count = prefix_tokens;
        Ok(())
    }
}

/// Streaming KV cache with attention sink retention.
/// Always keeps the first `num_sink_tokens` in cache, plus a sliding window
/// of the most recent tokens. Enables infinite-length generation with fixed memory.
///
/// Layout: `[sink_0, sink_1, ..., sink_N, window_0, window_1, ..., window_W]`
///
/// The attention sink phenomenon: the first few tokens (typically 4) receive
/// disproportionate attention weight (measured at 148x concentration ratio).
/// When these tokens are evicted from a sliding window KV cache, output quality
/// collapses. This cache prevents that by never evicting sink tokens.
///
/// Reference: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
pub struct StreamingKvCache {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_sink_tokens: u32,
    pub window_size: u32,
    pub total_capacity: u32,
    pub tokens_seen: u64,
    /// Per-layer key storage: `[total_capacity * num_kv_heads * head_dim]`
    key_data: Vec<Vec<f32>>,
    /// Per-layer value storage: `[total_capacity * num_kv_heads * head_dim]`
    value_data: Vec<Vec<f32>>,
}

impl StreamingKvCache {
    /// Create a new streaming KV cache.
    ///
    /// - `num_sink_tokens`: number of initial tokens to always retain (typically 4).
    /// - `window_size`: number of recent tokens to keep in the sliding window.
    pub fn new(
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        num_sink_tokens: u32,
        window_size: u32,
    ) -> Self {
        let total_capacity = num_sink_tokens + window_size;
        let slot_size = total_capacity as usize * num_kv_heads as usize * head_dim as usize;

        let mut key_data = Vec::with_capacity(num_layers as usize);
        let mut value_data = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            key_data.push(vec![0.0f32; slot_size]);
            value_data.push(vec![0.0f32; slot_size]);
        }

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            num_sink_tokens,
            window_size,
            total_capacity,
            tokens_seen: 0,
            key_data,
            value_data,
        }
    }

    /// Write KV for a new token. If we're past capacity:
    /// - Sink tokens (positions `0..num_sink_tokens`) are NEVER overwritten
    /// - Window tokens rotate: oldest window token is evicted, new one appended
    ///
    /// `k` and `v` must each have `num_kv_heads * head_dim` elements.
    pub fn write_kv(&mut self, layer: u32, k: &[f32], v: &[f32]) {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        assert_eq!(k.len(), stride, "k length must be num_kv_heads * head_dim");
        assert_eq!(v.len(), stride, "v length must be num_kv_heads * head_dim");
        assert!(
            (layer as usize) < self.num_layers as usize,
            "layer index out of range"
        );

        let layer_idx = layer as usize;
        let cap = self.total_capacity as usize;
        let sink = self.num_sink_tokens as usize;
        // For layer 0 we use tokens_seen directly; for subsequent layers the count
        // was already bumped by the layer-0 write, so subtract 1 to get the same
        // logical position.
        let seen = if layer == 0 {
            self.tokens_seen as usize
        } else {
            (self.tokens_seen - 1) as usize
        };

        let write_pos = if seen < cap {
            // Still filling slots (sink region or window region)
            seen
        } else {
            // Window is full — shift window slots left by 1, then write at end
            for i in sink..(cap - 1) {
                let dst_start = i * stride;
                let src_start = (i + 1) * stride;
                // Copy within the same vec: use split_at_mut to avoid aliasing
                let (left, right) = self.key_data[layer_idx].split_at_mut(src_start);
                left[dst_start..dst_start + stride].copy_from_slice(&right[..stride]);

                let (left, right) = self.value_data[layer_idx].split_at_mut(src_start);
                left[dst_start..dst_start + stride].copy_from_slice(&right[..stride]);
            }
            cap - 1
        };

        let offset = write_pos * stride;
        self.key_data[layer_idx][offset..offset + stride].copy_from_slice(k);
        self.value_data[layer_idx][offset..offset + stride].copy_from_slice(v);

        // Only bump tokens_seen once per token (on layer 0)
        if layer == 0 {
            self.tokens_seen += 1;
        }
    }

    /// Read K values for all cached positions (sinks + window).
    /// Returns a contiguous slice of `[effective_len * num_kv_heads * head_dim]`.
    pub fn read_k(&self, layer: u32) -> &[f32] {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        let len = self.effective_len() as usize * stride;
        &self.key_data[layer as usize][..len]
    }

    /// Read V values for all cached positions.
    /// Returns a contiguous slice of `[effective_len * num_kv_heads * head_dim]`.
    pub fn read_v(&self, layer: u32) -> &[f32] {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        let len = self.effective_len() as usize * stride;
        &self.value_data[layer as usize][..len]
    }

    /// Current effective sequence length for attention computation.
    /// `min(tokens_seen, total_capacity)`.
    pub fn effective_len(&self) -> u32 {
        std::cmp::min(self.tokens_seen as u32, self.total_capacity)
    }

    /// Total tokens that have been processed (may exceed capacity).
    pub fn tokens_seen(&self) -> u64 {
        self.tokens_seen
    }

    /// Size in bytes of the streaming KV cache.
    pub fn size_bytes(&self) -> u64 {
        // Per layer: 2 (K+V) * total_capacity * num_kv_heads * head_dim * 4 bytes
        let per_layer =
            2u64 * self.total_capacity as u64 * self.num_kv_heads as u64 * self.head_dim as u64 * 4;
        per_layer * self.num_layers as u64
    }

    /// Reset all state.
    pub fn clear(&mut self) {
        self.tokens_seen = 0;
        for layer_idx in 0..self.num_layers as usize {
            self.key_data[layer_idx].fill(0.0);
            self.value_data[layer_idx].fill(0.0);
        }
    }

    /// Check if we're in streaming mode (more tokens seen than capacity).
    pub fn is_streaming(&self) -> bool {
        self.tokens_seen > self.total_capacity as u64
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

    /// Minimal mock backend for unit tests — only allocate and copy_buffer_offset are needed.
    struct MockBackend {
        next_id: std::sync::atomic::AtomicU64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next_id: std::sync::atomic::AtomicU64::new(1),
            }
        }
    }

    impl ComputeBackend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }
        fn device_info(&self) -> yule_gpu::DeviceInfo {
            yule_gpu::DeviceInfo {
                name: "mock".into(),
                backend: yule_gpu::BackendKind::Cpu,
                memory_bytes: 0,
                compute_units: 0,
            }
        }
        fn allocate(&self, _size_bytes: usize) -> Result<BufferHandle> {
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(BufferHandle(id))
        }
        fn free(&self, _handle: BufferHandle) -> Result<()> {
            Ok(())
        }
        fn matmul(
            &self,
            _a: &BufferHandle,
            _b: &BufferHandle,
            _out: &BufferHandle,
            _m: u32,
            _n: u32,
            _k: u32,
        ) -> Result<()> {
            Ok(())
        }
        fn softmax(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
            Ok(())
        }
        fn rms_norm(
            &self,
            _input: &BufferHandle,
            _weight: &BufferHandle,
            _output: &BufferHandle,
            _size: u32,
            _eps: f32,
        ) -> Result<()> {
            Ok(())
        }
        fn rope(
            &self,
            _q: &BufferHandle,
            _k: &BufferHandle,
            _pos: u32,
            _head_dim: u32,
            _freq_base: f32,
            _n_heads_q: u32,
            _n_heads_k: u32,
        ) -> Result<()> {
            Ok(())
        }
        fn silu(&self, _input: &BufferHandle, _output: &BufferHandle, _size: u32) -> Result<()> {
            Ok(())
        }
        fn element_mul(
            &self,
            _a: &BufferHandle,
            _b: &BufferHandle,
            _output: &BufferHandle,
            _size: u32,
        ) -> Result<()> {
            Ok(())
        }
        fn add(
            &self,
            _a: &BufferHandle,
            _b: &BufferHandle,
            _output: &BufferHandle,
            _size: u32,
        ) -> Result<()> {
            Ok(())
        }
        fn copy_to_device(&self, _data: &[u8], _handle: &BufferHandle) -> Result<()> {
            Ok(())
        }
        fn copy_from_device(&self, _handle: &BufferHandle, _data: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn copy_buffer(
            &self,
            _src: &BufferHandle,
            _dst: &BufferHandle,
            _size: usize,
        ) -> Result<()> {
            Ok(())
        }
        fn copy_buffer_offset(
            &self,
            _src: &BufferHandle,
            _dst: &BufferHandle,
            _src_offset: usize,
            _dst_offset: usize,
            _size: usize,
        ) -> Result<()> {
            Ok(())
        }
        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_paged_kv_allocate_and_append() {
        let backend = MockBackend::new();
        let page_size = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let num_layers = 2;
        let max_pages = 8;

        let mut mgr = PagedKvManager::new(
            &backend,
            page_size,
            num_kv_heads,
            head_dim,
            num_layers,
            max_pages,
        )
        .unwrap();

        let mut seq = mgr.create_sequence(1, 16).unwrap();
        assert_eq!(seq.token_count, 0);
        assert!(seq.page_tables.iter().all(|pt| pt.is_empty()));

        // Append tokens — a new page should be allocated on the first token
        let k = backend.allocate(64).unwrap();
        let v = backend.allocate(64).unwrap();

        for layer in 0..num_layers {
            mgr.append_token(&backend, &mut seq, layer, &k, &v).unwrap();
        }
        assert_eq!(seq.token_count, 1);
        // Each layer should have exactly one page allocated
        assert_eq!(seq.page_tables[0].len(), 1);
        assert_eq!(seq.page_tables[1].len(), 1);

        // Append more tokens to fill the first page and spill into a second
        for _ in 1..5 {
            for layer in 0..num_layers {
                mgr.append_token(&backend, &mut seq, layer, &k, &v).unwrap();
            }
        }
        assert_eq!(seq.token_count, 5);
        // 5 tokens with page_size=4 → 2 pages per layer
        assert_eq!(seq.page_tables[0].len(), 2);

        // Verify get_kv_at returns valid references
        let (_k_buf, _v_buf, offset) = mgr.get_kv_at(&seq, 0, 0).unwrap();
        assert_eq!(offset, 0);

        let (_k_buf, _v_buf, offset) = mgr.get_kv_at(&seq, 0, 4).unwrap();
        // token 4 is offset 0 within the second page
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_paged_kv_free_reclaims_pages() {
        let backend = MockBackend::new();
        let max_pages = 8;
        let mut mgr = PagedKvManager::new(&backend, 4, 2, 8, 1, max_pages).unwrap();

        let initial_free = mgr.free_page_count();
        assert_eq!(initial_free, max_pages);

        let mut seq = mgr.create_sequence(1, 16).unwrap();
        let k = backend.allocate(64).unwrap();
        let v = backend.allocate(64).unwrap();

        // Append 5 tokens (layer 0 only) → uses 2 pages
        for _ in 0..5 {
            mgr.append_token(&backend, &mut seq, 0, &k, &v).unwrap();
        }
        assert_eq!(mgr.free_page_count(), max_pages - 2);

        mgr.free_sequence(&seq);
        assert_eq!(mgr.free_page_count(), max_pages);
    }

    #[test]
    fn test_paged_kv_share_prefix() {
        let backend = MockBackend::new();
        let page_size = 4;
        let num_layers = 2;
        let max_pages = 16;

        let mut mgr =
            PagedKvManager::new(&backend, page_size, 2, 8, num_layers, max_pages).unwrap();

        let mut source = mgr.create_sequence(1, 32).unwrap();
        let k = backend.allocate(64).unwrap();
        let v = backend.allocate(64).unwrap();

        // Fill source with 8 tokens (2 pages per layer)
        for _ in 0..8 {
            for layer in 0..num_layers {
                mgr.append_token(&backend, &mut source, layer, &k, &v)
                    .unwrap();
            }
        }
        assert_eq!(source.token_count, 8);

        let mut target = mgr.create_sequence(2, 32).unwrap();
        mgr.share_prefix(&source, &mut target, 8).unwrap();

        // Target should now have the same page IDs as source
        assert_eq!(target.token_count, 8);
        for layer_idx in 0..num_layers as usize {
            assert_eq!(source.page_tables[layer_idx], target.page_tables[layer_idx],);
        }

        // Target can continue appending independently
        for layer in 0..num_layers {
            mgr.append_token(&backend, &mut target, layer, &k, &v)
                .unwrap();
        }
        assert_eq!(target.token_count, 9);
        // Target should have a third page in each layer now
        assert_eq!(target.page_tables[0].len(), 3);
        // Source should still have only 2 pages
        assert_eq!(source.page_tables[0].len(), 2);
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

    #[test]
    fn test_kv_quantize_roundtrip() {
        let num_layers = 2;
        let num_kv_heads = 4;
        let head_dim = 64;
        let max_seq_len = 128;

        let mut cache = QuantizedKvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let stride = num_kv_heads as usize * head_dim as usize;

        // Generate test data with typical value range [-2.0, 2.0]
        let mut k = vec![0.0f32; stride];
        let mut v = vec![0.0f32; stride];
        for i in 0..stride {
            k[i] = (i as f32 / stride as f32) * 4.0 - 2.0; // [-2.0, 2.0]
            v[i] = ((stride - i) as f32 / stride as f32) * 4.0 - 2.0;
        }

        // Write at position 0, layer 0
        cache.write_kv(0, 0, &k, &v);
        assert_eq!(cache.current_len, 1);
        assert_eq!(cache.remaining_tokens(), max_seq_len - 1);

        // Read back and verify roundtrip error < 1%
        let k_out = cache.read_k(0, 0);
        let v_out = cache.read_v(0, 0);
        assert_eq!(k_out.len(), stride);
        assert_eq!(v_out.len(), stride);

        for i in 0..stride {
            if k[i].abs() > 0.01 {
                let k_err = (k_out[i] - k[i]).abs() / k[i].abs();
                assert!(
                    k_err < 0.01,
                    "K roundtrip error at {}: original={}, got={}, err={}",
                    i,
                    k[i],
                    k_out[i],
                    k_err
                );
            }
            if v[i].abs() > 0.01 {
                let v_err = (v_out[i] - v[i]).abs() / v[i].abs();
                assert!(
                    v_err < 0.01,
                    "V roundtrip error at {}: original={}, got={}, err={}",
                    i,
                    v[i],
                    v_out[i],
                    v_err
                );
            }
        }

        // Test clear
        cache.clear();
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.remaining_tokens(), max_seq_len);
    }

    #[test]
    fn test_streaming_kv_sink_retention() {
        // 4 sinks + 6 window = 10 total capacity, 2 layers, 2 heads, dim 4
        let num_layers = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = 4u32;
        let num_sink_tokens = 4u32;
        let window_size = 6u32;
        let stride = (num_kv_heads * head_dim) as usize;

        let mut cache = StreamingKvCache::new(
            num_layers,
            num_kv_heads,
            head_dim,
            num_sink_tokens,
            window_size,
        );

        // Generate unique KV data for each token: value = token_index + 1
        let make_kv = |token: usize| -> Vec<f32> { vec![(token + 1) as f32; stride] };

        // Write 100 tokens across all layers
        for t in 0..100usize {
            let kv = make_kv(t);
            for layer in 0..num_layers {
                cache.write_kv(layer, &kv, &kv);
            }
        }

        // Verify sinks 0-3 are unchanged from their original values
        for layer in 0..num_layers {
            let k = cache.read_k(layer);
            for sink_idx in 0..num_sink_tokens as usize {
                let expected = (sink_idx + 1) as f32;
                let offset = sink_idx * stride;
                for d in 0..stride {
                    assert_eq!(
                        k[offset + d],
                        expected,
                        "Sink token {} corrupted in layer {} at dim {}",
                        sink_idx,
                        layer,
                        d
                    );
                }
            }
        }
    }

    #[test]
    fn test_streaming_kv_window_rotation() {
        let num_layers = 1u32;
        let num_kv_heads = 1u32;
        let head_dim = 2u32;
        let num_sink_tokens = 4u32;
        let window_size = 6u32;
        let stride = (num_kv_heads * head_dim) as usize;

        let mut cache = StreamingKvCache::new(
            num_layers,
            num_kv_heads,
            head_dim,
            num_sink_tokens,
            window_size,
        );

        let make_kv = |token: usize| -> Vec<f32> { vec![(token + 1) as f32; stride] };

        // Write 100 tokens
        for t in 0..100usize {
            let kv = make_kv(t);
            cache.write_kv(0, &kv, &kv);
        }

        // Window should contain the most recent 6 tokens: 94..100 (values 95..101)
        let k = cache.read_k(0);
        let v = cache.read_v(0);
        let total_cap = (num_sink_tokens + window_size) as usize;
        assert_eq!(k.len(), total_cap * stride);

        for w in 0..window_size as usize {
            let slot = num_sink_tokens as usize + w;
            let expected_token = 100 - window_size as usize + w; // 94, 95, ..., 99
            let expected_val = (expected_token + 1) as f32;
            let offset = slot * stride;
            for d in 0..stride {
                assert_eq!(
                    k[offset + d],
                    expected_val,
                    "Window slot {} should have token {} (val {}), got {}",
                    w,
                    expected_token,
                    expected_val,
                    k[offset + d]
                );
                assert_eq!(v[offset + d], expected_val);
            }
        }
    }

    #[test]
    fn test_streaming_kv_effective_len() {
        let mut cache = StreamingKvCache::new(1, 1, 2, 4, 6);
        let stride = 2;
        let kv = vec![1.0f32; stride];

        // Before capacity: effective_len = tokens_seen
        for t in 0..10u64 {
            assert_eq!(cache.effective_len(), t as u32);
            cache.write_kv(0, &kv, &kv);
        }
        assert_eq!(cache.effective_len(), 10); // == total_capacity

        // After capacity: effective_len stays at total_capacity
        for _ in 0..20 {
            cache.write_kv(0, &kv, &kv);
        }
        assert_eq!(cache.effective_len(), 10);
        assert_eq!(cache.tokens_seen(), 30);
    }

    #[test]
    fn test_streaming_kv_is_streaming() {
        let mut cache = StreamingKvCache::new(1, 1, 2, 4, 6);
        let kv = vec![1.0f32; 2];

        // Under capacity: not streaming
        for _ in 0..10 {
            assert!(!cache.is_streaming());
            cache.write_kv(0, &kv, &kv);
        }
        // At exactly capacity (10 seen, 10 cap): not streaming
        assert!(!cache.is_streaming());

        // Over capacity: streaming
        cache.write_kv(0, &kv, &kv);
        assert!(cache.is_streaming());
    }

    #[test]
    fn test_streaming_kv_size() {
        let num_layers = 2u32;
        let num_kv_heads = 4u32;
        let head_dim = 64u32;
        let num_sink_tokens = 4u32;
        let window_size = 1020u32;

        let cache = StreamingKvCache::new(
            num_layers,
            num_kv_heads,
            head_dim,
            num_sink_tokens,
            window_size,
        );

        let total_cap = (num_sink_tokens + window_size) as u64; // 1024
        // per layer: 2 * 1024 * 4 * 64 * 4 = 2,097,152
        let expected =
            2u64 * total_cap * num_kv_heads as u64 * head_dim as u64 * 4 * num_layers as u64;
        assert_eq!(cache.size_bytes(), expected);
    }

    #[test]
    fn test_kv_quantized_size() {
        let num_layers = 2;
        let num_kv_heads = 4;
        let head_dim = 64;
        let max_seq_len = 1024;

        let f32_cache = KvCache {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
            key_buffers: Vec::new(),
            value_buffers: Vec::new(),
        };
        let f32_size = f32_cache.size_bytes();

        let q_cache = QuantizedKvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let q_size = q_cache.size_bytes();

        // Quantized cache should be significantly smaller than f32
        assert!(
            q_size < f32_size,
            "quantized size {} should be less than f32 size {}",
            q_size,
            f32_size
        );

        // Verify it's at least 3x smaller
        assert!(
            f32_size / q_size >= 3,
            "expected at least 3x reduction, got {:.2}x (f32={}, q={})",
            f32_size as f64 / q_size as f64,
            f32_size,
            q_size
        );
    }
}
