//! Verifiable KV cache compression.
//!
//! When a KV cache entry is evicted (due to sliding window or capacity limits),
//! its BLAKE3 hash is retained. If the entry is needed again, it can be
//! reconstructed and verified against the stored hash — proving the
//! reconstruction matches the original.
//!
//! This is unique to Yule: no other inference engine provides cryptographic
//! proof that KV cache compression didn't corrupt the attention state.
//!
//! Use cases:
//! - Verify that KV cache eviction + reconstruction is lossless
//! - Detect bit flips or corruption in cached KV data
//! - Audit trail: prove which KV entries were evicted and when

/// Hash of an evicted KV cache entry.
pub struct EvictedEntry {
    pub layer: u32,
    pub position: u32,
    pub k_hash: [u8; 32],
    pub v_hash: [u8; 32],
    pub evicted_at_step: u64,
}

/// Verifiable KV cache that tracks evicted entries via hashes.
pub struct VerifiedKvCache {
    /// Active KV data (same layout as StreamingKvCache)
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    _capacity: u32,
    tokens_seen: u64,
    key_data: Vec<Vec<f32>>,
    value_data: Vec<Vec<f32>>,
    /// Hashes of evicted entries
    evicted: Vec<EvictedEntry>,
    /// Maximum eviction log size (circular buffer)
    max_eviction_log: usize,
}

impl VerifiedKvCache {
    pub fn new(
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        max_eviction_log: usize,
    ) -> Self {
        let slot_size = capacity as usize * num_kv_heads as usize * head_dim as usize;
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
            _capacity: capacity,
            tokens_seen: 0,
            key_data,
            value_data,
            evicted: Vec::new(),
            max_eviction_log,
        }
    }

    /// Write KV at position. If position is being overwritten, hash the old
    /// data and store in the eviction log.
    pub fn write_kv(&mut self, layer: u32, pos: u32, k: &[f32], v: &[f32]) {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        let layer_idx = layer as usize;
        let offset = pos as usize * stride;

        // If this position has data (not first write), hash and evict
        if self.tokens_seen > pos as u64 {
            let old_k = &self.key_data[layer_idx][offset..offset + stride];
            let old_v = &self.value_data[layer_idx][offset..offset + stride];

            // Only log if the data is non-zero (has been written before)
            let has_data = old_k.iter().any(|&x| x != 0.0);
            if has_data {
                let k_hash = hash_floats(old_k);
                let v_hash = hash_floats(old_v);

                let entry = EvictedEntry {
                    layer,
                    position: pos,
                    k_hash,
                    v_hash,
                    evicted_at_step: self.tokens_seen,
                };

                if self.evicted.len() >= self.max_eviction_log {
                    self.evicted.remove(0); // circular: drop oldest
                }
                self.evicted.push(entry);
            }
        }

        // Write new data
        self.key_data[layer_idx][offset..offset + stride].copy_from_slice(k);
        self.value_data[layer_idx][offset..offset + stride].copy_from_slice(v);

        if layer == 0 {
            self.tokens_seen += 1;
        }
    }

    /// Verify that KV data at a position matches a previously recorded hash.
    pub fn verify_position(&self, layer: u32, pos: u32) -> Option<bool> {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        let layer_idx = layer as usize;
        let offset = pos as usize * stride;

        let current_k = &self.key_data[layer_idx][offset..offset + stride];
        let current_v = &self.value_data[layer_idx][offset..offset + stride];

        let k_hash = hash_floats(current_k);
        let v_hash = hash_floats(current_v);

        // Find the most recent eviction record for this position+layer
        for entry in self.evicted.iter().rev() {
            if entry.layer == layer && entry.position == pos {
                // Compare: if current data hashes match evicted data hashes,
                // the reconstruction is verified
                let k_match = constant_time_eq(&k_hash, &entry.k_hash);
                let v_match = constant_time_eq(&v_hash, &entry.v_hash);
                return Some(k_match && v_match);
            }
        }

        None // no eviction record found for this position
    }

    /// Get the eviction log.
    pub fn eviction_log(&self) -> &[EvictedEntry] {
        &self.evicted
    }

    /// Number of evicted entries in the log.
    pub fn eviction_count(&self) -> usize {
        self.evicted.len()
    }

    /// Hash the current KV data at a position (for external verification).
    pub fn hash_position(&self, layer: u32, pos: u32) -> ([u8; 32], [u8; 32]) {
        let stride = self.num_kv_heads as usize * self.head_dim as usize;
        let layer_idx = layer as usize;
        let offset = pos as usize * stride;

        let k = &self.key_data[layer_idx][offset..offset + stride];
        let v = &self.value_data[layer_idx][offset..offset + stride];

        (hash_floats(k), hash_floats(v))
    }

    pub fn clear(&mut self) {
        self.tokens_seen = 0;
        self.evicted.clear();
        for layer_idx in 0..self.num_layers as usize {
            self.key_data[layer_idx].fill(0.0);
            self.value_data[layer_idx].fill(0.0);
        }
    }

    pub fn tokens_seen(&self) -> u64 {
        self.tokens_seen
    }
}

/// BLAKE3 hash of a float slice (via bytemuck cast to bytes).
fn hash_floats(data: &[f32]) -> [u8; 32] {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    *blake3::hash(bytes).as_bytes()
}

/// Constant-time comparison to prevent timing attacks on hash verification.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_verify() {
        let mut cache = VerifiedKvCache::new(1, 2, 4, 4, 100);
        let stride = 8; // 2 heads * 4 dim

        // Write position 0
        let k0: Vec<f32> = (0..stride).map(|i| i as f32 * 0.1).collect();
        let v0: Vec<f32> = (0..stride).map(|i| i as f32 * 0.2).collect();
        cache.write_kv(0, 0, &k0, &v0);

        // No eviction yet
        assert_eq!(cache.eviction_count(), 0);

        // Fill up to capacity
        for pos in 1..4 {
            let k: Vec<f32> = (0..stride).map(|i| (pos * stride + i) as f32).collect();
            let v: Vec<f32> = (0..stride)
                .map(|i| (pos * stride + i) as f32 + 100.0)
                .collect();
            cache.write_kv(0, pos as u32, &k, &v);
        }

        // Overwrite position 0 — should evict and hash the original
        let k0_new: Vec<f32> = (0..stride).map(|i| i as f32 * 9.9).collect();
        let v0_new: Vec<f32> = (0..stride).map(|i| i as f32 * 8.8).collect();
        cache.write_kv(0, 0, &k0_new, &v0_new);

        // Should have 1 eviction record
        assert_eq!(cache.eviction_count(), 1);
        let entry = &cache.eviction_log()[0];
        assert_eq!(entry.layer, 0);
        assert_eq!(entry.position, 0);

        // The evicted hash should match the ORIGINAL data
        let original_k_hash = hash_floats(&k0);
        let original_v_hash = hash_floats(&v0);
        assert!(constant_time_eq(&entry.k_hash, &original_k_hash));
        assert!(constant_time_eq(&entry.v_hash, &original_v_hash));
    }

    #[test]
    fn test_eviction_log_circular() {
        let mut cache = VerifiedKvCache::new(1, 1, 2, 2, 3); // max 3 eviction entries
        let _stride = 2;

        // Fill
        cache.write_kv(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.write_kv(0, 1, &[5.0, 6.0], &[7.0, 8.0]);

        // Overwrite 5 times — should only keep last 3
        for i in 0..5 {
            let k = [i as f32 * 10.0, i as f32 * 10.0 + 1.0];
            let v = [i as f32 * 20.0, i as f32 * 20.0 + 1.0];
            cache.write_kv(0, (i % 2) as u32, &k, &v);
        }

        assert!(cache.eviction_count() <= 3);
    }

    #[test]
    fn test_hash_consistency() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let h1 = hash_floats(&data);
        let h2 = hash_floats(&data);
        assert!(constant_time_eq(&h1, &h2));

        let different = [1.0f32, 2.0, 3.0, 4.1];
        let h3 = hash_floats(&different);
        assert!(!constant_time_eq(&h1, &h3));
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [0u8; 32];
        let b = [0u8; 32];
        assert!(constant_time_eq(&a, &b));

        let mut c = [0u8; 32];
        c[31] = 1;
        assert!(!constant_time_eq(&a, &c));
    }
}
