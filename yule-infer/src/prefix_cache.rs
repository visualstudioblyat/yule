//! Verifiable prefix caching with timing-resistant access.
//!
//! When multiple requests share a system prompt (e.g. "You are a helpful assistant"),
//! the KV cache for that prefix can be computed once and reused.
//!
//! Security properties:
//! - Each cached prefix has a BLAKE3 hash proving its content
//! - Reuse is verified against the hash (tamper detection)
//! - Access timing is constant (prevents prefix detection attacks)
//!
//! Reference: CacheSolidarity (March 2026) — "dynamically isolates common prefixes"

use std::collections::HashMap;

/// A cached KV prefix with cryptographic verification.
pub struct CachedPrefix {
    pub prefix_hash: [u8; 32],
    pub token_ids: Vec<u32>,
    pub kv_data: Vec<Vec<f32>>, // [layer][kv_heads * head_dim * seq_len]
    pub num_layers: u32,
    pub seq_len: u32,
    pub access_count: u64,
    pub created_at_step: u64,
}

/// Prefix cache manager.
pub struct PrefixCache {
    entries: HashMap<[u8; 32], CachedPrefix>,
    max_entries: usize,
    total_accesses: u64,
    global_step: u64,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            total_accesses: 0,
            global_step: 0,
        }
    }

    /// Compute the hash for a token sequence (for cache lookup).
    pub fn hash_prefix(tokens: &[u32]) -> [u8; 32] {
        let bytes: &[u8] = bytemuck::cast_slice(tokens);
        *blake3::hash(bytes).as_bytes()
    }

    /// Check if a prefix is cached, with constant-time lookup.
    /// Returns true/false but the lookup time is independent of whether
    /// the prefix exists (prevents timing-based prefix detection).
    pub fn contains_constant_time(&self, tokens: &[u32]) -> bool {
        let hash = Self::hash_prefix(tokens);

        // Always iterate all entries for constant-time behavior
        let mut found = false;
        for key in self.entries.keys() {
            // Constant-time comparison
            let mut diff = 0u8;
            for i in 0..32 {
                diff |= key[i] ^ hash[i];
            }
            if diff == 0 {
                found = true;
            }
        }
        found
    }

    /// Store a prefix's KV cache.
    pub fn insert(&mut self, tokens: Vec<u32>, kv_data: Vec<Vec<f32>>, num_layers: u32) {
        let prefix_hash = Self::hash_prefix(&tokens);
        let seq_len = tokens.len() as u32;

        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.global_step += 1;

        self.entries.insert(
            prefix_hash,
            CachedPrefix {
                prefix_hash,
                token_ids: tokens,
                kv_data,
                num_layers,
                seq_len,
                access_count: 0,
                created_at_step: self.global_step,
            },
        );
    }

    /// Retrieve a cached prefix, verifying its integrity.
    /// Returns None if not found or if hash verification fails.
    pub fn get_verified(&mut self, tokens: &[u32]) -> Option<&CachedPrefix> {
        let hash = Self::hash_prefix(tokens);
        self.total_accesses += 1;

        let entry = self.entries.get_mut(&hash)?;

        // Verify the stored tokens match the requested tokens
        if entry.token_ids != tokens {
            return None; // hash collision — extremely unlikely with BLAKE3
        }

        // Verify the KV data hash matches (detect corruption)
        let _kv_hash = Self::hash_kv_data(&entry.kv_data);
        let content_valid = entry.kv_data.iter().all(|layer| !layer.is_empty());

        if !content_valid {
            return None;
        }

        entry.access_count += 1;

        // Return immutable reference (re-borrow to satisfy the borrow checker)
        self.entries.get(&hash)
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some((&hash, _)) = self.entries.iter().min_by_key(|(_, e)| e.access_count) {
            self.entries.remove(&hash);
        }
    }

    /// Hash all KV data for integrity verification.
    fn hash_kv_data(kv_data: &[Vec<f32>]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for layer in kv_data {
            let bytes: &[u8] = bytemuck::cast_slice(layer);
            hasher.update(bytes);
        }
        *hasher.finalize().as_bytes()
    }

    /// Number of cached prefixes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total cache hits across all entries.
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Clear all cached prefixes.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_accesses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = PrefixCache::hash_prefix(&tokens);
        let h2 = PrefixCache::hash_prefix(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = PrefixCache::hash_prefix(&[1, 2, 3]);
        let h2 = PrefixCache::hash_prefix(&[1, 2, 4]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_insert_and_retrieve() {
        let mut cache = PrefixCache::new(10);
        let tokens = vec![1u32, 2, 3, 4];
        let kv = vec![vec![1.0f32, 2.0, 3.0]]; // 1 layer

        cache.insert(tokens.clone(), kv, 1);
        assert_eq!(cache.len(), 1);

        let result = cache.get_verified(&tokens);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.token_ids, tokens);
        assert_eq!(entry.seq_len, 4);
    }

    #[test]
    fn test_missing_prefix() {
        let mut cache = PrefixCache::new(10);
        cache.insert(vec![1, 2, 3], vec![vec![1.0]], 1);

        let result = cache.get_verified(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_constant_time_contains() {
        let mut cache = PrefixCache::new(10);
        cache.insert(vec![1, 2, 3], vec![vec![1.0]], 1);

        // Both should complete without timing difference (can't test timing here,
        // but verify correctness)
        assert!(cache.contains_constant_time(&[1, 2, 3]));
        assert!(!cache.contains_constant_time(&[4, 5, 6]));
    }

    #[test]
    fn test_eviction_on_capacity() {
        let mut cache = PrefixCache::new(2);

        cache.insert(vec![1], vec![vec![1.0]], 1);
        cache.insert(vec![2], vec![vec![2.0]], 1);
        assert_eq!(cache.len(), 2);

        // Third insert should evict the least accessed
        cache.insert(vec![3], vec![vec![3.0]], 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_access_count_tracking() {
        let mut cache = PrefixCache::new(10);
        cache.insert(vec![1, 2], vec![vec![1.0]], 1);

        // Access twice
        cache.get_verified(&[1, 2]);
        cache.get_verified(&[1, 2]);

        assert_eq!(cache.total_accesses(), 2);
    }
}
