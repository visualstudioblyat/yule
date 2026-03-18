//! Block-sparse attention for efficient long-context inference.
//! Reference: XAttention (ICML 2025) — 13.5x speedup via antidiagonal scoring.

/// Attention block mask: which blocks of the attention matrix to compute.
pub struct BlockSparseMask {
    pub block_size: usize,
    pub num_blocks_q: usize,  // seq_len_q / block_size
    pub num_blocks_kv: usize, // seq_len_kv / block_size
    /// Which blocks are active. [num_blocks_q][num_blocks_kv] flattened.
    pub active: Vec<bool>,
}

impl BlockSparseMask {
    /// Create a causal mask (lower-triangular blocks).
    pub fn causal(seq_len: usize, block_size: usize) -> Self {
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let mut active = vec![false; num_blocks * num_blocks];
        for qb in 0..num_blocks {
            for kvb in 0..=qb {
                active[qb * num_blocks + kvb] = true;
            }
        }
        Self {
            block_size,
            num_blocks_q: num_blocks,
            num_blocks_kv: num_blocks,
            active,
        }
    }

    /// Create a local window mask (attend to ±window blocks).
    pub fn local_window(seq_len: usize, block_size: usize, window_blocks: usize) -> Self {
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let mut active = vec![false; num_blocks * num_blocks];
        for qb in 0..num_blocks {
            let start = qb.saturating_sub(window_blocks);
            let end = (qb + window_blocks + 1).min(num_blocks);
            for kvb in start..end {
                active[qb * num_blocks + kvb] = true;
            }
        }
        Self {
            block_size,
            num_blocks_q: num_blocks,
            num_blocks_kv: num_blocks,
            active,
        }
    }

    /// Create a strided mask (attend every N blocks for global coverage).
    pub fn strided(seq_len: usize, block_size: usize, stride: usize) -> Self {
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let mut active = vec![false; num_blocks * num_blocks];
        for qb in 0..num_blocks {
            for kvb in (0..num_blocks).step_by(stride) {
                active[qb * num_blocks + kvb] = true;
            }
        }
        Self {
            block_size,
            num_blocks_q: num_blocks,
            num_blocks_kv: num_blocks,
            active,
        }
    }

    /// Combine: local window + strided for hybrid attention.
    pub fn hybrid(seq_len: usize, block_size: usize, window_blocks: usize, stride: usize) -> Self {
        let local = Self::local_window(seq_len, block_size, window_blocks);
        let strided = Self::strided(seq_len, block_size, stride);
        let num_blocks = local.num_blocks_q;
        let mut active = vec![false; num_blocks * num_blocks];
        for i in 0..active.len() {
            active[i] = local.active[i] || strided.active[i];
        }
        Self {
            block_size,
            num_blocks_q: num_blocks,
            num_blocks_kv: num_blocks,
            active,
        }
    }

    /// Is block (q_block, kv_block) active?
    pub fn is_active(&self, q_block: usize, kv_block: usize) -> bool {
        self.active[q_block * self.num_blocks_kv + kv_block]
    }

    /// Fraction of blocks that are active (1.0 = full attention).
    pub fn sparsity(&self) -> f64 {
        let total = self.active.len();
        let active = self.active.iter().filter(|&&b| b).count();
        active as f64 / total as f64
    }

    /// Number of active blocks.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&b| b).count()
    }
}

/// Compute block-sparse attention scores.
/// Only computes Q@K^T for active blocks, skipping masked blocks entirely.
pub fn block_sparse_attention(
    q: &[f32], // [seq_len_q, head_dim]
    k: &[f32], // [seq_len_kv, head_dim]
    v: &[f32], // [seq_len_kv, head_dim]
    mask: &BlockSparseMask,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    // output: [seq_len_q, head_dim]
    let sq = mask.num_blocks_q * mask.block_size;
    let skv = mask.num_blocks_kv * mask.block_size;
    let bs = mask.block_size;
    let mut output = vec![0.0f32; sq * head_dim];

    for qb in 0..mask.num_blocks_q {
        // For this Q block, compute attention only against active KV blocks
        for qi in 0..bs {
            let q_pos = qb * bs + qi;
            if q_pos >= sq {
                break;
            }

            let mut scores = vec![f32::NEG_INFINITY; skv];

            for kvb in 0..mask.num_blocks_kv {
                if !mask.is_active(qb, kvb) {
                    continue;
                }

                for kvi in 0..bs {
                    let kv_pos = kvb * bs + kvi;
                    if kv_pos >= skv {
                        break;
                    }

                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_pos * head_dim + d] * k[kv_pos * head_dim + d];
                    }
                    scores[kv_pos] = dot * scale;
                }
            }

            // Softmax over non-masked positions
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                if *s > f32::NEG_INFINITY {
                    *s = (*s - max_s).exp();
                    sum += *s;
                } else {
                    *s = 0.0;
                }
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in &mut scores {
                    *s *= inv;
                }
            }

            // Weighted sum
            for kv_pos in 0..skv {
                if scores[kv_pos] > 0.0 {
                    for d in 0..head_dim {
                        output[q_pos * head_dim + d] += scores[kv_pos] * v[kv_pos * head_dim + d];
                    }
                }
            }
        }
    }

    output
}

/// Dense (full) attention for reference / testing.
#[allow(dead_code)]
fn dense_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];

    for qi in 0..seq_len {
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        let kv_end = if causal { qi + 1 } else { seq_len };

        for kvi in 0..kv_end {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[kvi * head_dim + d];
            }
            scores[kvi] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            if *s > f32::NEG_INFINITY {
                *s = (*s - max_s).exp();
                sum += *s;
            } else {
                *s = 0.0;
            }
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in &mut scores {
                *s *= inv;
            }
        }

        for kvi in 0..seq_len {
            if scores[kvi] > 0.0 {
                for d in 0..head_dim {
                    output[qi * head_dim + d] += scores[kvi] * v[kvi * head_dim + d];
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_structure() {
        let mask = BlockSparseMask::causal(8, 2);
        assert_eq!(mask.num_blocks_q, 4);
        assert_eq!(mask.num_blocks_kv, 4);
        // Lower triangular: block (i,j) active iff j <= i
        assert!(mask.is_active(0, 0));
        assert!(!mask.is_active(0, 1));
        assert!(!mask.is_active(0, 2));
        assert!(mask.is_active(1, 0));
        assert!(mask.is_active(1, 1));
        assert!(!mask.is_active(1, 2));
        assert!(mask.is_active(2, 0));
        assert!(mask.is_active(2, 1));
        assert!(mask.is_active(2, 2));
        assert!(!mask.is_active(2, 3));
        assert!(mask.is_active(3, 3));
    }

    #[test]
    fn local_window_mask() {
        let mask = BlockSparseMask::local_window(16, 4, 1);
        assert_eq!(mask.num_blocks_q, 4);
        // Block 0: window covers [0, 1]
        assert!(mask.is_active(0, 0));
        assert!(mask.is_active(0, 1));
        assert!(!mask.is_active(0, 2));
        // Block 2: window covers [1, 2, 3]
        assert!(!mask.is_active(2, 0));
        assert!(mask.is_active(2, 1));
        assert!(mask.is_active(2, 2));
        assert!(mask.is_active(2, 3));
    }

    #[test]
    fn hybrid_mask_sparsity() {
        // Hybrid should be at least as dense as either component alone
        let local = BlockSparseMask::local_window(32, 4, 1);
        let strided = BlockSparseMask::strided(32, 4, 2);
        let hybrid = BlockSparseMask::hybrid(32, 4, 1, 2);

        assert!(hybrid.active_count() >= local.active_count());
        assert!(hybrid.active_count() >= strided.active_count());
        // Hybrid sparsity should be between the two components
        assert!(hybrid.sparsity() > 0.0);
        assert!(hybrid.sparsity() <= 1.0);
    }

    #[test]
    fn attention_matches_dense_for_causal() {
        // When block_size = 1, causal block-sparse should match dense causal attention
        let seq_len = 4;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Simple deterministic data
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mask = BlockSparseMask::causal(seq_len, 1);
        let sparse_out = block_sparse_attention(&q, &k, &v, &mask, head_dim, scale);
        let dense_out = dense_attention(&q, &k, &v, seq_len, head_dim, scale, true);

        for i in 0..sparse_out.len() {
            assert!(
                (sparse_out[i] - dense_out[i]).abs() < 1e-5,
                "mismatch at {}: sparse={} dense={}",
                i,
                sparse_out[i],
                dense_out[i]
            );
        }
    }

    #[test]
    fn sparsity_calculation() {
        // Full causal mask on 4 blocks: 1+2+3+4 = 10 active out of 16
        let mask = BlockSparseMask::causal(16, 4);
        assert_eq!(mask.active_count(), 10);
        assert!((mask.sparsity() - 10.0 / 16.0).abs() < 1e-10);

        // Local window with window=0 is diagonal: 4 active out of 16
        let diag = BlockSparseMask::local_window(16, 4, 0);
        assert_eq!(diag.active_count(), 4);
        assert!((diag.sparsity() - 0.25).abs() < 1e-10);
    }
}
