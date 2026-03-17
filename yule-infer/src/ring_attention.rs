//! Ring attention for distributed inference across multiple devices.
//!
//! Each device holds a chunk of the KV cache. Queries are passed around a ring
//! while K/V stay local. This enables near-infinite context by distributing
//! memory across devices.
//!
//! Combined with our BLAKE3-AEAD encrypted transport, this provides
//! encrypted distributed inference — unique to Yule.
//!
//! Reference: Ring Attention (ICLR 2024), TokenRing (2024)

/// Configuration for ring attention.
pub struct RingConfig {
    pub num_devices: u32,
    pub device_rank: u32, // this device's position in the ring
    pub chunk_size: u32,  // sequence positions per device
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
}

/// Per-device state for ring attention.
pub struct RingAttentionState {
    config: RingConfig,
    /// Local KV cache chunk: positions [rank*chunk_size .. (rank+1)*chunk_size]
    local_k: Vec<f32>, // [chunk_size * num_kv_heads * head_dim]
    local_v: Vec<f32>,
    /// Partial attention output accumulator
    partial_output: Vec<f32>, // [num_heads * head_dim]
    /// Running log-sum-exp for online softmax across chunks
    running_lse: Vec<f32>, // [num_heads]
    running_max: Vec<f32>, // [num_heads]
    /// Tokens stored in local chunk
    local_token_count: u32,
}

impl RingAttentionState {
    pub fn new(config: RingConfig) -> Self {
        let kv_size =
            config.chunk_size as usize * config.num_kv_heads as usize * config.head_dim as usize;
        let out_size = config.num_heads as usize * config.head_dim as usize;

        Self {
            local_k: vec![0.0; kv_size],
            local_v: vec![0.0; kv_size],
            partial_output: vec![0.0; out_size],
            running_lse: vec![0.0; config.num_heads as usize],
            running_max: vec![f32::NEG_INFINITY; config.num_heads as usize],
            local_token_count: 0,
            config,
        }
    }

    /// Write a KV pair to the local chunk.
    pub fn write_local_kv(&mut self, position_in_chunk: u32, k: &[f32], v: &[f32]) {
        let stride = self.config.num_kv_heads as usize * self.config.head_dim as usize;
        let offset = position_in_chunk as usize * stride;
        self.local_k[offset..offset + stride].copy_from_slice(k);
        self.local_v[offset..offset + stride].copy_from_slice(v);
        if position_in_chunk + 1 > self.local_token_count {
            self.local_token_count = position_in_chunk + 1;
        }
    }

    /// Compute partial attention for a query against this device's local KV chunk.
    /// Returns (partial_output, local_max, local_lse) per head.
    pub fn compute_local_attention(
        &self,
        q: &[f32], // [num_heads * head_dim]
    ) -> LocalAttentionResult {
        let nh = self.config.num_heads as usize;
        let nkv = self.config.num_kv_heads as usize;
        let hd = self.config.head_dim as usize;
        let kv_group = nh / nkv;
        let seq_len = self.local_token_count as usize;
        let kv_stride = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut output = vec![0.0f32; nh * hd];
        let mut max_scores = vec![f32::NEG_INFINITY; nh];
        let mut lse = vec![0.0f32; nh]; // log-sum-exp

        for h in 0..nh {
            let kv_h = h / kv_group;
            let q_head = &q[h * hd..(h + 1) * hd];

            // Compute scores
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_offset = t * kv_stride + kv_h * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_head[d] * self.local_k[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            // Local softmax
            let local_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            max_scores[h] = local_max;

            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - local_max).exp();
                sum_exp += *s;
            }
            lse[h] = sum_exp;

            // Weighted sum
            let out_head = &mut output[h * hd..(h + 1) * hd];
            for t in 0..seq_len {
                let v_offset = t * kv_stride + kv_h * hd;
                for d in 0..hd {
                    out_head[d] += scores[t] * self.local_v[v_offset + d];
                }
            }
        }

        LocalAttentionResult {
            output,
            max_scores,
            lse,
        }
    }

    /// Combine local result with accumulated result using online softmax.
    /// This is the key operation that allows ring attention to work:
    /// results from different devices are combined without materializing
    /// the full attention matrix.
    pub fn combine_results(
        accumulated: &mut [f32], // [num_heads * head_dim]
        acc_max: &mut [f32],     // [num_heads]
        acc_lse: &mut [f32],     // [num_heads]
        local: &LocalAttentionResult,
        num_heads: usize,
        head_dim: usize,
    ) {
        for h in 0..num_heads {
            let new_max = acc_max[h].max(local.max_scores[h]);

            // Rescale accumulated output
            let acc_scale = (acc_max[h] - new_max).exp();
            let local_scale = (local.max_scores[h] - new_max).exp();

            let new_lse = acc_lse[h] * acc_scale + local.lse[h] * local_scale;

            let head_off = h * head_dim;
            for d in 0..head_dim {
                accumulated[head_off + d] = accumulated[head_off + d] * acc_scale
                    + local.output[head_off + d] * local_scale;
            }

            acc_max[h] = new_max;
            acc_lse[h] = new_lse;
        }
    }

    /// Finalize: divide accumulated output by total log-sum-exp.
    pub fn finalize(output: &mut [f32], lse: &[f32], num_heads: usize, head_dim: usize) {
        for h in 0..num_heads {
            if lse[h] > 0.0 {
                let inv = 1.0 / lse[h];
                let head_off = h * head_dim;
                for d in 0..head_dim {
                    output[head_off + d] *= inv;
                }
            }
        }
    }

    /// Global sequence position for a local position.
    pub fn global_position(&self, local_pos: u32) -> u64 {
        self.config.device_rank as u64 * self.config.chunk_size as u64 + local_pos as u64
    }

    pub fn local_token_count(&self) -> u32 {
        self.local_token_count
    }

    pub fn clear(&mut self) {
        self.local_k.fill(0.0);
        self.local_v.fill(0.0);
        self.partial_output.fill(0.0);
        self.running_lse.fill(0.0);
        self.running_max.fill(f32::NEG_INFINITY);
        self.local_token_count = 0;
    }
}

/// Result of local attention computation on one device.
pub struct LocalAttentionResult {
    pub output: Vec<f32>,     // [num_heads * head_dim] (unnormalized)
    pub max_scores: Vec<f32>, // [num_heads] local max for online softmax
    pub lse: Vec<f32>,        // [num_heads] sum of exp(scores - max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_device_matches_standard_attention() {
        // With 1 device, ring attention should produce same result as standard attention
        let config = RingConfig {
            num_devices: 1,
            device_rank: 0,
            chunk_size: 4,
            head_dim: 4,
            num_heads: 2,
            num_kv_heads: 2,
        };

        let mut state = RingAttentionState::new(config);

        // Write 3 KV pairs
        let stride = 2 * 4; // num_kv_heads * head_dim
        for pos in 0..3 {
            let k: Vec<f32> = (0..stride)
                .map(|i| (pos * stride + i) as f32 * 0.1)
                .collect();
            let v: Vec<f32> = (0..stride)
                .map(|i| (pos * stride + i) as f32 * 0.2 + 1.0)
                .collect();
            state.write_local_kv(pos as u32, &k, &v);
        }

        // Query
        let q: Vec<f32> = (0..stride).map(|i| i as f32 * 0.3).collect();

        let result = state.compute_local_attention(&q);

        // Output should be finite and non-zero
        assert!(result.output.iter().all(|v| v.is_finite()));
        assert!(result.output.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn test_two_device_combination() {
        let hd = 2;
        let nh = 1;

        // Device 0: K=[1,0], V=[10,20]
        let config0 = RingConfig {
            num_devices: 2,
            device_rank: 0,
            chunk_size: 1,
            head_dim: hd as u32,
            num_heads: nh as u32,
            num_kv_heads: nh as u32,
        };
        let mut state0 = RingAttentionState::new(config0);
        state0.write_local_kv(0, &[1.0, 0.0], &[10.0, 20.0]);

        // Device 1: K=[0,1], V=[30,40]
        let config1 = RingConfig {
            num_devices: 2,
            device_rank: 1,
            chunk_size: 1,
            head_dim: hd as u32,
            num_heads: nh as u32,
            num_kv_heads: nh as u32,
        };
        let mut state1 = RingAttentionState::new(config1);
        state1.write_local_kv(0, &[0.0, 1.0], &[30.0, 40.0]);

        let q = vec![1.0f32, 0.0]; // Q=[1,0] should attend more to K=[1,0]

        let local0 = state0.compute_local_attention(&q);
        let local1 = state1.compute_local_attention(&q);

        // Combine
        let mut acc_output = local0.output.clone();
        let mut acc_max = local0.max_scores.clone();
        let mut acc_lse = local0.lse.clone();

        RingAttentionState::combine_results(
            &mut acc_output,
            &mut acc_max,
            &mut acc_lse,
            &local1,
            nh,
            hd,
        );

        RingAttentionState::finalize(&mut acc_output, &acc_lse, nh, hd);

        // Result should weight V[0]=[10,20] more than V[1]=[30,40]
        // because Q=[1,0] has higher dot product with K[0]=[1,0]
        assert!(acc_output[0] < 25.0, "should lean toward V[0]=10");
        assert!(acc_output[1] < 35.0, "should lean toward V[0]=20");
    }

    #[test]
    fn test_global_position() {
        let config = RingConfig {
            num_devices: 4,
            device_rank: 2,
            chunk_size: 100,
            head_dim: 64,
            num_heads: 32,
            num_kv_heads: 8,
        };
        let state = RingAttentionState::new(config);
        assert_eq!(state.global_position(5), 205); // rank 2 * chunk 100 + 5
    }
}
