#![allow(clippy::needless_range_loop)]

use yule_core::dtype::DType;
use yule_core::error::Result;
use yule_gpu::{BufferHandle, ComputeBackend};

pub struct AttentionLayer {
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub layer_idx: u32,
    // Weight buffers (already uploaded to backend)
    pub w_q: BufferHandle,
    pub w_q_dtype: DType,
    pub w_k: BufferHandle,
    pub w_k_dtype: DType,
    pub w_v: BufferHandle,
    pub w_v_dtype: DType,
    pub w_o: BufferHandle,
    pub w_o_dtype: DType,
    // Config
    pub dim: u32,
    pub freq_base: f32,
    pub sliding_window: Option<u32>,
    pub attn_logit_softcap: Option<f32>,
    // Scratch buffers
    pub q_buf: BufferHandle,
    pub k_buf: BufferHandle,
    pub v_buf: BufferHandle,
    pub attn_out: BufferHandle,
    pub scores_buf: BufferHandle,
}

impl AttentionLayer {
    pub fn forward(
        &self,
        backend: &dyn ComputeBackend,
        hidden: &BufferHandle,
        kv_cache_k: &BufferHandle,
        kv_cache_v: &BufferHandle,
        pos: u32,
        _seq_len: u32,
    ) -> Result<BufferHandle> {
        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let kv_group_size = nh / nkv;
        let kv_dim = nkv * hd;
        let kv_stride = kv_dim;

        // 1. Q, K, V projections
        backend.matmul(hidden, &self.w_q, &self.q_buf, 1, nh * hd, self.dim)?;
        backend.matmul(hidden, &self.w_k, &self.k_buf, 1, kv_dim, self.dim)?;
        backend.matmul(hidden, &self.w_v, &self.v_buf, 1, kv_dim, self.dim)?;

        // 2. Apply RoPE to Q and K
        backend.rope(&self.q_buf, &self.k_buf, pos, hd, self.freq_base, nh, nkv)?;

        // 3. Update KV cache: write K and V at the current position
        let cache_byte_offset = pos as usize * kv_stride as usize * 4;
        backend.copy_buffer_offset(
            &self.k_buf,
            kv_cache_k,
            0,
            cache_byte_offset,
            kv_dim as usize * 4,
        )?;
        backend.copy_buffer_offset(
            &self.v_buf,
            kv_cache_v,
            0,
            cache_byte_offset,
            kv_dim as usize * 4,
        )?;

        // 4-6. Attention: score, mask, softmax, weighted sum per head
        let cur_seq_len = pos + 1;

        // Determine effective sequence range for sliding window
        let start_pos = if let Some(w) = self.sliding_window {
            cur_seq_len.saturating_sub(w)
        } else {
            0
        };
        let _effective_len = cur_seq_len - start_pos;

        for h in 0..nh {
            let kv_h = h / kv_group_size;
            let head_offset = h * hd;
            let kv_offset = kv_h * hd;

            // Compute attention scores
            // If using sliding window with start_pos > 0, we need to adjust
            // the K cache pointer. For simplicity, compute over the full range
            // and mask out positions outside the window.
            backend.attn_score(
                &self.q_buf,
                kv_cache_k,
                &self.scores_buf,
                hd,
                cur_seq_len,
                head_offset,
                kv_offset,
                kv_stride,
            )?;

            // Apply sliding window mask if needed: set scores outside window to -inf
            if start_pos > 0 {
                let mut scores_bytes = vec![0u8; cur_seq_len as usize * 4];
                backend.copy_from_device(&self.scores_buf, &mut scores_bytes)?;
                let scores_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut scores_bytes);
                for t in 0..start_pos as usize {
                    scores_f32[t] = f32::NEG_INFINITY;
                }
                backend.copy_to_device(&scores_bytes, &self.scores_buf)?;
            }

            // Apply attention logit softcap if configured
            if let Some(cap) = self.attn_logit_softcap {
                let mut scores_bytes = vec![0u8; cur_seq_len as usize * 4];
                backend.copy_from_device(&self.scores_buf, &mut scores_bytes)?;
                let scores_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut scores_bytes);
                for t in 0..cur_seq_len as usize {
                    scores_f32[t] = cap * (scores_f32[t] / cap).tanh();
                }
                backend.copy_to_device(&scores_bytes, &self.scores_buf)?;
            }

            // 5. Softmax
            backend.softmax(&self.scores_buf, &self.scores_buf, cur_seq_len)?;

            // 6. Weighted value sum
            backend.attn_value(
                &self.scores_buf,
                kv_cache_v,
                &self.attn_out,
                hd,
                cur_seq_len,
                kv_offset,
                kv_stride,
                head_offset,
            )?;
        }

        // 7. Output projection
        let out = backend.allocate(self.dim as usize * 4)?;
        backend.matmul(&self.attn_out, &self.w_o, &out, 1, self.dim, nh * hd)?;
        Ok(out)
    }
}

pub struct FlashAttention {
    pub block_size_q: u32,
    pub block_size_kv: u32,
}

pub struct FlashAttentionParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub seq_len_q: u32,
    pub seq_len_kv: u32,
    pub head_dim: u32,
    pub causal: bool,
}

impl FlashAttention {
    pub fn new(head_dim: u32) -> Self {
        let block_size = if head_dim <= 64 { 128 } else { 64 };
        Self {
            block_size_q: block_size,
            block_size_kv: block_size,
        }
    }

    pub fn forward(
        &self,
        backend: &dyn ComputeBackend,
        q: &BufferHandle,
        k: &BufferHandle,
        v: &BufferHandle,
        params: &FlashAttentionParams,
    ) -> Result<BufferHandle> {
        let nh = params.num_heads as usize;
        let nkv = params.num_kv_heads as usize;
        let sq = params.seq_len_q as usize;
        let skv = params.seq_len_kv as usize;
        let hd = params.head_dim as usize;
        let kv_group_size = nh / nkv;

        // Read Q, K, V from backend buffers into CPU f32 vecs
        let q_bytes_len = nh * sq * hd * 4;
        let mut q_bytes = vec![0u8; q_bytes_len];
        backend.copy_from_device(q, &mut q_bytes)?;
        let q_f32: &[f32] = bytemuck::cast_slice(&q_bytes);

        let kv_bytes_len = nkv * skv * hd * 4;
        let mut k_bytes = vec![0u8; kv_bytes_len];
        backend.copy_from_device(k, &mut k_bytes)?;
        let k_f32: &[f32] = bytemuck::cast_slice(&k_bytes);

        let mut v_bytes = vec![0u8; kv_bytes_len];
        backend.copy_from_device(v, &mut v_bytes)?;
        let v_f32: &[f32] = bytemuck::cast_slice(&v_bytes);

        // Output: [num_heads, seq_len_q, head_dim]
        let mut out_f32 = vec![0.0f32; nh * sq * hd];

        for h in 0..nh {
            let kv_h = h / kv_group_size;

            // Slice Q for this head: [sq, hd]
            let q_head = &q_f32[h * sq * hd..(h + 1) * sq * hd];
            // Slice K, V for the corresponding kv head: [skv, hd]
            let k_head = &k_f32[kv_h * skv * hd..(kv_h + 1) * skv * hd];
            let v_head = &v_f32[kv_h * skv * hd..(kv_h + 1) * skv * hd];

            let o_head = &mut out_f32[h * sq * hd..(h + 1) * sq * hd];

            flash_attention_single_head(
                q_head,
                k_head,
                v_head,
                o_head,
                sq,
                skv,
                hd,
                self.block_size_q as usize,
                self.block_size_kv as usize,
                params.causal,
            );
        }

        // Write output back to a backend buffer
        let out_handle = backend.allocate(out_f32.len() * 4)?;
        let out_bytes: &[u8] = bytemuck::cast_slice(&out_f32);
        backend.copy_to_device(out_bytes, &out_handle)?;
        Ok(out_handle)
    }
}

/// FlashAttention-2 single head with tiled online softmax.
/// Q: [sq, hd], K: [skv, hd], V: [skv, hd], O: [sq, hd]
fn flash_attention_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    sq: usize,
    skv: usize,
    hd: usize,
    block_q: usize,
    block_kv: usize,
    causal: bool,
) {
    let scale = 1.0 / (hd as f32).sqrt();

    // Outer loop over Q blocks
    for qi_start in (0..sq).step_by(block_q) {
        let qi_end = (qi_start + block_q).min(sq);
        let bq = qi_end - qi_start;

        // Per-row running state: max (m), sum (l), output (O)
        let mut m = vec![f32::NEG_INFINITY; bq];
        let mut l = vec![0.0f32; bq];
        let mut o_acc = vec![0.0f32; bq * hd];

        // Inner loop over KV blocks
        for kvi_start in (0..skv).step_by(block_kv) {
            let kvi_end = (kvi_start + block_kv).min(skv);
            let bkv = kvi_end - kvi_start;

            // Compute S = Q_block @ K_block^T * scale  [bq x bkv]
            let mut s = vec![0.0f32; bq * bkv];
            for i in 0..bq {
                for j in 0..bkv {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[(qi_start + i) * hd + d] * k[(kvi_start + j) * hd + d];
                    }
                    s[i * bkv + j] = dot * scale;
                }
            }

            // Apply causal mask: set S[i,j] = -inf where kv_pos > q_pos
            if causal {
                for i in 0..bq {
                    let q_pos = qi_start + i;
                    for j in 0..bkv {
                        let kv_pos = kvi_start + j;
                        if kv_pos > q_pos {
                            s[i * bkv + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Online softmax update per row
            for i in 0..bq {
                // Find new max for this row
                let mut new_m = m[i];
                for j in 0..bkv {
                    let val = s[i * bkv + j];
                    if val > new_m {
                        new_m = val;
                    }
                }

                // Rescale existing accumulator
                let rescale = if m[i] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m[i] - new_m).exp()
                };
                for d in 0..hd {
                    o_acc[i * hd + d] *= rescale;
                }
                l[i] *= rescale;

                // Accumulate new block
                for j in 0..bkv {
                    let p = (s[i * bkv + j] - new_m).exp();
                    l[i] += p;
                    for d in 0..hd {
                        o_acc[i * hd + d] += p * v[(kvi_start + j) * hd + d];
                    }
                }

                m[i] = new_m;
            }
        }

        // Final: O /= l
        for i in 0..bq {
            let inv_l = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
            for d in 0..hd {
                o[(qi_start + i) * hd + d] = o_acc[i * hd + d] * inv_l;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use yule_gpu::cpu::CpuBackend;

    fn write_f32(backend: &CpuBackend, handle: &BufferHandle, data: &[f32]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        backend.copy_to_device(bytes, handle).unwrap();
    }

    fn read_f32(backend: &CpuBackend, handle: &BufferHandle, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        backend.copy_from_device(handle, &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Naive attention for reference: softmax(Q @ K^T / sqrt(hd)) @ V
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sq: usize,
        skv: usize,
        hd: usize,
        causal: bool,
    ) -> Vec<f32> {
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; sq * hd];

        for i in 0..sq {
            // Compute scores
            let mut scores = vec![0.0f32; skv];
            for j in 0..skv {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[i * hd + d] * k[j * hd + d];
                }
                scores[j] = dot * scale;
            }

            // Apply causal mask
            if causal {
                for j in 0..skv {
                    if j > i {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut exp_scores = vec![0.0f32; skv];
            for j in 0..skv {
                exp_scores[j] = (scores[j] - max_s).exp();
                sum += exp_scores[j];
            }
            for j in 0..skv {
                exp_scores[j] /= sum;
            }

            // Weighted sum
            for d in 0..hd {
                let mut val = 0.0f32;
                for j in 0..skv {
                    val += exp_scores[j] * v[j * hd + d];
                }
                out[i * hd + d] = val;
            }
        }
        out
    }

    #[test]
    fn test_flash_attention_matches_naive() {
        let b = CpuBackend::new();
        let hd = 4;
        let sq = 3;
        let skv = 3;
        let nh = 1;
        let nkv = 1;

        // Q, K, V: [1 head, seq, hd]
        let q_data: Vec<f32> = (0..nh * sq * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let k_data: Vec<f32> = (0..nkv * skv * hd).map(|i| i as f32 * 0.15 - 0.3).collect();
        let v_data: Vec<f32> = (0..nkv * skv * hd)
            .map(|i| (i as f32 * 0.2) + 1.0)
            .collect();

        let q_handle = b.allocate(q_data.len() * 4).unwrap();
        let k_handle = b.allocate(k_data.len() * 4).unwrap();
        let v_handle = b.allocate(v_data.len() * 4).unwrap();

        write_f32(&b, &q_handle, &q_data);
        write_f32(&b, &k_handle, &k_data);
        write_f32(&b, &v_handle, &v_data);

        let flash = FlashAttention::new(hd as u32);
        let params = FlashAttentionParams {
            num_heads: nh as u32,
            num_kv_heads: nkv as u32,
            seq_len_q: sq as u32,
            seq_len_kv: skv as u32,
            head_dim: hd as u32,
            causal: false,
        };

        let out_handle = flash
            .forward(&b, &q_handle, &k_handle, &v_handle, &params)
            .unwrap();
        let result = read_f32(&b, &out_handle, nh * sq * hd);

        let expected = naive_attention(&q_data, &k_data, &v_data, sq, skv, hd, false);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "mismatch at index {}: flash={} naive={}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_causal() {
        let b = CpuBackend::new();
        let hd = 4;
        let sq = 4;
        let skv = 4;
        let nh = 1;
        let nkv = 1;

        let q_data: Vec<f32> = (0..nh * sq * hd).map(|i| i as f32 * 0.1 - 0.2).collect();
        let k_data: Vec<f32> = (0..nkv * skv * hd).map(|i| i as f32 * 0.12 + 0.1).collect();
        let v_data: Vec<f32> = (0..nkv * skv * hd).map(|i| i as f32 + 1.0).collect();

        let q_handle = b.allocate(q_data.len() * 4).unwrap();
        let k_handle = b.allocate(k_data.len() * 4).unwrap();
        let v_handle = b.allocate(v_data.len() * 4).unwrap();

        write_f32(&b, &q_handle, &q_data);
        write_f32(&b, &k_handle, &k_data);
        write_f32(&b, &v_handle, &v_data);

        let flash = FlashAttention::new(hd as u32);
        let params = FlashAttentionParams {
            num_heads: nh as u32,
            num_kv_heads: nkv as u32,
            seq_len_q: sq as u32,
            seq_len_kv: skv as u32,
            head_dim: hd as u32,
            causal: true,
        };

        let out_handle = flash
            .forward(&b, &q_handle, &k_handle, &v_handle, &params)
            .unwrap();
        let result = read_f32(&b, &out_handle, nh * sq * hd);

        let expected = naive_attention(&q_data, &k_data, &v_data, sq, skv, hd, true);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "causal mismatch at index {}: flash={} naive={}",
                i,
                result[i],
                expected[i]
            );
        }

        // Verify that row 0 only attends to position 0 (causal):
        // Row 0 output should equal V[0] exactly
        for d in 0..hd {
            assert!(
                (result[d] - v_data[d]).abs() < 1e-4,
                "row 0 should equal V[0] at dim {}: got {} expected {}",
                d,
                result[d],
                v_data[d]
            );
        }
    }

    #[test]
    fn test_flash_attention_gqa() {
        let b = CpuBackend::new();
        let hd = 4;
        let sq = 2;
        let skv = 2;
        let nh = 4; // 4 query heads
        let nkv = 2; // 2 kv heads, group size = 2

        // Q: [4 heads, 2 seq, 4 dim]
        let q_data: Vec<f32> = (0..nh * sq * hd).map(|i| i as f32 * 0.1).collect();
        // K, V: [2 heads, 2 seq, 4 dim]
        let k_data: Vec<f32> = (0..nkv * skv * hd).map(|i| i as f32 * 0.15).collect();
        let v_data: Vec<f32> = (0..nkv * skv * hd).map(|i| i as f32 + 1.0).collect();

        let q_handle = b.allocate(q_data.len() * 4).unwrap();
        let k_handle = b.allocate(k_data.len() * 4).unwrap();
        let v_handle = b.allocate(v_data.len() * 4).unwrap();

        write_f32(&b, &q_handle, &q_data);
        write_f32(&b, &k_handle, &k_data);
        write_f32(&b, &v_handle, &v_data);

        let flash = FlashAttention::new(hd as u32);
        let params = FlashAttentionParams {
            num_heads: nh as u32,
            num_kv_heads: nkv as u32,
            seq_len_q: sq as u32,
            seq_len_kv: skv as u32,
            head_dim: hd as u32,
            causal: false,
        };

        let out_handle = flash
            .forward(&b, &q_handle, &k_handle, &v_handle, &params)
            .unwrap();
        let result = read_f32(&b, &out_handle, nh * sq * hd);

        // Heads 0 and 1 share KV head 0, heads 2 and 3 share KV head 1
        // Verify by computing naive attention for each head with mapped KV
        for h in 0..nh {
            let kv_h = h / 2;
            let q_head = &q_data[h * sq * hd..(h + 1) * sq * hd];
            let k_head = &k_data[kv_h * skv * hd..(kv_h + 1) * skv * hd];
            let v_head = &v_data[kv_h * skv * hd..(kv_h + 1) * skv * hd];
            let expected = naive_attention(q_head, k_head, v_head, sq, skv, hd, false);
            let actual = &result[h * sq * hd..(h + 1) * sq * hd];

            for i in 0..expected.len() {
                assert!(
                    (actual[i] - expected[i]).abs() < 1e-4,
                    "GQA mismatch at head {} index {}: flash={} naive={}",
                    h,
                    i,
                    actual[i],
                    expected[i]
                );
            }
        }
    }
}
