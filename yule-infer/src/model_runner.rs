#![allow(clippy::needless_range_loop)]
use crate::weight_loader::TransformerWeights;
use yule_core::dequant;
use yule_core::dtype::DType;
use yule_core::error::{Result, YuleError};
use yule_core::model::Architecture;
use yule_core::tensor::TensorInfo;

pub trait ModelRunner: Send {
    fn architecture(&self) -> Architecture;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn decode_step(&mut self, token: u32) -> Result<Vec<f32>>;
    fn reset(&mut self);
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Activation {
    SwiGLU,
    GeGLU,
}

#[derive(Debug, Clone)]
struct RunnerConfig {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    ff_dim: usize,
    norm_eps: f32,
    rope_freq_base: f32,
    max_seq_len: usize,
    sliding_window: Option<usize>,
    #[allow(dead_code)]
    partial_rotary_dim: Option<usize>,
    has_qkv_bias: bool,
    activation: Activation,
    has_post_attn_norm: bool,
    has_post_ffn_norm: bool,
    logit_softcap: Option<f32>,
    attn_logit_softcap: Option<f32>,
    norm_weight_offset: f32,
}

struct ScratchBuffers {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_proj: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_out: Vec<f32>,
    logits: Vec<f32>,
    post_norm_tmp: Vec<f32>,
}

struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half_dim: usize,
}

impl RopeTable {
    fn new(max_seq_len: usize, rotary_dim: usize, freq_base: f32) -> Self {
        let half = rotary_dim / 2;
        let len = max_seq_len * half;
        let mut cos = vec![0.0f32; len];
        let mut sin = vec![0.0f32; len];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / freq_base.powf(2.0 * i as f32 / rotary_dim as f32);
                let theta = pos as f32 * freq;
                let (s, c) = theta.sin_cos();
                cos[pos * half + i] = c;
                sin[pos * half + i] = s;
            }
        }

        Self {
            cos,
            sin,
            half_dim: half,
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn prefetch_next_block(
    data: &[u8],
    _cur: usize,
    row_off: usize,
    row: usize,
    b: usize,
    bpr: usize,
    bb: usize,
    n_rows: usize,
) {
    let next = if b + 1 < bpr {
        row_off + (b + 1) * bb
    } else if row + 1 < n_rows {
        (row + 1) * bpr * bb
    } else {
        return; // last block overall, nothing to prefetch
    };
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    unsafe {
        let ptr = data.as_ptr().add(next) as *const i8;
        _mm_prefetch(ptr, _MM_HINT_T0);
        // second cache line for blocks >64B (Q4_K=144B, Q6_K=210B)
        if bb > 64 {
            _mm_prefetch(ptr.add(64), _MM_HINT_T0);
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn prefetch_next_block(
    _data: &[u8],
    _cur: usize,
    _row_off: usize,
    _row: usize,
    _b: usize,
    _bpr: usize,
    _bb: usize,
    _n_rows: usize,
) {
}

fn qmv(weight_info: &TensorInfo, weight_data: &[u8], input: &[f32], out: &mut [f32]) -> Result<()> {
    let dtype = weight_info.dtype;
    let block_size = dtype.block_size();
    let block_bytes = dtype.size_of_block();
    let n_rows = out.len();
    let n_cols = input.len();
    let blocks_per_row = n_cols / block_size;

    for row in 0..n_rows {
        let mut sum = 0.0f32;
        let row_offset = row * blocks_per_row * block_bytes;
        for b in 0..blocks_per_row {
            let block_start = row_offset + b * block_bytes;
            let block = &weight_data[block_start..block_start + block_bytes];
            let act_start = b * block_size;
            let act = &input[act_start..act_start + block_size];

            // prefetch next weight block into L1 while computing current dot
            prefetch_next_block(
                weight_data,
                block_start,
                row_offset,
                row,
                b,
                blocks_per_row,
                block_bytes,
                n_rows,
            );

            sum += dequant::vec_dot_block(dtype, block, act).unwrap_or_else(|_| {
                let mut tmp = vec![0.0f32; block_size];
                let _ = dequant::dequant_block(dtype, block, &mut tmp);
                tmp.iter().zip(act.iter()).map(|(w, a)| w * a).sum::<f32>()
            });
        }
        out[row] = sum;
    }
    Ok(())
}

fn rms_norm(
    x: &[f32],
    weight_data: &[u8],
    weight_info: &TensorInfo,
    eps: f32,
    offset: f32,
    out: &mut [f32],
) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();

    for i in 0..n {
        let w = if weight_info.dtype == DType::F32 {
            f32::from_le_bytes([
                weight_data[i * 4],
                weight_data[i * 4 + 1],
                weight_data[i * 4 + 2],
                weight_data[i * 4 + 3],
            ])
        } else {
            let bits = u16::from_le_bytes([weight_data[i * 2], weight_data[i * 2 + 1]]);
            dequant::f16_to_f32(bits)
        };
        out[i] = x[i] * inv * (w + offset);
    }
}

fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, rope: &RopeTable) {
    let half = rope.half_dim;
    let n_heads = vec.len() / head_dim;
    let base = pos * half;

    for h in 0..n_heads {
        let head_off = h * head_dim;
        for i in 0..half {
            let cos_t = rope.cos[base + i];
            let sin_t = rope.sin[base + i];
            let x0 = vec[head_off + 2 * i];
            let x1 = vec[head_off + 2 * i + 1];
            vec[head_off + 2 * i] = x0 * cos_t - x1 * sin_t;
            vec[head_off + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

fn add_bias(vec: &mut [f32], bias_info: &TensorInfo, bias_data: &[u8]) {
    for i in 0..vec.len() {
        let b = if bias_info.dtype == DType::F32 {
            f32::from_le_bytes([
                bias_data[i * 4],
                bias_data[i * 4 + 1],
                bias_data[i * 4 + 2],
                bias_data[i * 4 + 3],
            ])
        } else {
            let bits = u16::from_le_bytes([bias_data[i * 2], bias_data[i * 2 + 1]]);
            dequant::f16_to_f32(bits)
        };
        vec[i] += b;
    }
}

fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

pub struct TransformerRunner<'a> {
    cfg: RunnerConfig,
    arch: Architecture,
    weights: TransformerWeights<'a>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pos: usize,
    hidden: Vec<f32>,
    residual: Vec<f32>,
    scratch: ScratchBuffers,
    rope: RopeTable,
}

pub type LlamaRunner<'a> = TransformerRunner<'a>;

impl<'a> TransformerRunner<'a> {
    pub fn new(weights: TransformerWeights<'a>) -> Result<Self> {
        let meta = &weights.store.meta;
        let arch = meta.architecture.clone();

        let dim = meta.embedding_dim as usize;
        let n_heads = meta.head_count as usize;
        let n_kv_heads = meta.head_count_kv as usize;
        let n_layers = meta.layer_count as usize;
        let head_dim = dim / n_heads;
        let max_seq_len = meta.context_length as usize;

        let ff_dim = if let Ok((info, _)) = weights.ffn_gate(0) {
            info.shape[1] as usize
        } else {
            ((dim as f64 * 8.0 / 3.0 / 256.0).ceil() as usize) * 256
        };

        let default_eps = match arch {
            Architecture::Phi | Architecture::Qwen | Architecture::Gemma => 1e-6,
            _ => 1e-5,
        };

        let is_gemma = matches!(arch, Architecture::Gemma);
        let has_qkv_bias = weights.attn_q_bias(0).is_some();
        let has_post_attn_norm = weights.attn_post_norm(0).is_some();
        let has_post_ffn_norm = weights.ffn_post_norm(0).is_some();

        let activation = if is_gemma {
            Activation::GeGLU
        } else {
            Activation::SwiGLU
        };

        let rotary_dim = meta
            .partial_rotary_dim
            .map(|d| d as usize)
            .filter(|&d| d < head_dim)
            .unwrap_or(head_dim);

        let cfg = RunnerConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size: meta.vocab_size as usize,
            ff_dim,
            norm_eps: meta.norm_eps.unwrap_or(default_eps),
            rope_freq_base: meta.rope_freq_base.unwrap_or(10000.0) as f32,
            max_seq_len,
            sliding_window: meta.sliding_window.map(|w| w as usize),
            partial_rotary_dim: if rotary_dim < head_dim {
                Some(rotary_dim)
            } else {
                None
            },
            has_qkv_bias,
            activation,
            has_post_attn_norm,
            has_post_ffn_norm,
            logit_softcap: meta.logit_softcap,
            attn_logit_softcap: meta.attn_logit_softcap,
            norm_weight_offset: if is_gemma { 1.0 } else { 0.0 },
        };

        let kv_len = max_seq_len * n_kv_heads * head_dim;
        let k_cache = vec![vec![0.0f32; kv_len]; n_layers];
        let v_cache = vec![vec![0.0f32; kv_len]; n_layers];

        let scratch = ScratchBuffers {
            normed: vec![0.0; dim],
            q: vec![0.0; n_heads * head_dim],
            k: vec![0.0; n_kv_heads * head_dim],
            v: vec![0.0; n_kv_heads * head_dim],
            attn_out: vec![0.0; n_heads * head_dim],
            attn_proj: vec![0.0; dim],
            scores: vec![0.0; max_seq_len],
            gate: vec![0.0; ff_dim],
            up: vec![0.0; ff_dim],
            ffn_out: vec![0.0; dim],
            logits: vec![0.0; cfg.vocab_size],
            post_norm_tmp: vec![0.0; dim],
        };

        let rope = RopeTable::new(max_seq_len, rotary_dim, cfg.rope_freq_base);

        Ok(Self {
            cfg,
            arch,
            weights,
            k_cache,
            v_cache,
            pos: 0,
            hidden: vec![0.0; dim],
            residual: vec![0.0; dim],
            scratch,
            rope,
        })
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let dim = self.cfg.dim;
        let pos = self.pos;
        let eps = self.cfg.norm_eps;
        let norm_off = self.cfg.norm_weight_offset;

        if pos >= self.cfg.max_seq_len {
            return Err(YuleError::Inference("context length exceeded".into()));
        }

        // 1. token embedding
        let (embd_info, embd_data) = self.weights.token_embd()?;
        let embd_dtype = embd_info.dtype;
        let embd_row_bytes = embd_info.size_bytes as usize / self.cfg.vocab_size;
        let tok_offset = token as usize * embd_row_bytes;
        let tok_data = &embd_data[tok_offset..tok_offset + embd_row_bytes];

        if embd_dtype == DType::F32 {
            for i in 0..dim {
                self.hidden[i] = f32::from_le_bytes([
                    tok_data[i * 4],
                    tok_data[i * 4 + 1],
                    tok_data[i * 4 + 2],
                    tok_data[i * 4 + 3],
                ]);
            }
        } else {
            let bs = embd_dtype.block_size();
            let bb = embd_dtype.size_of_block();
            for b in 0..(dim / bs) {
                let block = &tok_data[b * bb..(b + 1) * bb];
                dequant::dequant_block(embd_dtype, block, &mut self.hidden[b * bs..(b + 1) * bs])?;
            }
        }

        // Gemma: scale embeddings by sqrt(dim)
        if self.cfg.norm_weight_offset > 0.0 {
            let scale = (dim as f32).sqrt();
            for h in &mut self.hidden {
                *h *= scale;
            }
        }

        // 2. transformer layers
        let n_heads = self.cfg.n_heads;
        let n_kv_heads = self.cfg.n_kv_heads;
        let hd = self.cfg.head_dim;
        let kv_group_size = n_heads / n_kv_heads;
        let kv_stride = n_kv_heads * hd;
        let seq_len = pos + 1;

        for layer in 0..self.cfg.n_layers {
            self.residual.copy_from_slice(&self.hidden);

            // attention norm
            let (norm_info, norm_data) = self.weights.attn_norm(layer as u32)?;
            rms_norm(
                &self.hidden,
                norm_data,
                norm_info,
                eps,
                norm_off,
                &mut self.scratch.normed,
            );

            // QKV projections
            let (qi, qd) = self.weights.attn_q(layer as u32)?;
            let (ki, kd) = self.weights.attn_k(layer as u32)?;
            let (vi, vd) = self.weights.attn_v(layer as u32)?;

            qmv(qi, qd, &self.scratch.normed, &mut self.scratch.q)?;
            qmv(ki, kd, &self.scratch.normed, &mut self.scratch.k)?;
            qmv(vi, vd, &self.scratch.normed, &mut self.scratch.v)?;

            // QKV bias (Qwen2)
            if self.cfg.has_qkv_bias {
                if let Some((bi, bd)) = self.weights.attn_q_bias(layer as u32) {
                    add_bias(&mut self.scratch.q, bi, bd);
                }
                if let Some((bi, bd)) = self.weights.attn_k_bias(layer as u32) {
                    add_bias(&mut self.scratch.k, bi, bd);
                }
                if let Some((bi, bd)) = self.weights.attn_v_bias(layer as u32) {
                    add_bias(&mut self.scratch.v, bi, bd);
                }
            }

            // RoPE
            apply_rope(&mut self.scratch.q, pos, hd, &self.rope);
            apply_rope(&mut self.scratch.k, pos, hd, &self.rope);

            // write KV to cache
            let cache_off = pos * kv_stride;
            self.k_cache[layer][cache_off..cache_off + kv_stride]
                .copy_from_slice(&self.scratch.k[..kv_stride]);
            self.v_cache[layer][cache_off..cache_off + kv_stride]
                .copy_from_slice(&self.scratch.v[..kv_stride]);

            // attention
            self.scratch.attn_out.fill(0.0);
            let scale = 1.0 / (hd as f32).sqrt();

            for h in 0..n_heads {
                let kv_h = h / kv_group_size;
                let q_head = &self.scratch.q[h * hd..(h + 1) * hd];

                let scores = &mut self.scratch.scores[..seq_len];
                for t in 0..seq_len {
                    // sliding window masking
                    if let Some(w) = self.cfg.sliding_window {
                        if pos >= w && t < pos - w {
                            scores[t] = f32::NEG_INFINITY;
                            continue;
                        }
                    }

                    let k_offset = t * kv_stride + kv_h * hd;
                    let k_head = &self.k_cache[layer][k_offset..k_offset + hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q_head[d] * k_head[d];
                    }
                    scores[t] = dot * scale;
                }

                // attention logit softcap (Gemma2)
                if let Some(cap) = self.cfg.attn_logit_softcap {
                    for s in scores.iter_mut() {
                        *s = cap * (*s / cap).tanh();
                    }
                }

                // softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }

                let out_head = &mut self.scratch.attn_out[h * hd..(h + 1) * hd];
                for t in 0..seq_len {
                    let v_offset = t * kv_stride + kv_h * hd;
                    let v_head = &self.v_cache[layer][v_offset..v_offset + hd];
                    let w = scores[t];
                    for d in 0..hd {
                        out_head[d] += w * v_head[d];
                    }
                }
            }

            // output projection
            let (oi, od) = self.weights.attn_output(layer as u32)?;
            qmv(oi, od, &self.scratch.attn_out, &mut self.scratch.attn_proj)?;

            // post-attention norm (Gemma2)
            if self.cfg.has_post_attn_norm {
                if let Some((ni, nd)) = self.weights.attn_post_norm(layer as u32) {
                    self.scratch
                        .post_norm_tmp
                        .copy_from_slice(&self.scratch.attn_proj);
                    rms_norm(
                        &self.scratch.post_norm_tmp,
                        nd,
                        ni,
                        eps,
                        norm_off,
                        &mut self.scratch.attn_proj,
                    );
                }
            }

            // residual
            for i in 0..dim {
                self.hidden[i] = self.residual[i] + self.scratch.attn_proj[i];
            }

            // FFN
            self.residual.copy_from_slice(&self.hidden);

            let (fn_info, fn_data) = self.weights.ffn_norm(layer as u32)?;
            rms_norm(
                &self.hidden,
                fn_data,
                fn_info,
                eps,
                norm_off,
                &mut self.scratch.normed,
            );

            let ff = self.cfg.ff_dim;
            let (gi, gd) = self.weights.ffn_gate(layer as u32)?;
            let (ui, ud) = self.weights.ffn_up(layer as u32)?;
            qmv(gi, gd, &self.scratch.normed, &mut self.scratch.gate[..ff])?;
            qmv(ui, ud, &self.scratch.normed, &mut self.scratch.up[..ff])?;

            match self.cfg.activation {
                Activation::SwiGLU => {
                    for i in 0..ff {
                        let sigmoid = 1.0 / (1.0 + (-self.scratch.gate[i]).exp());
                        self.scratch.gate[i] = self.scratch.gate[i] * sigmoid * self.scratch.up[i];
                    }
                }
                Activation::GeGLU => {
                    for i in 0..ff {
                        self.scratch.gate[i] = gelu(self.scratch.gate[i]) * self.scratch.up[i];
                    }
                }
            }

            let (di, dd) = self.weights.ffn_down(layer as u32)?;
            qmv(di, dd, &self.scratch.gate[..ff], &mut self.scratch.ffn_out)?;

            // post-FFN norm (Gemma2)
            if self.cfg.has_post_ffn_norm {
                if let Some((ni, nd)) = self.weights.ffn_post_norm(layer as u32) {
                    self.scratch
                        .post_norm_tmp
                        .copy_from_slice(&self.scratch.ffn_out);
                    rms_norm(
                        &self.scratch.post_norm_tmp,
                        nd,
                        ni,
                        eps,
                        norm_off,
                        &mut self.scratch.ffn_out,
                    );
                }
            }

            for i in 0..dim {
                self.hidden[i] = self.residual[i] + self.scratch.ffn_out[i];
            }
        }

        // 3. final norm
        let (on_info, on_data) = self.weights.output_norm()?;
        rms_norm(
            &self.hidden,
            on_data,
            on_info,
            eps,
            norm_off,
            &mut self.scratch.normed,
        );

        // 4. output → logits
        let (out_info, out_data) = self.weights.output()?;
        qmv(
            out_info,
            out_data,
            &self.scratch.normed,
            &mut self.scratch.logits,
        )?;

        // final logit softcap (Gemma2)
        if let Some(cap) = self.cfg.logit_softcap {
            for l in self.scratch.logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }

        self.pos += 1;
        Ok(self.scratch.logits.clone())
    }
}

impl<'a> ModelRunner for TransformerRunner<'a> {
    fn architecture(&self) -> Architecture {
        self.arch.clone()
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for &tok in tokens {
            logits = self.forward(tok)?;
        }
        Ok(logits)
    }

    fn decode_step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.forward(token)
    }

    fn reset(&mut self) {
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_values() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.8413).abs() < 1e-3);
        assert!((gelu(-1.0) - (-0.1587)).abs() < 1e-3);
    }

    #[test]
    fn test_sliding_window_masking() {
        let mut scores = [1.0f32; 10];
        let window = 4usize;
        let pos = 8;
        for t in 0..scores.len() {
            if pos >= window && t < pos - window {
                scores[t] = f32::NEG_INFINITY;
            }
        }
        // pos=8, window=4 → visible range [4..=8], positions 0-3 masked
        for t in 0..4 {
            assert!(scores[t] == f32::NEG_INFINITY);
        }
        for t in 4..10 {
            assert!(scores[t] == 1.0);
        }
    }

    #[test]
    fn test_logit_softcap() {
        let cap = 30.0f32;
        let mut logits = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
        for l in logits.iter_mut() {
            *l = cap * (*l / cap).tanh();
        }
        for &l in &logits {
            assert!(l.abs() <= cap + 1e-6);
        }
        assert!((logits[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_with_offset() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // F32 weight data: all 1.0
        let w_bytes: Vec<u8> = (0..4).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let info = TensorInfo {
            name: "test".into(),
            dtype: DType::F32,
            shape: vec![4],
            offset: 0,
            size_bytes: 16,
        };
        let mut out_normal = vec![0.0f32; 4];
        let mut out_offset = vec![0.0f32; 4];

        rms_norm(&x, &w_bytes, &info, 1e-5, 0.0, &mut out_normal);
        rms_norm(&x, &w_bytes, &info, 1e-5, 1.0, &mut out_offset);

        // with offset 1.0, weight becomes 2.0, so output should be 2x
        for i in 0..4 {
            assert!((out_offset[i] - out_normal[i] * 2.0).abs() < 1e-4);
        }
    }
}
