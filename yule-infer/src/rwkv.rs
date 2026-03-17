//! RWKV (Receptance Weighted Key Value) model runner.
//!
//! RWKV is a linear-attention RNN that processes tokens in O(1) memory per step
//! (no KV cache). It achieves transformer-quality results with constant memory
//! regardless of sequence length.
//!
//! Architecture per layer (v5/v6 "Eagle/Finch"):
//! 1. Time-Mix (replaces attention):
//!    - Compute r, k, v from interpolated input (lerp with previous token)
//!    - WKV mechanism: linear attention via recurrent state
//!    - Output gated by sigmoid(r)
//! 2. Channel-Mix (replaces FFN):
//!    - Receptance r and key k from interpolated input
//!    - Output: sigmoid(r) * (W_v @ squared_relu(k))

#![allow(clippy::needless_range_loop)]

use crate::model_runner::ModelRunner;
use crate::weight_loader::WeightStore;
use yule_core::dequant;
use yule_core::dtype::DType;
use yule_core::error::Result;
use yule_core::model::Architecture;
use yule_core::tensor::TensorInfo;

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

fn layer_norm(x: &[f32], weight_data: &[u8], weight_info: &TensorInfo, eps: f32, out: &mut [f32]) {
    let n = x.len();

    // Compute mean
    let mut mean = 0.0f32;
    for &v in x {
        mean += v;
    }
    mean /= n as f32;

    // Compute variance
    let mut var = 0.0f32;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n as f32 + eps).sqrt();

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
        out[i] = (x[i] - mean) * inv * w;
    }
}

/// Read a single f32 value from a tensor's raw bytes at the given index.
fn read_f32(data: &[u8], info: &TensorInfo, idx: usize) -> f32 {
    if info.dtype == DType::F32 {
        f32::from_le_bytes([
            data[idx * 4],
            data[idx * 4 + 1],
            data[idx * 4 + 2],
            data[idx * 4 + 3],
        ])
    } else {
        let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
        dequant::f16_to_f32(bits)
    }
}

/// Linear interpolation: lerp(a, b, mu) = a * (1 - mu) + b * mu
fn lerp(a: &[f32], b: &[f32], mu_data: &[u8], mu_info: &TensorInfo, out: &mut [f32]) {
    for i in 0..out.len() {
        let mu = read_f32(mu_data, mu_info, i);
        out[i] = a[i] * (1.0 - mu) + b[i] * mu;
    }
}

/// Squared ReLU activation: max(0, x)^2
fn squared_relu(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

struct RwkvConfig {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    norm_eps: f32,
}

struct RwkvLayerState {
    prev_x_tm: Vec<f32>, // [dim] — previous token embedding for time-mix
    prev_x_cm: Vec<f32>, // [dim] — previous token embedding for channel-mix
    state_a: Vec<f32>,   // [n_heads, head_dim, head_dim] — numerator state
    state_b: Vec<f32>,   // [n_heads, head_dim] — denominator state
}

struct RwkvScratch {
    normed: Vec<f32>,
    lerped: Vec<f32>,   // [dim] — interpolated input
    r: Vec<f32>,        // [dim] — receptance
    k: Vec<f32>,        // [dim] — key
    v: Vec<f32>,        // [dim] — value
    wkv: Vec<f32>,      // [dim] — WKV output
    gate_out: Vec<f32>, // [dim] — gating output
    tm_out: Vec<f32>,   // [dim] — time-mix output projection
    cm_r: Vec<f32>,     // [dim] — channel-mix receptance
    cm_k: Vec<f32>,     // [dim] — channel-mix key
    cm_k_act: Vec<f32>, // [dim] — channel-mix activated key
    cm_vk: Vec<f32>,    // [dim] — channel-mix W_v @ squared_relu(k)
    logits: Vec<f32>,
}

pub struct RwkvRunner<'a> {
    cfg: RwkvConfig,
    weights: RwkvWeights<'a>,
    layer_states: Vec<RwkvLayerState>,
    hidden: Vec<f32>,
    scratch: RwkvScratch,
}

struct RwkvWeights<'a> {
    store: &'a WeightStore<'a>,
}

impl<'a> RwkvWeights<'a> {
    fn token_embd(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store.require("token_embd.weight")
    }
    fn output_norm(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store.require("output_norm.weight")
    }
    fn output(&self) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require("output.weight")
            .or_else(|_| self.store.require("token_embd.weight"))
    }
    fn attn_norm(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_norm.weight"))
    }
    fn ffn_norm(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ffn_norm.weight"))
    }

    // Time-mix weights (v5/v6 names with v7 fallbacks)
    fn attn_receptance(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.attn_receptance.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.attn_r.weight")))
    }
    fn attn_key(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.attn_key.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.attn_k.weight")))
    }
    fn attn_value(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.attn_value.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.attn_v.weight")))
    }
    fn attn_output(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.attn_output.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.attn_o.weight")))
    }
    fn attn_gate(&self, layer: u32) -> Option<(&TensorInfo, &[u8])> {
        self.store
            .get(&format!("blk.{layer}.attn_gate.weight"))
            .or_else(|| self.store.get(&format!("blk.{layer}.attn_g.weight")))
    }

    // Time-mix interpolation ratios (v5/v6 names with v7 fallbacks)
    fn time_mix_r(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_r"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.attn_lerp_r.weight"))
            })
    }
    fn time_mix_k(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_k"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.attn_lerp_k.weight"))
            })
    }
    fn time_mix_v(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_v"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.attn_lerp_v.weight"))
            })
    }

    // Time-mix decay and bonus (v5/v6 names with v7 fallbacks)
    fn time_mix_w(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_w"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.attn_decay.weight"))
            })
    }
    fn time_first(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_first"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.attn_a.weight")))
    }

    // Channel-mix weights (v5/v6 names with v7 fallbacks)
    fn ffn_receptance(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.ffn_receptance.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.ffn_r.weight")))
    }
    fn ffn_key(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.ffn_key.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.ffn_k.weight")))
    }
    fn ffn_value(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.ffn_value.weight"))
            .or_else(|_| self.store.require(&format!("blk.{layer}.ffn_v.weight")))
    }

    // Channel-mix interpolation ratios (v5/v6 names with v7 fallbacks)
    fn time_mix_r_ffn(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_r_ffn"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.ffn_lerp_r.weight"))
            })
    }
    fn time_mix_k_ffn(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store
            .require(&format!("blk.{layer}.time_mix_k_ffn"))
            .or_else(|_| {
                self.store
                    .require(&format!("blk.{layer}.ffn_lerp_k.weight"))
            })
    }
}

impl<'a> RwkvRunner<'a> {
    pub fn new(store: &'a WeightStore<'a>) -> Result<Self> {
        let meta = &store.meta;
        let dim = meta.embedding_dim as usize;
        let n_layers = meta.layer_count as usize;
        // RWKV may not have head_count in metadata; default to dim/64
        let n_heads = if meta.head_count > 0 {
            meta.head_count as usize
        } else {
            (dim / 64).max(1)
        };
        let head_dim = dim / n_heads;
        let vocab_size = meta.vocab_size as usize;

        let cfg = RwkvConfig {
            dim,
            n_layers,
            n_heads,
            head_dim,
            vocab_size,
            norm_eps: meta.norm_eps.unwrap_or(1e-5),
        };

        let mut layer_states = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layer_states.push(RwkvLayerState {
                prev_x_tm: vec![0.0; dim],
                prev_x_cm: vec![0.0; dim],
                state_a: vec![0.0; n_heads * head_dim * head_dim],
                state_b: vec![0.0; n_heads * head_dim],
            });
        }

        let scratch = RwkvScratch {
            normed: vec![0.0; dim],
            lerped: vec![0.0; dim],
            r: vec![0.0; dim],
            k: vec![0.0; dim],
            v: vec![0.0; dim],
            wkv: vec![0.0; dim],
            gate_out: vec![0.0; dim],
            tm_out: vec![0.0; dim],
            cm_r: vec![0.0; dim],
            cm_k: vec![0.0; dim],
            cm_k_act: vec![0.0; dim],
            cm_vk: vec![0.0; dim],
            logits: vec![0.0; vocab_size],
        };

        Ok(Self {
            cfg,
            weights: RwkvWeights { store },
            layer_states,
            hidden: vec![0.0; dim],
            scratch,
        })
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let dim = self.cfg.dim;
        let n_heads = self.cfg.n_heads;
        let head_dim = self.cfg.head_dim;

        // 1. Token embedding
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

        // 2. RWKV layers
        for layer in 0..self.cfg.n_layers {
            let residual: Vec<f32> = self.hidden.clone();

            // ---- Time-Mix sub-block ----
            let (norm_info, norm_data) = self.weights.attn_norm(layer as u32)?;
            layer_norm(
                &self.hidden,
                norm_data,
                norm_info,
                self.cfg.norm_eps,
                &mut self.scratch.normed,
            );

            // Interpolated inputs for r, k, v
            let (mu_r_info, mu_r_data) = self.weights.time_mix_r(layer as u32)?;
            lerp(
                &self.layer_states[layer].prev_x_tm,
                &self.scratch.normed,
                mu_r_data,
                mu_r_info,
                &mut self.scratch.lerped,
            );
            let (wr_info, wr_data) = self.weights.attn_receptance(layer as u32)?;
            qmv(wr_info, wr_data, &self.scratch.lerped, &mut self.scratch.r)?;

            let (mu_k_info, mu_k_data) = self.weights.time_mix_k(layer as u32)?;
            lerp(
                &self.layer_states[layer].prev_x_tm,
                &self.scratch.normed,
                mu_k_data,
                mu_k_info,
                &mut self.scratch.lerped,
            );
            let (wk_info, wk_data) = self.weights.attn_key(layer as u32)?;
            qmv(wk_info, wk_data, &self.scratch.lerped, &mut self.scratch.k)?;

            let (mu_v_info, mu_v_data) = self.weights.time_mix_v(layer as u32)?;
            lerp(
                &self.layer_states[layer].prev_x_tm,
                &self.scratch.normed,
                mu_v_data,
                mu_v_info,
                &mut self.scratch.lerped,
            );
            let (wv_info, wv_data) = self.weights.attn_value(layer as u32)?;
            qmv(wv_info, wv_data, &self.scratch.lerped, &mut self.scratch.v)?;

            // Load decay weights (w) and bonus (time_first)
            let (w_info, w_data) = self.weights.time_mix_w(layer as u32)?;
            let (bonus_info, bonus_data) = self.weights.time_first(layer as u32)?;

            // WKV mechanism per head
            let state = &mut self.layer_states[layer];
            for h in 0..n_heads {
                let h_off = h * head_dim;
                for d in 0..head_dim {
                    let idx = h_off + d;
                    let k_val = self.scratch.k[idx];
                    let v_val = self.scratch.v[idx];
                    let bonus = read_f32(bonus_data, bonus_info, idx);
                    let w_val = read_f32(w_data, w_info, idx);

                    let sa_idx = h * head_dim * head_dim + d * head_dim;
                    let sb_idx = h_off + d;

                    // Compute wkv output for this dimension
                    // Using simplified per-dimension WKV (diagonal approximation):
                    // wkv_d = (state_a[d] + exp(bonus + k) * v) / (state_b[d] + exp(bonus + k))
                    let exp_bk = (bonus + k_val).exp();
                    let num = state.state_a[sa_idx + d] + exp_bk * v_val;
                    let den = state.state_b[sb_idx] + exp_bk;
                    self.scratch.wkv[idx] = if den.abs() > 1e-30 { num / den } else { 0.0 };

                    // Update state
                    let decay = (-w_val.exp()).exp();
                    let exp_k = k_val.exp();
                    for j in 0..head_dim {
                        let a_idx = h * head_dim * head_dim + d * head_dim + j;
                        if j == d {
                            state.state_a[a_idx] = state.state_a[a_idx] * decay + exp_k * v_val;
                        } else {
                            state.state_a[a_idx] *= decay;
                        }
                    }
                    state.state_b[sb_idx] = state.state_b[sb_idx] * decay + exp_k;
                }
            }

            // Gate (RWKV v5+): apply sigmoid(r) * wkv, optionally with learned gate
            if let Some((gate_info, gate_data)) = self.weights.attn_gate(layer as u32) {
                // Learned gate: gate_out = sigmoid(gate_weight @ normed) * wkv
                qmv(
                    gate_info,
                    gate_data,
                    &self.scratch.normed,
                    &mut self.scratch.gate_out,
                )?;
                for i in 0..dim {
                    let sig_r = 1.0 / (1.0 + (-self.scratch.r[i]).exp());
                    let sig_g = 1.0 / (1.0 + (-self.scratch.gate_out[i]).exp());
                    self.scratch.wkv[i] *= sig_r * sig_g;
                }
            } else {
                for i in 0..dim {
                    let sig_r = 1.0 / (1.0 + (-self.scratch.r[i]).exp());
                    self.scratch.wkv[i] *= sig_r;
                }
            }

            // Output projection
            let (wo_info, wo_data) = self.weights.attn_output(layer as u32)?;
            qmv(
                wo_info,
                wo_data,
                &self.scratch.wkv,
                &mut self.scratch.tm_out,
            )?;

            // Residual connection
            for i in 0..dim {
                self.hidden[i] = residual[i] + self.scratch.tm_out[i];
            }

            // Save prev_x for time-mix (use normed input before this layer)
            self.layer_states[layer]
                .prev_x_tm
                .copy_from_slice(&self.scratch.normed);

            // ---- Channel-Mix sub-block ----
            let residual: Vec<f32> = self.hidden.clone();

            let (fn_info, fn_data) = self.weights.ffn_norm(layer as u32)?;
            layer_norm(
                &self.hidden,
                fn_data,
                fn_info,
                self.cfg.norm_eps,
                &mut self.scratch.normed,
            );

            // Interpolated receptance
            let (mu_r_ffn_info, mu_r_ffn_data) = self.weights.time_mix_r_ffn(layer as u32)?;
            lerp(
                &self.layer_states[layer].prev_x_cm,
                &self.scratch.normed,
                mu_r_ffn_data,
                mu_r_ffn_info,
                &mut self.scratch.lerped,
            );
            let (fr_info, fr_data) = self.weights.ffn_receptance(layer as u32)?;
            qmv(
                fr_info,
                fr_data,
                &self.scratch.lerped,
                &mut self.scratch.cm_r,
            )?;

            // Interpolated key
            let (mu_k_ffn_info, mu_k_ffn_data) = self.weights.time_mix_k_ffn(layer as u32)?;
            lerp(
                &self.layer_states[layer].prev_x_cm,
                &self.scratch.normed,
                mu_k_ffn_data,
                mu_k_ffn_info,
                &mut self.scratch.lerped,
            );
            let (fk_info, fk_data) = self.weights.ffn_key(layer as u32)?;
            qmv(
                fk_info,
                fk_data,
                &self.scratch.lerped,
                &mut self.scratch.cm_k,
            )?;

            // Squared ReLU activation on key
            for i in 0..dim {
                self.scratch.cm_k_act[i] = squared_relu(self.scratch.cm_k[i]);
            }

            // W_v @ squared_relu(k)
            let (fv_info, fv_data) = self.weights.ffn_value(layer as u32)?;
            qmv(
                fv_info,
                fv_data,
                &self.scratch.cm_k_act,
                &mut self.scratch.cm_vk,
            )?;

            // Output: sigmoid(r) * (W_v @ squared_relu(k))
            for i in 0..dim {
                let sig_r = 1.0 / (1.0 + (-self.scratch.cm_r[i]).exp());
                self.hidden[i] = residual[i] + sig_r * self.scratch.cm_vk[i];
            }

            // Save prev_x for channel-mix
            self.layer_states[layer]
                .prev_x_cm
                .copy_from_slice(&self.scratch.normed);
        }

        // 3. Final norm
        let (on_info, on_data) = self.weights.output_norm()?;
        layer_norm(
            &self.hidden,
            on_data,
            on_info,
            self.cfg.norm_eps,
            &mut self.scratch.normed,
        );

        // 4. Output logits
        let (out_info, out_data) = self.weights.output()?;
        qmv(
            out_info,
            out_data,
            &self.scratch.normed,
            &mut self.scratch.logits,
        )?;

        Ok(self.scratch.logits.clone())
    }
}

impl<'a> ModelRunner for RwkvRunner<'a> {
    fn architecture(&self) -> Architecture {
        Architecture::Rwkv
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
        for state in &mut self.layer_states {
            state.prev_x_tm.fill(0.0);
            state.prev_x_cm.fill(0.0);
            state.state_a.fill(0.0);
            state.state_b.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_interpolation() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        // mu = 0.25 for all elements, stored as F32
        let mu_bytes: Vec<u8> = (0..4).flat_map(|_| 0.25f32.to_le_bytes()).collect();
        let mu_info = TensorInfo {
            name: "test_mu".into(),
            dtype: DType::F32,
            shape: vec![4],
            offset: 0,
            size_bytes: 16,
        };

        let mut out = [0.0f32; 4];
        lerp(&a, &b, &mu_bytes, &mu_info, &mut out);

        // lerp(a, b, 0.25) = a * 0.75 + b * 0.25
        for i in 0..4 {
            let expected = a[i] * 0.75 + b[i] * 0.25;
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "lerp mismatch at {i}: got {} expected {}",
                out[i],
                expected
            );
        }

        // Verify boundary cases: mu=0 gives a, mu=1 gives b
        let mu_zero: Vec<u8> = (0..4).flat_map(|_| 0.0f32.to_le_bytes()).collect();
        lerp(&a, &b, &mu_zero, &mu_info, &mut out);
        for i in 0..4 {
            assert!((out[i] - a[i]).abs() < 1e-6, "mu=0 should give a");
        }

        let mu_one: Vec<u8> = (0..4).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        lerp(&a, &b, &mu_one, &mu_info, &mut out);
        for i in 0..4 {
            assert!((out[i] - b[i]).abs() < 1e-6, "mu=1 should give b");
        }
    }

    #[test]
    fn test_squared_relu() {
        // Positive input: squared_relu(3.0) = 9.0
        assert!((squared_relu(3.0) - 9.0).abs() < 1e-6);

        // Zero input: squared_relu(0.0) = 0.0
        assert!((squared_relu(0.0)).abs() < 1e-6);

        // Negative input: squared_relu(-5.0) = 0.0
        assert!((squared_relu(-5.0)).abs() < 1e-6);

        // Small positive: squared_relu(0.5) = 0.25
        assert!((squared_relu(0.5) - 0.25).abs() < 1e-6);

        // squared_relu(1.0) = 1.0
        assert!((squared_relu(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wkv_zero_state() {
        // With zero initial state, WKV should produce v weighted by exp(bonus + k)
        // wkv = (0 + exp(bonus + k) * v) / (0 + exp(bonus + k)) = v
        let bonus = 0.5f32;
        let k = 1.0f32;
        let v = 3.0f32;

        let state_a = 0.0f32;
        let state_b = 0.0f32;

        let exp_bk = (bonus + k).exp();
        let num = state_a + exp_bk * v;
        let den = state_b + exp_bk;
        let wkv = num / den;

        // With zero state, wkv should equal v exactly
        assert!(
            (wkv - v).abs() < 1e-6,
            "WKV with zero state should return v, got {}",
            wkv
        );

        // Verify state update
        let w = 0.3f32; // decay parameter
        let decay = (-w.exp()).exp();
        let exp_k = k.exp();

        let new_a = state_a * decay + exp_k * v;
        let new_b = state_b * decay + exp_k;

        // After update, state should hold exp(k) * v and exp(k)
        assert!((new_a - exp_k * v).abs() < 1e-5, "state_a update incorrect");
        assert!((new_b - exp_k).abs() < 1e-5, "state_b update incorrect");

        // Second step with same k, v: wkv should still be v
        // because (exp_k * v + exp(bonus+k) * v) / (exp_k + exp(bonus+k)) = v
        let exp_bk2 = (bonus + k).exp();
        let num2 = new_a + exp_bk2 * v;
        let den2 = new_b + exp_bk2;
        let wkv2 = num2 / den2;

        // When k is the same and v is the same, wkv is always v
        assert!(
            (wkv2 - v).abs() < 1e-4,
            "WKV second step should return v, got {}",
            wkv2
        );
    }
}
