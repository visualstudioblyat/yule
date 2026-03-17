//! Mamba (Selective State Space Model) runner.
//!
//! Mamba replaces attention with a selective state-space mechanism:
//! - Linear time complexity O(L) vs O(L²) for transformers
//! - Constant memory per token (state instead of KV cache)
//! - 5x throughput vs transformers on long sequences
//!
//! Architecture per layer:
//! 1. Norm → linear projection (input → 2*d_inner)
//! 2. Split into x and z branches
//! 3. x: Conv1D(d_inner, kernel_size) → SiLU → SSM
//! 4. z: SiLU gate
//! 5. output = (SSM(x) * SiLU(z)) → output projection
//!
//! The SSM (Structured State Space) operation:
//!   h[t] = A_bar * h[t-1] + B_bar * x[t]
//!   y[t] = C * h[t] + D * x[t]
//! where A_bar, B_bar are discretized from continuous parameters via ZOH.

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

    // Compute actual number of rows from data size to handle shape mismatches
    let total_blocks = if block_bytes > 0 {
        weight_data.len() / block_bytes
    } else {
        0
    };
    let actual_rows = if blocks_per_row > 0 {
        total_blocks / blocks_per_row
    } else {
        0
    };
    let safe_rows = n_rows.min(actual_rows);

    for row in 0..safe_rows {
        let mut sum = 0.0f32;
        let row_offset = row * blocks_per_row * block_bytes;
        for b in 0..blocks_per_row {
            let block_start = row_offset + b * block_bytes;
            if block_start + block_bytes > weight_data.len() {
                break;
            }
            let block = &weight_data[block_start..block_start + block_bytes];
            let act_start = b * block_size;
            if act_start + block_size > input.len() {
                break;
            }
            let act = &input[act_start..act_start + block_size];
            sum += dequant::vec_dot_block(dtype, block, act).unwrap_or_else(|_| {
                let mut tmp = vec![0.0f32; block_size];
                let _ = dequant::dequant_block(dtype, block, &mut tmp);
                tmp.iter().zip(act.iter()).map(|(w, a)| w * a).sum::<f32>()
            });
        }
        out[row] = sum;
    }
    // Zero remaining rows if actual data was shorter than expected
    for row in safe_rows..n_rows {
        out[row] = 0.0;
    }
    Ok(())
}

/// Dequantize a full tensor to f32. Works for any dtype including quantized formats.
fn dequant_tensor(info: &TensorInfo, data: &[u8]) -> Vec<f32> {
    let dtype = info.dtype;
    let n_elements = info.num_elements() as usize;

    if dtype == DType::F32 {
        let available = data.len() / 4;
        let count = n_elements.min(available);
        let mut out = vec![0.0f32; n_elements];
        for i in 0..count {
            out[i] = f32::from_le_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
        }
        return out;
    }

    if dtype == DType::F16 {
        let available = data.len() / 2;
        let count = n_elements.min(available);
        let mut out = vec![0.0f32; n_elements];
        for i in 0..count {
            let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            out[i] = dequant::f16_to_f32(bits);
        }
        return out;
    }

    // Quantized format: dequantize block by block
    let block_size = dtype.block_size();
    let block_bytes = dtype.size_of_block();
    if block_bytes == 0 {
        return vec![0.0f32; n_elements];
    }
    let n_blocks = data.len() / block_bytes;
    if n_blocks == 0 {
        return vec![0.0f32; n_elements];
    }
    let mut out = vec![0.0f32; n_blocks * block_size];
    for b in 0..n_blocks {
        let block = &data[b * block_bytes..(b + 1) * block_bytes];
        let _ =
            dequant::dequant_block(dtype, block, &mut out[b * block_size..(b + 1) * block_size]);
    }
    // Truncate to requested element count (quantized blocks may produce padding)
    out.truncate(n_elements);
    out
}

/// Read a single f32 value from a tensor at a given element index, handling any dtype.
/// Returns 0.0 if the index is out of bounds for the underlying data.
fn read_weight_f32(info: &TensorInfo, data: &[u8], index: usize) -> f32 {
    match info.dtype {
        DType::F32 => {
            let byte_idx = index * 4;
            if byte_idx + 4 > data.len() {
                return 0.0;
            }
            f32::from_le_bytes([
                data[byte_idx],
                data[byte_idx + 1],
                data[byte_idx + 2],
                data[byte_idx + 3],
            ])
        }
        DType::F16 => {
            let byte_idx = index * 2;
            if byte_idx + 2 > data.len() {
                return 0.0;
            }
            let bits = u16::from_le_bytes([data[byte_idx], data[byte_idx + 1]]);
            dequant::f16_to_f32(bits)
        }
        _ => {
            // For quantized types, dequant the containing block and extract
            let bs = info.dtype.block_size();
            let bb = info.dtype.size_of_block();
            let block_idx = index / bs;
            let byte_start = block_idx * bb;
            if byte_start + bb > data.len() {
                return 0.0;
            }
            let block = &data[byte_start..byte_start + bb];
            let mut tmp = vec![0.0f32; bs];
            let _ = dequant::dequant_block(info.dtype, block, &mut tmp);
            tmp[index % bs]
        }
    }
}

fn rms_norm(x: &[f32], weight_data: &[u8], weight_info: &TensorInfo, eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();

    for i in 0..n {
        let w = read_weight_f32(weight_info, weight_data, i);
        out[i] = x[i] * inv * w;
    }
}

struct MambaConfig {
    dim: usize,
    n_layers: usize,
    d_inner: usize, // typically 2 * dim
    d_state: usize, // SSM state dimension (typically 16)
    d_conv: usize,  // conv1d kernel size (typically 4)
    dt_rank: usize, // rank of dt projection (typically dim / 16)
    vocab_size: usize,
    norm_eps: f32,
}

struct MambaLayerState {
    conv_state: Vec<f32>, // [d_inner, d_conv] — ring buffer for conv1d
    ssm_state: Vec<f32>,  // [d_inner, d_state] — recurrent state
}

struct MambaScratch {
    normed: Vec<f32>,
    xz: Vec<f32>,       // [2 * d_inner] — combined x and z from in_proj
    x: Vec<f32>,        // [d_inner]
    z: Vec<f32>,        // [d_inner]
    x_conv: Vec<f32>,   // [d_inner] — after conv1d
    dt: Vec<f32>,       // [d_inner] — discretization timestep
    dt_proj: Vec<f32>,  // [dt_rank]
    b: Vec<f32>,        // [d_state]
    c: Vec<f32>,        // [d_state]
    y: Vec<f32>,        // [d_inner] — SSM output
    out_proj: Vec<f32>, // [dim]
    logits: Vec<f32>,
}

pub struct MambaRunner<'a> {
    cfg: MambaConfig,
    weights: MambaWeights<'a>,
    layer_states: Vec<MambaLayerState>,
    hidden: Vec<f32>,
    scratch: MambaScratch,
    // Pre-dequantized small per-layer tensors (avoids repeated quantized reads
    // and prevents index-out-of-bounds when small tensors have a different dtype
    // than the main weight matrices in quantized GGUF files).
    conv_weights: Vec<Vec<f32>>,           // [n_layers][d_inner * d_conv]
    conv_biases: Vec<Option<Vec<f32>>>,    // [n_layers][d_inner] or None
    a_logs: Vec<Vec<f32>>,                 // [n_layers][d_inner * d_state]
    d_params: Vec<Option<Vec<f32>>>,       // [n_layers][d_inner] or None
    dt_proj_biases: Vec<Option<Vec<f32>>>, // [n_layers][d_inner] or None
}

struct MambaWeights<'a> {
    store: &'a WeightStore<'a>,
}

impl<'a> MambaWeights<'a> {
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
    fn layer_norm(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.attn_norm.weight"))
    }
    fn in_proj(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ssm_in.weight"))
    }
    fn dt_proj_weight(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ssm_dt.weight"))
    }
    fn out_proj(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ssm_out.weight"))
    }
    fn x_proj(&self, layer: u32) -> Result<(&TensorInfo, &[u8])> {
        self.store.require(&format!("blk.{layer}.ssm_x.weight"))
    }
}

impl<'a> MambaRunner<'a> {
    pub fn new(store: &'a WeightStore<'a>) -> Result<Self> {
        let meta = &store.meta;
        let dim = meta.embedding_dim as usize;
        let n_layers = meta.layer_count as usize;

        // Infer SSM dimensions from tensor shapes when available,
        // falling back to Mamba defaults if tensors are missing.
        let d_inner = if let Some((info, _)) = store.get("blk.0.ssm_in.weight") {
            // in_proj maps dim → 2*d_inner, so shape[0] = 2*d_inner
            if info.shape.len() >= 2 {
                info.shape[0] as usize / 2
            } else {
                2 * dim
            }
        } else {
            2 * dim
        };

        let d_conv = if let Some((info, _)) = store.get("blk.0.ssm_conv1d.weight") {
            // conv1d weight shape: [d_inner, 1, d_conv] or [d_inner, d_conv]
            *info.shape.last().unwrap_or(&4) as usize
        } else {
            4
        };

        let (dt_rank, d_state) = if let Some((info, _)) = store.get("blk.0.ssm_x.weight") {
            // x_proj maps d_inner → dt_rank + 2*d_state, so shape[0] = dt_rank + 2*d_state
            if info.shape.len() >= 2 {
                let x_proj_out = info.shape[0] as usize;
                // Assume d_state=16 (universal across known Mamba models) to solve for dt_rank
                let ds = 16usize;
                let dr = x_proj_out.saturating_sub(2 * ds);
                (dr, ds)
            } else {
                (dim.div_ceil(16), 16)
            }
        } else {
            (dim.div_ceil(16), 16)
        };
        let vocab_size = meta.vocab_size as usize;

        let cfg = MambaConfig {
            dim,
            n_layers,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            vocab_size,
            norm_eps: meta.norm_eps.unwrap_or(1e-5),
        };

        let mut layer_states = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layer_states.push(MambaLayerState {
                conv_state: vec![0.0; d_inner * d_conv],
                ssm_state: vec![0.0; d_inner * d_state],
            });
        }

        // Pre-dequantize small per-layer tensors. These don't change between
        // tokens and may have a different dtype than the main weight matrices
        // in quantized GGUF files (e.g. F32 while the model is Q4_K_M).
        let mut conv_weights = Vec::with_capacity(n_layers);
        let mut conv_biases = Vec::with_capacity(n_layers);
        let mut a_logs = Vec::with_capacity(n_layers);
        let mut d_params = Vec::with_capacity(n_layers);
        let mut dt_proj_biases = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            let l = layer as u32;

            // conv1d weight
            if let Some((info, data)) = store.get(&format!("blk.{l}.ssm_conv1d.weight")) {
                conv_weights.push(dequant_tensor(info, data));
            } else {
                conv_weights.push(vec![0.0; d_inner * d_conv]);
            }

            // conv1d bias (optional)
            conv_biases.push(
                store
                    .get(&format!("blk.{l}.ssm_conv1d.bias"))
                    .map(|(info, data)| dequant_tensor(info, data)),
            );

            // A_log
            if let Some((info, data)) = store.get(&format!("blk.{l}.ssm_a")) {
                a_logs.push(dequant_tensor(info, data));
            } else {
                a_logs.push(vec![0.0; d_inner * d_state]);
            }

            // D (optional)
            d_params.push(
                store
                    .get(&format!("blk.{l}.ssm_d"))
                    .map(|(info, data)| dequant_tensor(info, data)),
            );

            // dt_proj bias (optional)
            dt_proj_biases.push(
                store
                    .get(&format!("blk.{l}.ssm_dt.bias"))
                    .map(|(info, data)| dequant_tensor(info, data)),
            );
        }

        let scratch = MambaScratch {
            normed: vec![0.0; dim],
            xz: vec![0.0; 2 * d_inner],
            x: vec![0.0; d_inner],
            z: vec![0.0; d_inner],
            x_conv: vec![0.0; d_inner],
            dt: vec![0.0; d_inner],
            dt_proj: vec![0.0; dt_rank],
            b: vec![0.0; d_state],
            c: vec![0.0; d_state],
            y: vec![0.0; d_inner],
            out_proj: vec![0.0; dim],
            logits: vec![0.0; vocab_size],
        };

        Ok(Self {
            cfg,
            weights: MambaWeights { store },
            layer_states,
            hidden: vec![0.0; dim],
            scratch,
            conv_weights,
            conv_biases,
            a_logs,
            d_params,
            dt_proj_biases,
        })
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let dim = self.cfg.dim;
        let d_inner = self.cfg.d_inner;
        let d_state = self.cfg.d_state;
        let d_conv = self.cfg.d_conv;

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

        // 2. Mamba layers
        for layer in 0..self.cfg.n_layers {
            let residual: Vec<f32> = self.hidden.clone();

            // Layer norm
            let (norm_info, norm_data) = self.weights.layer_norm(layer as u32)?;
            rms_norm(
                &self.hidden,
                norm_data,
                norm_info,
                self.cfg.norm_eps,
                &mut self.scratch.normed,
            );

            // In projection: normed → [x, z] (2 * d_inner)
            let (in_info, in_data) = self.weights.in_proj(layer as u32)?;
            qmv(in_info, in_data, &self.scratch.normed, &mut self.scratch.xz)?;

            // Split into x and z
            self.scratch.x[..d_inner].copy_from_slice(&self.scratch.xz[..d_inner]);
            self.scratch.z[..d_inner].copy_from_slice(&self.scratch.xz[d_inner..2 * d_inner]);

            // Conv1D: shift conv state and apply
            let state = &mut self.layer_states[layer];
            // Shift conv state: move columns left, insert x as rightmost column
            for i in 0..d_inner {
                for k in 0..d_conv - 1 {
                    state.conv_state[i * d_conv + k] = state.conv_state[i * d_conv + k + 1];
                }
                state.conv_state[i * d_conv + d_conv - 1] = self.scratch.x[i];
            }

            // Conv1D output: for each d_inner channel, dot product with conv kernel
            // Uses pre-dequantized weights to avoid dtype mismatch crashes on quantized models
            let conv_w = &self.conv_weights[layer];
            for i in 0..d_inner {
                let mut sum = 0.0f32;
                for k in 0..d_conv {
                    let w_idx = i * d_conv + k;
                    let w = if w_idx < conv_w.len() {
                        conv_w[w_idx]
                    } else {
                        0.0
                    };
                    sum += state.conv_state[i * d_conv + k] * w;
                }
                self.scratch.x_conv[i] = sum;
            }

            // Add conv bias if present
            if let Some(ref bias) = self.conv_biases[layer] {
                for i in 0..d_inner {
                    if i < bias.len() {
                        self.scratch.x_conv[i] += bias[i];
                    }
                }
            }

            // SiLU on x_conv
            for i in 0..d_inner {
                let x = self.scratch.x_conv[i];
                self.scratch.x_conv[i] = x / (1.0 + (-x).exp());
            }

            // SSM: x_proj → [dt_proj_input, B, C]
            let (xp_info, xp_data) = self.weights.x_proj(layer as u32)?;
            let x_proj_dim = self.cfg.dt_rank + 2 * d_state;
            let mut x_proj_out = vec![0.0f32; x_proj_dim];
            qmv(xp_info, xp_data, &self.scratch.x_conv, &mut x_proj_out)?;

            // Split x_proj output
            self.scratch.dt_proj[..self.cfg.dt_rank]
                .copy_from_slice(&x_proj_out[..self.cfg.dt_rank]);
            self.scratch.b[..d_state]
                .copy_from_slice(&x_proj_out[self.cfg.dt_rank..self.cfg.dt_rank + d_state]);
            self.scratch.c[..d_state].copy_from_slice(
                &x_proj_out[self.cfg.dt_rank + d_state..self.cfg.dt_rank + 2 * d_state],
            );

            // dt = softplus(dt_proj_weight @ dt_proj_input + dt_proj_bias)
            let (dt_info, dt_data) = self.weights.dt_proj_weight(layer as u32)?;
            qmv(
                dt_info,
                dt_data,
                &self.scratch.dt_proj[..self.cfg.dt_rank],
                &mut self.scratch.dt,
            )?;
            if let Some(ref bias) = self.dt_proj_biases[layer] {
                for i in 0..d_inner {
                    if i < bias.len() {
                        self.scratch.dt[i] += bias[i];
                    }
                }
            }
            // Softplus: log(1 + exp(x))
            for i in 0..d_inner {
                self.scratch.dt[i] = (1.0 + self.scratch.dt[i].exp()).ln();
            }

            // SSM step: h = A_bar * h + B_bar * x, y = C * h + D * x
            // A_bar = exp(dt * A), B_bar = dt * B
            // Uses pre-dequantized A_log and D to avoid dtype mismatch crashes
            let a_log = &self.a_logs[layer];
            let d_param = &self.d_params[layer];
            for i in 0..d_inner {
                let dt_i = self.scratch.dt[i];
                let x_i = self.scratch.x_conv[i];

                let mut y_i = 0.0f32;
                for j in 0..d_state {
                    let a_idx = i * d_state + j;
                    let a_val = if a_idx < a_log.len() {
                        a_log[a_idx]
                    } else {
                        0.0
                    };
                    // A is stored as -exp(a_log), so A = -exp(a_val)
                    let a = -(a_val.exp());
                    let a_bar = (dt_i * a).exp();
                    let b_bar = dt_i * self.scratch.b[j];

                    // Update state
                    let h = &mut state.ssm_state[i * d_state + j];
                    *h = a_bar * *h + b_bar * x_i;
                    y_i += self.scratch.c[j] * *h;
                }

                // D * x (skip connection)
                if let Some(d_vec) = d_param {
                    if i < d_vec.len() {
                        y_i += d_vec[i] * x_i;
                    }
                }

                self.scratch.y[i] = y_i;
            }

            // Gate: y = y * SiLU(z)
            for i in 0..d_inner {
                let z = self.scratch.z[i];
                let silu_z = z / (1.0 + (-z).exp());
                self.scratch.y[i] *= silu_z;
            }

            // Output projection
            let (out_info, out_data) = self.weights.out_proj(layer as u32)?;
            qmv(
                out_info,
                out_data,
                &self.scratch.y,
                &mut self.scratch.out_proj,
            )?;

            // Residual
            for i in 0..dim {
                self.hidden[i] = residual[i] + self.scratch.out_proj[i];
            }
        }

        // 3. Final norm
        let (on_info, on_data) = self.weights.output_norm()?;
        rms_norm(
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

impl<'a> ModelRunner for MambaRunner<'a> {
    fn architecture(&self) -> Architecture {
        Architecture::Mamba
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
            state.conv_state.fill(0.0);
            state.ssm_state.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_softplus() {
        let x = 1.0f32;
        let sp = (1.0 + x.exp()).ln();
        assert!((sp - 1.3133).abs() < 1e-3);

        // softplus(0) = ln(2)
        let sp0 = (1.0 + 0.0f32.exp()).ln();
        assert!((sp0 - std::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn test_ssm_step_zero_state() {
        // With zero initial state and identity-like parameters,
        // the SSM should pass through the input scaled by B*dt*C
        let d_state = 2;
        let mut h = vec![0.0f32; d_state];
        let x = 1.0f32;
        let dt = 0.1f32;
        let a = -1.0f32; // negative (stable)
        let b = vec![1.0f32; d_state];
        let c = vec![1.0f32; d_state];

        let mut y = 0.0f32;
        for j in 0..d_state {
            let a_bar = (dt * a).exp();
            let b_bar = dt * b[j];
            h[j] = a_bar * h[j] + b_bar * x;
            y += c[j] * h[j];
        }

        // y should be approximately dt * x * d_state = 0.1 * 1.0 * 2 = 0.2
        assert!((y - 0.2).abs() < 1e-4, "y = {y}");
    }
}
