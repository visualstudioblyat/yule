use yule_core::dequant;
use yule_core::dtype::DType;
use yule_core::error::{Result, YuleError};
use yule_core::model::Architecture;
use yule_gpu::vk;
use yule_gpu::vulkan::VulkanBackend;
use yule_gpu::vulkan::pipeline::ShaderKey;
use yule_gpu::{BufferHandle, ComputeBackend};

use crate::model_runner::ModelRunner;
use crate::weight_loader::TransformerWeights;

struct GpuConfig {
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
    norm_weight_offset: f32,
}

struct LayerBuffers {
    attn_norm: BufferHandle,
    attn_q: BufferHandle,
    attn_q_dtype: DType,
    attn_k: BufferHandle,
    attn_k_dtype: DType,
    attn_v: BufferHandle,
    attn_v_dtype: DType,
    attn_output: BufferHandle,
    attn_output_dtype: DType,
    ffn_norm: BufferHandle,
    ffn_gate: BufferHandle,
    ffn_gate_dtype: DType,
    ffn_up: BufferHandle,
    ffn_up_dtype: DType,
    ffn_down: BufferHandle,
    ffn_down_dtype: DType,
}

struct GpuWeightBuffers {
    output_norm: BufferHandle,
    output_weight: BufferHandle,
    output_dtype: DType,
    layers: Vec<LayerBuffers>,
}

struct GpuScratch {
    hidden: BufferHandle,
    residual: BufferHandle,
    normed: BufferHandle,
    q: BufferHandle,
    k_tmp: BufferHandle,
    v_tmp: BufferHandle,
    attn_out: BufferHandle,
    attn_proj: BufferHandle,
    scores: BufferHandle,
    gate: BufferHandle,
    up: BufferHandle,
    ffn_out: BufferHandle,
    logits: BufferHandle,
}

pub struct GpuTransformerRunner<'a> {
    cfg: GpuConfig,
    arch: Architecture,
    backend: VulkanBackend,
    gpu_weights: GpuWeightBuffers,
    scratch: GpuScratch,
    k_cache: Vec<BufferHandle>,
    v_cache: Vec<BufferHandle>,
    pos: usize,

    // Keep CPU-side reference for embedding lookup (single row copy, negligible)
    cpu_weights: TransformerWeights<'a>,
}

impl<'a> GpuTransformerRunner<'a> {
    pub fn new(cpu_weights: TransformerWeights<'a>) -> Result<Self> {
        let backend = VulkanBackend::new()?;
        let meta = &cpu_weights.store.meta;
        let arch = meta.architecture.clone();

        let dim = meta.embedding_dim as usize;
        let n_heads = meta.head_count as usize;
        let n_kv_heads = meta.head_count_kv as usize;
        let n_layers = meta.layer_count as usize;
        let head_dim = dim / n_heads;
        let max_seq_len = meta.context_length as usize;
        let vocab_size = meta.vocab_size as usize;
        let is_gemma = matches!(arch, Architecture::Gemma);

        let ff_dim = if let Ok((info, _)) = cpu_weights.ffn_gate(0) {
            info.shape[1] as usize
        } else {
            ((dim as f64 * 8.0 / 3.0 / 256.0).ceil() as usize) * 256
        };

        let cfg = GpuConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            ff_dim,
            norm_eps: meta.norm_eps.unwrap_or(if is_gemma { 1e-6 } else { 1e-5 }),
            rope_freq_base: meta.rope_freq_base.unwrap_or(10000.0) as f32,
            max_seq_len,
            norm_weight_offset: if is_gemma { 1.0 } else { 0.0 },
        };

        let gpu_weights = Self::upload_weights(&backend, &cpu_weights, &cfg)?;

        // Allocate f32 scratch buffers
        let alloc_f32 = |n: usize| -> Result<BufferHandle> { backend.allocate(n * 4) };

        let scratch = GpuScratch {
            hidden: alloc_f32(dim)?,
            residual: alloc_f32(dim)?,
            normed: alloc_f32(dim)?,
            q: alloc_f32(n_heads * head_dim)?,
            k_tmp: alloc_f32(n_kv_heads * head_dim)?,
            v_tmp: alloc_f32(n_kv_heads * head_dim)?,
            attn_out: alloc_f32(n_heads * head_dim)?,
            attn_proj: alloc_f32(dim)?,
            scores: alloc_f32(max_seq_len)?,
            gate: alloc_f32(ff_dim)?,
            up: alloc_f32(ff_dim)?,
            ffn_out: alloc_f32(dim)?,
            logits: alloc_f32(vocab_size)?,
        };

        let kv_len = max_seq_len * n_kv_heads * head_dim;
        let mut k_cache = Vec::with_capacity(n_layers);
        let mut v_cache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_cache.push(alloc_f32(kv_len)?);
            v_cache.push(alloc_f32(kv_len)?);
        }

        let info = backend.device_info();
        tracing::info!(
            "GPU runner: {} ({:.0} MB)",
            info.name,
            info.memory_bytes as f64 / 1e6
        );

        Ok(Self {
            cfg,
            arch,
            backend,
            gpu_weights,
            scratch,
            k_cache,
            v_cache,
            pos: 0,
            cpu_weights,
        })
    }

    fn upload_weights(
        backend: &VulkanBackend,
        w: &TransformerWeights,
        cfg: &GpuConfig,
    ) -> Result<GpuWeightBuffers> {
        let upload_raw = |name: &str| -> Result<(BufferHandle, DType)> {
            let (info, data) = w.store.require(name)?;
            let handle = backend.allocate(data.len())?;
            backend.copy_to_device(data, &handle)?;
            Ok((handle, info.dtype))
        };

        let upload_norm = |name: &str| -> Result<BufferHandle> {
            let (info, data) = w.store.require(name)?;
            let n = cfg.dim;
            let mut f32_data = vec![0.0f32; n];
            if info.dtype == DType::F32 {
                for i in 0..n {
                    f32_data[i] = f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ]);
                }
            } else {
                for i in 0..n {
                    let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    f32_data[i] = dequant::f16_to_f32(bits);
                }
            }
            if cfg.norm_weight_offset != 0.0 {
                for v in &mut f32_data {
                    *v += cfg.norm_weight_offset;
                }
            }
            let bytes: &[u8] = bytemuck::cast_slice(&f32_data);
            let handle = backend.allocate(bytes.len())?;
            backend.copy_to_device(bytes, &handle)?;
            Ok(handle)
        };

        let (out_h, out_dt) =
            upload_raw("output.weight").or_else(|_| upload_raw("token_embd.weight"))?;
        let output_norm = upload_norm("output_norm.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let li = l as u32;
            let (aq, aq_dt) = upload_raw(&format!("blk.{li}.attn_q.weight"))?;
            let (ak, ak_dt) = upload_raw(&format!("blk.{li}.attn_k.weight"))?;
            let (av, av_dt) = upload_raw(&format!("blk.{li}.attn_v.weight"))?;
            let (ao, ao_dt) = upload_raw(&format!("blk.{li}.attn_output.weight"))?;
            let (fg, fg_dt) = upload_raw(&format!("blk.{li}.ffn_gate.weight"))?;
            let (fu, fu_dt) = upload_raw(&format!("blk.{li}.ffn_up.weight"))?;
            let (fd, fd_dt) = upload_raw(&format!("blk.{li}.ffn_down.weight"))?;

            layers.push(LayerBuffers {
                attn_norm: upload_norm(&format!("blk.{li}.attn_norm.weight"))?,
                attn_q: aq,
                attn_q_dtype: aq_dt,
                attn_k: ak,
                attn_k_dtype: ak_dt,
                attn_v: av,
                attn_v_dtype: av_dt,
                attn_output: ao,
                attn_output_dtype: ao_dt,
                ffn_norm: upload_norm(&format!("blk.{li}.ffn_norm.weight"))?,
                ffn_gate: fg,
                ffn_gate_dtype: fg_dt,
                ffn_up: fu,
                ffn_up_dtype: fu_dt,
                ffn_down: fd,
                ffn_down_dtype: fd_dt,
            });
        }

        Ok(GpuWeightBuffers {
            output_norm,
            output_weight: out_h,
            output_dtype: out_dt,
            layers,
        })
    }

    /// Map DType to ShaderKey for quantized matmul.
    fn qmv_key(dtype: DType) -> Result<ShaderKey> {
        match dtype {
            DType::Q4_0 => Ok(ShaderKey::QmvQ4_0),
            DType::Q4_K => Ok(ShaderKey::QmvQ4K),
            DType::Q6_K => Ok(ShaderKey::QmvQ6K),
            DType::Q8_0 => Ok(ShaderKey::QmvQ8_0),
            _ => Err(YuleError::Gpu(format!(
                "unsupported dtype for GPU qmv: {dtype:?}"
            ))),
        }
    }

    /// Record a quantized matmul dispatch into a command buffer.
    fn record_qmv(
        &self,
        cmd: vk::CommandBuffer,
        weights: &BufferHandle,
        input: &BufferHandle,
        output: &BufferHandle,
        n_rows: u32,
        n_cols: u32,
        dtype: DType,
    ) -> Result<()> {
        let key = Self::qmv_key(dtype)?;
        let block_size = dtype.block_size() as u32;
        let blocks_per_row = (n_cols + block_size - 1) / block_size;
        let push = [n_rows, n_cols, blocks_per_row];
        let push_bytes: &[u8] = bytemuck::cast_slice(&push);
        self.backend.dispatch_batched(
            cmd,
            key,
            &[weights, input, output],
            push_bytes,
            n_rows,
            1,
            1,
        )
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let dim = self.cfg.dim;
        let pos = self.pos;
        let n_heads = self.cfg.n_heads;
        let n_kv_heads = self.cfg.n_kv_heads;
        let hd = self.cfg.head_dim;
        let kv_stride = n_kv_heads * hd;
        let seq_len = (pos + 1) as u32;

        if pos >= self.cfg.max_seq_len {
            return Err(YuleError::Inference("context length exceeded".into()));
        }

        // 1. Embedding — dequant on CPU, upload f32 to hidden (separate transfer)
        let (embd_info, embd_data) = self.cpu_weights.token_embd()?;
        let embd_dtype = embd_info.dtype;
        let embd_row_bytes = embd_info.size_bytes as usize / self.cfg.vocab_size;
        let tok_offset = token as usize * embd_row_bytes;
        let tok_data = &embd_data[tok_offset..tok_offset + embd_row_bytes];

        let mut hidden_f32 = vec![0.0f32; dim];
        if embd_dtype == DType::F32 {
            for i in 0..dim {
                hidden_f32[i] = f32::from_le_bytes([
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
                dequant::dequant_block(embd_dtype, block, &mut hidden_f32[b * bs..(b + 1) * bs])?;
            }
        }

        if self.cfg.norm_weight_offset > 0.0 {
            let scale = (dim as f32).sqrt();
            for h in &mut hidden_f32 {
                *h *= scale;
            }
        }

        self.backend
            .copy_to_device(bytemuck::cast_slice(&hidden_f32), &self.scratch.hidden)?;

        // 2. Batched forward pass — single command buffer for all layers
        // Reset descriptor pool from previous forward pass (all sets are stale after fence wait)
        self.backend.reset_descriptors()?;
        let cmd = self.backend.begin_batch()?;

        for layer in 0..self.cfg.n_layers {
            let lb = &self.gpu_weights.layers[layer];
            let q_out_dim = (n_heads * hd) as u32;
            let kv_out_dim = (n_kv_heads * hd) as u32;
            let in_dim = dim as u32;
            let ff = self.cfg.ff_dim as u32;

            // residual = hidden (GPU copy, no barrier needed before — hidden is ready)
            self.backend.copy_buffer_batched(
                cmd,
                &self.scratch.hidden,
                &self.scratch.residual,
                0,
                0,
                dim * 4,
            )?;
            self.backend.transfer_barrier(cmd);

            // normed = rms_norm(hidden, attn_norm)
            {
                let push = [dim as u32, self.cfg.norm_eps.to_bits()];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::RmsNorm,
                    &[&self.scratch.hidden, &lb.attn_norm, &self.scratch.normed],
                    push_bytes,
                    1,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);

            // QKV projections (all read normed, write to independent buffers — no barriers between them)
            self.record_qmv(
                cmd,
                &lb.attn_q,
                &self.scratch.normed,
                &self.scratch.q,
                q_out_dim,
                in_dim,
                lb.attn_q_dtype,
            )?;
            self.record_qmv(
                cmd,
                &lb.attn_k,
                &self.scratch.normed,
                &self.scratch.k_tmp,
                kv_out_dim,
                in_dim,
                lb.attn_k_dtype,
            )?;
            self.record_qmv(
                cmd,
                &lb.attn_v,
                &self.scratch.normed,
                &self.scratch.v_tmp,
                kv_out_dim,
                in_dim,
                lb.attn_v_dtype,
            )?;
            self.backend.barrier(cmd);

            // RoPE (in-place on q and k_tmp)
            {
                let half_dim = (hd / 2) as u32;
                let total_threads = n_heads as u32 * half_dim + n_kv_heads as u32 * half_dim;
                let wg_x = (total_threads + 63) / 64;
                let push = [
                    pos as u32,
                    hd as u32,
                    self.cfg.rope_freq_base.to_bits(),
                    n_heads as u32,
                    n_kv_heads as u32,
                ];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::Rope,
                    &[
                        &self.scratch.q,
                        &self.scratch.k_tmp,
                        &self.scratch.q,
                        &self.scratch.k_tmp,
                    ],
                    push_bytes,
                    wg_x,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);

            // KV cache write (offset copy, k_tmp → k_cache, v_tmp → v_cache)
            let kv_bytes = kv_stride * 4;
            let cache_byte_offset = pos * kv_bytes;
            self.backend.copy_buffer_batched(
                cmd,
                &self.scratch.k_tmp,
                &self.k_cache[layer],
                0,
                cache_byte_offset,
                kv_bytes,
            )?;
            self.backend.copy_buffer_batched(
                cmd,
                &self.scratch.v_tmp,
                &self.v_cache[layer],
                0,
                cache_byte_offset,
                kv_bytes,
            )?;
            self.backend.transfer_barrier(cmd);

            // Attention — per-head GQA dispatch
            let kv_group = n_heads / n_kv_heads;
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let head_offset = (h * hd) as u32;
                let kv_off = (kv_h * hd) as u32;

                // attn_score: Q · K_cache → scores
                {
                    let push = [hd as u32, seq_len, head_offset, kv_off, kv_stride as u32];
                    let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                    self.backend.dispatch_batched(
                        cmd,
                        ShaderKey::AttnScore,
                        &[
                            &self.scratch.q,
                            &self.k_cache[layer],
                            &self.scratch.scores,
                            &self.scratch.scores,
                        ],
                        push_bytes,
                        seq_len,
                        1,
                        1,
                    )?;
                }
                self.backend.barrier(cmd);

                // softmax(scores) in-place
                {
                    let push = [seq_len];
                    let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                    self.backend.dispatch_batched(
                        cmd,
                        ShaderKey::Softmax,
                        &[&self.scratch.scores, &self.scratch.scores],
                        push_bytes,
                        1,
                        1,
                        1,
                    )?;
                }
                self.backend.barrier(cmd);

                // attn_value: scores × V_cache → attn_out[head_offset..]
                {
                    let push = [hd as u32, seq_len, kv_off, kv_stride as u32, head_offset];
                    let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                    self.backend.dispatch_batched(
                        cmd,
                        ShaderKey::AttnValue,
                        &[
                            &self.scratch.scores,
                            &self.v_cache[layer],
                            &self.scratch.attn_out,
                            &self.scratch.attn_out,
                        ],
                        push_bytes,
                        hd as u32,
                        1,
                        1,
                    )?;
                }
                self.backend.barrier(cmd);
            }

            // Output projection: attn_out → attn_proj
            self.record_qmv(
                cmd,
                &lb.attn_output,
                &self.scratch.attn_out,
                &self.scratch.attn_proj,
                in_dim,
                q_out_dim,
                lb.attn_output_dtype,
            )?;
            self.backend.barrier(cmd);

            // Residual: hidden = residual + attn_proj
            {
                let push = [dim as u32];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::Add,
                    &[
                        &self.scratch.residual,
                        &self.scratch.attn_proj,
                        &self.scratch.hidden,
                    ],
                    push_bytes,
                    (dim as u32 + 255) / 256,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);

            // FFN: residual = hidden
            self.backend.copy_buffer_batched(
                cmd,
                &self.scratch.hidden,
                &self.scratch.residual,
                0,
                0,
                dim * 4,
            )?;
            self.backend.transfer_barrier(cmd);

            // FFN norm
            {
                let push = [dim as u32, self.cfg.norm_eps.to_bits()];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::RmsNorm,
                    &[&self.scratch.hidden, &lb.ffn_norm, &self.scratch.normed],
                    push_bytes,
                    1,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);

            // Gate + Up projections (independent, no barrier between)
            self.record_qmv(
                cmd,
                &lb.ffn_gate,
                &self.scratch.normed,
                &self.scratch.gate,
                ff,
                in_dim,
                lb.ffn_gate_dtype,
            )?;
            self.record_qmv(
                cmd,
                &lb.ffn_up,
                &self.scratch.normed,
                &self.scratch.up,
                ff,
                in_dim,
                lb.ffn_up_dtype,
            )?;
            self.backend.barrier(cmd);

            // SwiGLU: gate = SiLU(gate) * up (fused single dispatch)
            {
                let push = [ff];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::SiluMul,
                    &[&self.scratch.gate, &self.scratch.up, &self.scratch.gate],
                    push_bytes,
                    (ff + 255) / 256,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);

            // Down projection
            self.record_qmv(
                cmd,
                &lb.ffn_down,
                &self.scratch.gate,
                &self.scratch.ffn_out,
                in_dim,
                ff,
                lb.ffn_down_dtype,
            )?;
            self.backend.barrier(cmd);

            // Residual: hidden = residual + ffn_out
            {
                let push = [dim as u32];
                let push_bytes: &[u8] = bytemuck::cast_slice(&push);
                self.backend.dispatch_batched(
                    cmd,
                    ShaderKey::Add,
                    &[
                        &self.scratch.residual,
                        &self.scratch.ffn_out,
                        &self.scratch.hidden,
                    ],
                    push_bytes,
                    (dim as u32 + 255) / 256,
                    1,
                    1,
                )?;
            }
            self.backend.barrier(cmd);
        }

        // 3. Final norm
        {
            let push = [dim as u32, self.cfg.norm_eps.to_bits()];
            let push_bytes: &[u8] = bytemuck::cast_slice(&push);
            self.backend.dispatch_batched(
                cmd,
                ShaderKey::RmsNorm,
                &[
                    &self.scratch.hidden,
                    &self.gpu_weights.output_norm,
                    &self.scratch.normed,
                ],
                push_bytes,
                1,
                1,
                1,
            )?;
        }
        self.backend.barrier(cmd);

        // 4. Output projection → logits
        self.record_qmv(
            cmd,
            &self.gpu_weights.output_weight,
            &self.scratch.normed,
            &self.scratch.logits,
            self.cfg.vocab_size as u32,
            dim as u32,
            self.gpu_weights.output_dtype,
        )?;

        // Submit the entire forward pass as one command buffer
        self.backend.submit_batch(cmd)?;

        // 5. Copy logits back to CPU (separate transfer)
        let mut logits_cpu = vec![0.0f32; self.cfg.vocab_size];
        self.backend.copy_from_device(
            &self.scratch.logits,
            bytemuck::cast_slice_mut(&mut logits_cpu),
        )?;

        self.pos += 1;
        Ok(logits_cpu)
    }
}

impl<'a> ModelRunner for GpuTransformerRunner<'a> {
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
