#![allow(clippy::needless_range_loop)]
use crate::{BackendKind, BufferHandle, ComputeBackend, DeviceInfo, buffer::next_buffer_handle};
use std::collections::HashMap;
use std::sync::Mutex;
use yule_core::error::{Result, YuleError};

pub struct CpuBackend {
    buffers: Mutex<HashMap<u64, Vec<u8>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
        }
    }

    /// Get a raw byte buffer, returning an error if not found.
    fn get_buf<'a>(
        buffers: &'a HashMap<u64, Vec<u8>>,
        handle: &BufferHandle,
    ) -> Result<&'a Vec<u8>> {
        buffers
            .get(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("buffer {} not found", handle.0)))
    }

    /// Get a mutable raw byte buffer.
    fn get_buf_mut<'a>(
        buffers: &'a mut HashMap<u64, Vec<u8>>,
        handle: &BufferHandle,
    ) -> Result<&'a mut Vec<u8>> {
        buffers
            .get_mut(&handle.0)
            .ok_or_else(|| YuleError::Gpu(format!("buffer {} not found", handle.0)))
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Reinterpret &mut [u8] as &mut [f32] (LE).
#[inline]
fn as_f32_slice_mut(data: &mut [u8]) -> &mut [f32] {
    debug_assert!(data.len() % 4 == 0);
    bytemuck::cast_slice_mut(data)
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "CPU".into(),
            backend: BackendKind::Cpu,
            memory_bytes: 0, // TODO: detect system RAM
            compute_units: std::thread::available_parallelism()
                .map(|p| p.get() as u32)
                .unwrap_or(1),
        }
    }

    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        let handle = next_buffer_handle();
        // Allocate aligned to 64 bytes for SIMD. Vec<u8> doesn't guarantee
        // this, but on most allocators, allocations >= 64 bytes are aligned.
        // For a production engine we'd use aligned_alloc or a custom allocator.
        let buf = vec![0u8; size_bytes];
        self.buffers.lock().unwrap().insert(handle.0, buf);
        Ok(handle)
    }

    fn free(&self, handle: BufferHandle) -> Result<()> {
        self.buffers.lock().unwrap().remove(&handle.0);
        Ok(())
    }

    /// Matrix multiply: C[m,n] = A[m,k] * B[k,n]
    /// For single-token decode (m=1), this is a GEMV.
    /// Buffers hold row-major f32 data.
    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        out: &BufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        // We need to borrow a, b immutably and out mutably.
        // Since they share one HashMap, we extract raw pointers carefully.
        let a_data = Self::get_buf(&buffers, a)?.as_ptr();
        let b_data = Self::get_buf(&buffers, b)?.as_ptr();
        let a_len = Self::get_buf(&buffers, a)?.len();
        let b_len = Self::get_buf(&buffers, b)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, out)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        // SAFETY: pointers stay valid while lock is held, and out is distinct
        let a_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(a_data, a_len) });
        let b_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(b_data, b_len) });

        let (m, n, k) = (m as usize, n as usize, k as usize);

        // Naive GEMM: C[i,j] = sum_p A[i,p] * B[p,j]
        // For m=1 this is a GEMV and dominates inference time.
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_f32[i * k + p] * b_f32[p * n + j];
                }
                out_f32[i * n + j] = sum;
            }
        }
        Ok(())
    }

    /// Softmax: out[i] = exp(input[i] - max) / sum(exp(input - max))
    /// Numerically stable via max subtraction.
    fn softmax(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp_data = Self::get_buf(&buffers, input)?.as_ptr();
        let inp_len = Self::get_buf(&buffers, input)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, output)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        let inp_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(inp_data, inp_len) });

        let n = size as usize;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            if inp_f32[i] > max_val {
                max_val = inp_f32[i];
            }
        }

        let mut sum = 0.0f32;
        for i in 0..n {
            let e = (inp_f32[i] - max_val).exp();
            out_f32[i] = e;
            sum += e;
        }

        let inv_sum = 1.0 / sum;
        for i in 0..n {
            out_f32[i] *= inv_sum;
        }
        Ok(())
    }

    /// RMSNorm: out = (input / rms) * weight
    /// where rms = sqrt(mean(input^2) + eps)
    fn rms_norm(
        &self,
        input: &BufferHandle,
        weight: &BufferHandle,
        output: &BufferHandle,
        size: u32,
        eps: f32,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp_data = Self::get_buf(&buffers, input)?.as_ptr();
        let inp_len = Self::get_buf(&buffers, input)?.len();
        let wt_data = Self::get_buf(&buffers, weight)?.as_ptr();
        let wt_len = Self::get_buf(&buffers, weight)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, output)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        let inp_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(inp_data, inp_len) });
        let wt_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(wt_data, wt_len) });

        let n = size as usize;
        let mut ss = 0.0f32;
        for i in 0..n {
            ss += inp_f32[i] * inp_f32[i];
        }
        let rms = (ss / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            out_f32[i] = inp_f32[i] * inv_rms * wt_f32[i];
        }
        Ok(())
    }

    /// RoPE (Rotary Position Embedding) applied in-place to Q and K buffers.
    /// head_dim: dimension per attention head (typically 128).
    /// Applies rotation in pairs: (q[2i], q[2i+1]) rotated by pos * freq.
    fn rope(
        &self,
        q: &BufferHandle,
        k: &BufferHandle,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        _n_heads_q: u32,
        _n_heads_k: u32,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();

        // Apply RoPE to both Q and K in sequence
        for handle in [q, k] {
            let buf = Self::get_buf_mut(&mut buffers, handle)?;
            let f32_data = as_f32_slice_mut(buf);
            let hd = head_dim as usize;
            let n_heads = f32_data.len() / hd;

            for h in 0..n_heads {
                let base = h * hd;
                for i in 0..(hd / 2) {
                    let freq = 1.0 / freq_base.powf(2.0 * i as f32 / hd as f32);
                    let theta = pos as f32 * freq;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    let x0 = f32_data[base + 2 * i];
                    let x1 = f32_data[base + 2 * i + 1];
                    f32_data[base + 2 * i] = x0 * cos_t - x1 * sin_t;
                    f32_data[base + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
        Ok(())
    }

    /// SiLU (Sigmoid Linear Unit): out[i] = input[i] * sigmoid(input[i])
    /// Also known as swish. Used in SwiGLU FFN.
    fn silu(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp_data = Self::get_buf(&buffers, input)?.as_ptr();
        let inp_len = Self::get_buf(&buffers, input)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, output)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        let inp_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(inp_data, inp_len) });

        let n = size as usize;
        for i in 0..n {
            let x = inp_f32[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            out_f32[i] = x * sigmoid;
        }
        Ok(())
    }

    /// Element-wise multiply: out[i] = a[i] * b[i]
    fn element_mul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let a_data = Self::get_buf(&buffers, a)?.as_ptr();
        let a_len = Self::get_buf(&buffers, a)?.len();
        let b_data = Self::get_buf(&buffers, b)?.as_ptr();
        let b_len = Self::get_buf(&buffers, b)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, output)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        let a_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(a_data, a_len) });
        let b_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(b_data, b_len) });

        let n = size as usize;
        for i in 0..n {
            out_f32[i] = a_f32[i] * b_f32[i];
        }
        Ok(())
    }

    /// Element-wise add: out[i] = a[i] + b[i]
    fn add(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let a_data = Self::get_buf(&buffers, a)?.as_ptr();
        let a_len = Self::get_buf(&buffers, a)?.len();
        let b_data = Self::get_buf(&buffers, b)?.as_ptr();
        let b_len = Self::get_buf(&buffers, b)?.len();
        let out_buf = Self::get_buf_mut(&mut buffers, output)?;
        let out_f32 = as_f32_slice_mut(out_buf);

        let a_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(a_data, a_len) });
        let b_f32: &[f32] =
            bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(b_data, b_len) });

        let n = size as usize;
        for i in 0..n {
            out_f32[i] = a_f32[i] + b_f32[i];
        }
        Ok(())
    }

    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get_mut(&handle.0)
            .ok_or_else(|| YuleError::Gpu("buffer not found".into()))?;
        buf[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle.0)
            .ok_or_else(|| YuleError::Gpu("buffer not found".into()))?;
        data.copy_from_slice(&buf[..data.len()]);
        Ok(())
    }

    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let src_ptr = Self::get_buf(&buffers, src)?.as_ptr();
        let src_len = Self::get_buf(&buffers, src)?.len();
        let dst_buf = Self::get_buf_mut(&mut buffers, dst)?;
        let n = size.min(src_len).min(dst_buf.len());
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, n) };
        dst_buf[..n].copy_from_slice(src_slice);
        Ok(())
    }

    fn copy_buffer_offset(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        let src_ptr = Self::get_buf(&buffers, src)?.as_ptr();
        let src_len = Self::get_buf(&buffers, src)?.len();
        let dst_buf = Self::get_buf_mut(&mut buffers, dst)?;
        let src_slice = unsafe {
            std::slice::from_raw_parts(src_ptr.add(src_offset), size.min(src_len - src_offset))
        };
        dst_buf[dst_offset..dst_offset + size].copy_from_slice(src_slice);
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(()) // CPU is synchronous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_f32(backend: &CpuBackend, handle: &BufferHandle, data: &[f32]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        backend.copy_to_device(bytes, handle).unwrap();
    }

    fn read_f32(backend: &CpuBackend, handle: &BufferHandle, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        backend.copy_from_device(handle, &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    #[test]
    fn test_softmax() {
        let b = CpuBackend::new();
        let inp = b.allocate(16).unwrap(); // 4 floats
        let out = b.allocate(16).unwrap();
        write_f32(&b, &inp, &[1.0, 2.0, 3.0, 4.0]);

        b.softmax(&inp, &out, 4).unwrap();
        let result = read_f32(&b, &out, 4);

        // Check sums to 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Check monotonicity
        assert!(result[3] > result[2]);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_rms_norm() {
        let b = CpuBackend::new();
        let inp = b.allocate(16).unwrap();
        let wt = b.allocate(16).unwrap();
        let out = b.allocate(16).unwrap();

        write_f32(&b, &inp, &[1.0, 2.0, 3.0, 4.0]);
        write_f32(&b, &wt, &[1.0, 1.0, 1.0, 1.0]);

        b.rms_norm(&inp, &wt, &out, 4, 1e-6).unwrap();
        let result = read_f32(&b, &out, 4);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-6).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 1e-4);
        assert!((result[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_silu() {
        let b = CpuBackend::new();
        let inp = b.allocate(12).unwrap();
        let out = b.allocate(12).unwrap();

        write_f32(&b, &inp, &[0.0, 1.0, -1.0]);
        b.silu(&inp, &out, 3).unwrap();
        let result = read_f32(&b, &out, 3);

        // silu(0) = 0 * 0.5 = 0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((result[1] - 0.7311).abs() < 1e-3);
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_element_mul() {
        let b = CpuBackend::new();
        let a = b.allocate(12).unwrap();
        let bh = b.allocate(12).unwrap();
        let out = b.allocate(12).unwrap();

        write_f32(&b, &a, &[2.0, 3.0, 4.0]);
        write_f32(&b, &bh, &[5.0, 6.0, 7.0]);
        b.element_mul(&a, &bh, &out, 3).unwrap();
        let result = read_f32(&b, &out, 3);

        assert!((result[0] - 10.0).abs() < 1e-5);
        assert!((result[1] - 18.0).abs() < 1e-5);
        assert!((result[2] - 28.0).abs() < 1e-5);
    }

    #[test]
    fn test_add() {
        let b = CpuBackend::new();
        let a = b.allocate(12).unwrap();
        let bh = b.allocate(12).unwrap();
        let out = b.allocate(12).unwrap();

        write_f32(&b, &a, &[1.0, 2.0, 3.0]);
        write_f32(&b, &bh, &[4.0, 5.0, 6.0]);
        b.add(&a, &bh, &out, 3).unwrap();
        let result = read_f32(&b, &out, 3);

        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 7.0).abs() < 1e-5);
        assert!((result[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_gemv() {
        // GEMV: 1×4 times 4×3 = 1×3
        let b = CpuBackend::new();
        let a = b.allocate(16).unwrap(); // 1×4
        let bh = b.allocate(48).unwrap(); // 4×3
        let out = b.allocate(12).unwrap(); // 1×3

        write_f32(&b, &a, &[1.0, 2.0, 3.0, 4.0]);
        // B row-major: [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
        write_f32(
            &b,
            &bh,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );

        b.matmul(&a, &bh, &out, 1, 3, 4).unwrap();
        let result = read_f32(&b, &out, 3);

        // C[0,0] = 1*1 + 2*0 + 3*0 + 4*1 = 5
        // C[0,1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
        // C[0,2] = 1*0 + 2*0 + 3*1 + 4*1 = 7
        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 6.0).abs() < 1e-5);
        assert!((result[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_single_head_pos0() {
        let b = CpuBackend::new();
        let q = b.allocate(16).unwrap(); // 1 head, head_dim=4
        let k = b.allocate(16).unwrap();

        write_f32(&b, &q, &[1.0, 0.0, 1.0, 0.0]);
        write_f32(&b, &k, &[0.0, 1.0, 0.0, 1.0]);

        b.rope(&q, &k, 0, 4, 10000.0, 1, 1).unwrap();
        let q_result = read_f32(&b, &q, 4);
        let k_result = read_f32(&b, &k, 4);

        // At pos=0, theta=0 for all freqs, so cos=1, sin=0 → no change
        assert!((q_result[0] - 1.0).abs() < 1e-5);
        assert!((q_result[1] - 0.0).abs() < 1e-5);
        assert!((k_result[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_multi_head() {
        let b = CpuBackend::new();
        // 2 Q heads, 1 KV head, head_dim=4
        let q = b.allocate(32).unwrap(); // 2 * 4 floats
        let k = b.allocate(16).unwrap(); // 1 * 4 floats

        write_f32(&b, &q, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        write_f32(&b, &k, &[1.0, 0.0, 1.0, 0.0]);

        // pos=0 → no rotation, but verify both heads are processed
        b.rope(&q, &k, 0, 4, 10000.0, 2, 1).unwrap();
        let q_result = read_f32(&b, &q, 8);
        let k_result = read_f32(&b, &k, 4);

        // Both Q heads should be unchanged at pos=0
        assert!((q_result[0] - 1.0).abs() < 1e-5); // head 0
        assert!((q_result[4] - 0.0).abs() < 1e-5); // head 1
        assert!((q_result[5] - 1.0).abs() < 1e-5); // head 1
        assert!((k_result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_nonzero_pos() {
        let b = CpuBackend::new();
        let q = b.allocate(16).unwrap(); // 1 head, head_dim=4
        let k = b.allocate(16).unwrap();

        write_f32(&b, &q, &[1.0, 0.0, 1.0, 0.0]);
        write_f32(&b, &k, &[1.0, 0.0, 1.0, 0.0]);

        b.rope(&q, &k, 5, 4, 10000.0, 1, 1).unwrap();
        let q_result = read_f32(&b, &q, 4);

        // At pos=5, pair (q[0], q[1]) should be rotated
        // freq = 1/10000^(0/4) = 1.0, theta = 5.0
        let cos5 = 5.0f32.cos();
        let sin5 = 5.0f32.sin();
        // q[0] = 1.0 * cos5 - 0.0 * sin5 = cos5
        assert!((q_result[0] - cos5).abs() < 1e-4);
        // q[1] = 1.0 * sin5 + 0.0 * cos5 = sin5
        assert!((q_result[1] - sin5).abs() < 1e-4);
    }

    #[test]
    fn test_copy_buffer() {
        let b = CpuBackend::new();
        let src = b.allocate(16).unwrap();
        let dst = b.allocate(16).unwrap();

        write_f32(&b, &src, &[1.0, 2.0, 3.0, 4.0]);
        b.copy_buffer(&src, &dst, 16).unwrap();
        let result = read_f32(&b, &dst, 4);

        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_copy_buffer_offset() {
        let b = CpuBackend::new();
        let src = b.allocate(16).unwrap(); // 4 floats
        let dst = b.allocate(32).unwrap(); // 8 floats, initially zeros

        write_f32(&b, &src, &[10.0, 20.0, 30.0, 40.0]);

        // Copy 2 floats from src offset 4 (starting at float[1]) to dst offset 8 (float[2])
        b.copy_buffer_offset(&src, &dst, 4, 8, 8).unwrap();
        let result = read_f32(&b, &dst, 8);

        assert!((result[0] - 0.0).abs() < 1e-5); // untouched
        assert!((result[1] - 0.0).abs() < 1e-5); // untouched
        assert!((result[2] - 20.0).abs() < 1e-5); // copied from src[1]
        assert!((result[3] - 30.0).abs() < 1e-5); // copied from src[2]
        assert!((result[4] - 0.0).abs() < 1e-5); // untouched
    }

    #[test]
    fn test_copy_buffer_offset_kv_cache_pattern() {
        // Simulate KV cache write: copy one position's worth of data to a specific offset
        let b = CpuBackend::new();
        let n_kv_heads = 2;
        let head_dim = 4;
        let kv_stride = n_kv_heads * head_dim; // 8 floats per position
        let max_seq_len = 4;

        let k_tmp = b.allocate(kv_stride * 4).unwrap();
        let k_cache = b.allocate(max_seq_len * kv_stride * 4).unwrap();

        // Write position data
        let k_data: Vec<f32> = (0..kv_stride).map(|i| (i + 1) as f32).collect();
        write_f32(&b, &k_tmp, &k_data);

        // Write to position 2
        let pos = 2;
        let cache_byte_offset = pos * kv_stride * 4;
        b.copy_buffer_offset(&k_tmp, &k_cache, 0, cache_byte_offset, kv_stride * 4)
            .unwrap();

        let cache = read_f32(&b, &k_cache, max_seq_len * kv_stride);

        // Position 0 and 1 should be zeros
        assert!((cache[0] - 0.0).abs() < 1e-5);
        assert!((cache[kv_stride - 1] - 0.0).abs() < 1e-5);
        // Position 2 should have our data
        assert!((cache[pos * kv_stride] - 1.0).abs() < 1e-5);
        assert!((cache[pos * kv_stride + kv_stride - 1] - kv_stride as f32).abs() < 1e-5);
        // Position 3 should be zeros
        assert!((cache[3 * kv_stride] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_manual() {
        // Test the full attention pipeline: score → softmax → value aggregation
        // Single head, head_dim=2, seq_len=3
        let b = CpuBackend::new();
        let hd = 2;
        let seq_len = 3;

        let q = b.allocate(hd * 4).unwrap();
        let k_cache = b.allocate(seq_len * hd * 4).unwrap();
        let v_cache = b.allocate(seq_len * hd * 4).unwrap();
        let scores = b.allocate(seq_len * 4).unwrap();
        let out = b.allocate(hd * 4).unwrap();

        // Q = [1, 0]
        write_f32(&b, &q, &[1.0, 0.0]);

        // K cache: 3 positions, each 2 dims
        // K[0] = [1, 0], K[1] = [0, 1], K[2] = [1, 1]
        write_f32(&b, &k_cache, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        // V cache: V[0] = [10, 20], V[1] = [30, 40], V[2] = [50, 60]
        write_f32(&b, &v_cache, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        // Compute scores manually: Q · K[t] / sqrt(hd)
        // score[0] = (1*1 + 0*0) / sqrt(2) = 1/sqrt(2) ≈ 0.7071
        // score[1] = (1*0 + 0*1) / sqrt(2) = 0
        // score[2] = (1*1 + 0*1) / sqrt(2) = 1/sqrt(2) ≈ 0.7071
        let scale = 1.0 / (hd as f32).sqrt();
        let s0 = 1.0 * scale;
        let s1 = 0.0 * scale;
        let s2 = 1.0 * scale;

        // Softmax
        let max_s = s0.max(s1).max(s2);
        let e0 = (s0 - max_s).exp();
        let e1 = (s1 - max_s).exp();
        let e2 = (s2 - max_s).exp();
        let sum = e0 + e1 + e2;
        let w0 = e0 / sum;
        let w1 = e1 / sum;
        let w2 = e2 / sum;

        // Weighted V: out = w0*V[0] + w1*V[1] + w2*V[2]
        let expected_0 = w0 * 10.0 + w1 * 30.0 + w2 * 50.0;
        let expected_1 = w0 * 20.0 + w1 * 40.0 + w2 * 60.0;

        // Now use the actual backend ops
        // Step 1: compute scores
        let q_f32 = read_f32(&b, &q, hd);
        let k_f32 = read_f32(&b, &k_cache, seq_len * hd);
        let mut scores_f32 = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q_f32[d] * k_f32[t * hd + d];
            }
            scores_f32[t] = dot * scale;
        }
        write_f32(&b, &scores, &scores_f32);

        // Step 2: softmax
        b.softmax(&scores, &scores, seq_len as u32).unwrap();
        let weights = read_f32(&b, &scores, seq_len);

        // Verify softmax weights match
        assert!((weights[0] - w0).abs() < 1e-4);
        assert!((weights[1] - w1).abs() < 1e-4);
        assert!((weights[2] - w2).abs() < 1e-4);

        // Step 3: weighted value sum
        let v_f32 = read_f32(&b, &v_cache, seq_len * hd);
        let mut out_f32 = vec![0.0f32; hd];
        for t in 0..seq_len {
            for d in 0..hd {
                out_f32[d] += weights[t] * v_f32[t * hd + d];
            }
        }
        write_f32(&b, &out, &out_f32);
        let result = read_f32(&b, &out, hd);

        assert!((result[0] - expected_0).abs() < 1e-3);
        assert!((result[1] - expected_1).abs() < 1e-3);
    }

    #[test]
    fn test_attention_gqa() {
        // 2 Q heads sharing 1 KV head (GQA ratio 2:1)
        // head_dim=2, seq_len=2
        let b = CpuBackend::new();
        let hd = 2;
        let n_heads = 2;
        let n_kv_heads = 1;
        let kv_stride = n_kv_heads * hd;
        let seq_len = 2;

        // Q: 2 heads × 2 dims = 4 floats
        let q = b.allocate(n_heads * hd * 4).unwrap();
        write_f32(&b, &q, &[1.0, 0.0, 0.0, 1.0]); // head0=[1,0], head1=[0,1]

        // KV cache: 2 positions × 1 kv head × 2 dims
        let k_cache = b.allocate(seq_len * kv_stride * 4).unwrap();
        let v_cache = b.allocate(seq_len * kv_stride * 4).unwrap();
        write_f32(&b, &k_cache, &[1.0, 0.0, 0.0, 1.0]); // K[0]=[1,0], K[1]=[0,1]
        write_f32(&b, &v_cache, &[10.0, 20.0, 30.0, 40.0]); // V[0]=[10,20], V[1]=[30,40]

        let scores_buf = b.allocate(seq_len * 4).unwrap();
        let attn_out = b.allocate(n_heads * hd * 4).unwrap();

        let scale = 1.0 / (hd as f32).sqrt();
        let kv_group = n_heads / n_kv_heads;

        // Process each Q head, both share the same KV head
        for h in 0..n_heads {
            let kv_h = h / kv_group;
            let head_offset = h * hd;
            let kv_off = kv_h * hd;

            // Compute scores for this head
            let q_f32 = read_f32(&b, &q, n_heads * hd);
            let k_f32 = read_f32(&b, &k_cache, seq_len * kv_stride);
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_f32[head_offset + d] * k_f32[t * kv_stride + kv_off + d];
                }
                scores[t] = dot * scale;
            }
            write_f32(&b, &scores_buf, &scores);

            b.softmax(&scores_buf, &scores_buf, seq_len as u32).unwrap();
            let weights = read_f32(&b, &scores_buf, seq_len);

            // Weighted value
            let v_f32 = read_f32(&b, &v_cache, seq_len * kv_stride);
            let mut head_out = vec![0.0f32; hd];
            for t in 0..seq_len {
                for d in 0..hd {
                    head_out[d] += weights[t] * v_f32[t * kv_stride + kv_off + d];
                }
            }

            // Write to attn_out at head offset
            let mut full_out = read_f32(&b, &attn_out, n_heads * hd);
            full_out[head_offset..head_offset + hd].copy_from_slice(&head_out);
            write_f32(&b, &attn_out, &full_out);
        }

        let result = read_f32(&b, &attn_out, n_heads * hd);

        // Head 0: Q=[1,0], K[0]=[1,0]→dot=1, K[1]=[0,1]→dot=0
        // Scores: [1/√2, 0], softmax gives more weight to position 0
        // Head 1: Q=[0,1], K[0]=[1,0]→dot=0, K[1]=[0,1]→dot=1
        // Scores: [0, 1/√2], softmax gives more weight to position 1

        // Head 0 should weight V[0]=[10,20] more heavily
        assert!(result[0] < 25.0); // closer to 10 than 30
        assert!(result[1] < 35.0); // closer to 20 than 40

        // Head 1 should weight V[1]=[30,40] more heavily
        assert!(result[2] > 15.0); // closer to 30 than 10
        assert!(result[3] > 25.0); // closer to 40 than 20
    }
}
