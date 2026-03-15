//! Hand-tuned assembly kernels for quantized GEMV operations.
//!
//! These are linked via the `cc` crate (see build.rs) and called through
//! FFI from the SIMD dispatch layer. Each kernel matches the contract of
//! its corresponding `vec_dot_*` scalar function in `dequant.rs`.

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    /// Q4_K fused dequant + dot product, AVX2 implementation.
    ///
    /// # Arguments
    /// * `weights` — pointer to a single 144-byte Q4_K super-block
    /// * `activations` — pointer to 256 f32 activation values (32-byte aligned preferred)
    ///
    /// # Returns
    /// The dot product of the dequantized weights and activations as f32.
    ///
    /// # Safety
    /// - Caller must verify AVX2+FMA support before calling.
    /// - `weights` must point to at least 144 valid bytes.
    /// - `activations` must point to at least 256 valid f32 values.
    pub fn yule_q4k_gemv_avx2(weights: *const u8, activations: *const f32) -> f32;
}

/// Safe wrapper for the Q4_K AVX2 assembly kernel.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available (checked by dispatch layer).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn vec_dot_q4_k_asm(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 144, "Q4_K block must be at least 144 bytes");
    debug_assert!(
        act.len() >= 256,
        "Q4_K activations must be at least 256 floats"
    );
    unsafe { yule_q4k_gemv_avx2(block.as_ptr(), act.as_ptr()) }
}
