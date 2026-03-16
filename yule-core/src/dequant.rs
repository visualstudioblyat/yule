#![allow(clippy::needless_range_loop)]
//! Dequantization kernels for GGML quantization formats.
//!
//! Each kernel converts a quantized block into f32 values and/or computes
//! a fused dot product with an activation vector. The hot path for inference
//! is `vec_dot` — dequant-then-dot fused into one pass to avoid intermediate
//! memory traffic.
//!
//! Scalar implementations first. SIMD (AVX2/NEON) will be added behind
//! cfg(target_feature) gates.

use crate::dtype::DType;
use crate::error::{Result, YuleError};

// ─── f16 conversion helpers ──────────────────────────────────────────

/// Convert IEEE 754 half-precision (f16) to f32.
/// Layout: 1 sign, 5 exponent, 10 mantissa.
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // +/- zero
            f32::from_bits(sign << 31)
        } else {
            // denormalized: normalize by shifting mantissa
            let mut m = mant;
            let mut e: i32 = -14; // denorm exponent offset
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // clear leading 1
            let f32_exp = ((e + 127) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // normalized: rebias exponent from f16 bias (15) to f32 bias (127)
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Convert BF16 to f32. Just shift left by 16 bits.
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Read a little-endian u16 from a byte slice at the given offset.
#[inline(always)]
fn read_f16(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    f16_to_f32(bits)
}

// ─── IQ4_NL codebook ─────────────────────────────────────────────────

/// The 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS.
pub const IQ4_NL_CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// ─── Block dequantization functions ──────────────────────────────────
// Each function takes a raw block slice and writes `block_size` f32 values.

/// Q4_0: 32 weights, symmetric 4-bit. 18 bytes per block.
/// Layout: [d: f16 (2B)] [qs: 16B nibbles]
/// Dequant: w_i = (nibble_i - 8) * d
pub fn dequant_q4_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 18);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    for i in 0..32 {
        let byte = block[2 + i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        out[i] = (nibble as f32 - 8.0) * d;
    }
}

/// Q4_1: 32 weights, asymmetric 4-bit. 20 bytes per block.
/// Layout: [d: f16 (2B)] [m: f16 (2B)] [qs: 16B nibbles]
/// Dequant: w_i = nibble_i * d + m
pub fn dequant_q4_1(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 20);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    for i in 0..32 {
        let byte = block[4 + i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        out[i] = nibble as f32 * d + m;
    }
}

/// Q5_0: 32 weights, symmetric 5-bit. 22 bytes per block.
/// Layout: [d: f16 (2B)] [qh: 4B high bits] [qs: 16B low nibbles]
/// Dequant: w_i = (q5_i - 16) * d
pub fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 22);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    // qh is 4 bytes at offset 2 = 32 bits, one per weight
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for i in 0..32 {
        let byte = block[6 + i / 2];
        let lo = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let hi = ((qh >> i) & 1) as u8;
        let q = (hi << 4) | lo; // 5-bit value [0..31]
        out[i] = (q as f32 - 16.0) * d;
    }
}

/// Q5_1: 32 weights, asymmetric 5-bit. 24 bytes per block.
/// Layout: [d: f16 (2B)] [m: f16 (2B)] [qh: 4B] [qs: 16B]
/// Dequant: w_i = q5_i * d + m
pub fn dequant_q5_1(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 24);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for i in 0..32 {
        let byte = block[8 + i / 2];
        let lo = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let hi = ((qh >> i) & 1) as u8;
        let q = (hi << 4) | lo;
        out[i] = q as f32 * d + m;
    }
}

/// Q8_0: 32 weights, symmetric 8-bit. 34 bytes per block.
/// Layout: [d: f16 (2B)] [qs: 32B int8]
/// Dequant: w_i = qs_i * d
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 34);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    for i in 0..32 {
        let q = block[2 + i] as i8;
        out[i] = q as f32 * d;
    }
}

/// Q8_1: 32 weights, 8-bit with precomputed sum. 36 bytes per block.
/// Layout: [d: f16 (2B)] [s: f16 (2B)] [qs: 32B int8]
/// s = d * sum(qs[i]), used for fast dot products with asymmetric weight types.
/// Dequant: w_i = qs_i * d
pub fn dequant_q8_1(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 36);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    // s at offset 2 is the precomputed sum, not needed for dequant
    for i in 0..32 {
        let q = block[4 + i] as i8;
        out[i] = q as f32 * d;
    }
}

// ─── K-Quant super-block dequantization (QK_K = 256) ────────────────

/// Extract 6-bit scales and mins from Q4_K/Q5_K scales[12] array.
/// Returns (scales[8], mins[8]).
///
/// Matches ggml's get_scale_min_k4():
///   j < 4: scale = q[j] & 63,  min = q[j+4] & 63
///   j >= 4: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///            min   = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)
pub fn extract_q4k_scales_mins(q: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    for j in 0..4 {
        sc[j] = q[j] & 63;
        mn[j] = q[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }

    (sc, mn)
}

/// Q2_K: 256 weights, 2-bit asymmetric. 84 bytes per super-block.
/// Layout: [scales: 16B] [qs: 64B 2-bit packed] [d: f16 (2B)] [dmin: f16 (2B)]
/// scales[i] packs 4-bit scale (lo nibble) + 4-bit min (hi nibble) for sub-block i.
/// 16 sub-blocks of 16 weights each.
pub fn dequant_q2_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 84);
    debug_assert!(out.len() >= 256);
    let d = read_f16(block, 80);
    let dmin = read_f16(block, 82);

    for sb in 0..16 {
        let scale = (block[sb] & 0x0F) as f32;
        let min = ((block[sb] >> 4) & 0x0F) as f32;

        for k in 0..16 {
            let wi = sb * 16 + k;
            let byte_idx = 16 + wi / 4;
            let bit_shift = 2 * (wi % 4);
            let q = ((block[byte_idx] >> bit_shift) & 0x03) as f32;
            out[wi] = d * scale * q - dmin * min;
        }
    }
}

/// Q3_K: 256 weights, 3-bit symmetric. 110 bytes per super-block.
/// Layout: [hmask: 32B] [qs: 64B 2-bit lo] [scales: 12B 6-bit packed] [d: f16 (2B)]
/// 16 sub-blocks of 16 weights each.
pub fn dequant_q3_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 110);
    debug_assert!(out.len() >= 256);
    let d = read_f16(block, 108);

    // Extract 16 6-bit scales from scales[12] at offset 96
    let mut scales = [0i8; 16];
    for j in 0..16 {
        let lo = if j < 8 {
            block[96 + j] & 0x0F
        } else {
            (block[96 + j - 8] >> 4) & 0x0F
        };
        let hi_byte = 96 + 8 + j / 4;
        let hi_shift = 2 * (j % 4);
        let hi = (block[hi_byte] >> hi_shift) & 0x03;
        let raw = lo | (hi << 4); // 6-bit unsigned
        scales[j] = raw as i8 - 32; // Q3_K scales are centered at 32
    }

    for sb in 0..16 {
        let sc = scales[sb] as f32;
        for k in 0..16 {
            let wi = sb * 16 + k;
            // Low 2 bits from qs[64] at offset 32
            let qs_byte = block[32 + wi / 4];
            let lo = (qs_byte >> (2 * (wi % 4))) & 0x03;
            // High bit from hmask[32] at offset 0
            let hi = (block[wi / 8] >> (wi % 8)) & 0x01;
            let q = (lo | (hi << 2)) as i8 - 4; // 3-bit signed, centered at 4
            out[wi] = d * sc * q as f32;
        }
    }
}

/// Q4_K: 256 weights, 4-bit asymmetric. 144 bytes per super-block.
/// Layout: [d: f16 (2B)] [dmin: f16 (2B)] [scales: 12B] [qs: 128B nibbles]
/// 8 sub-blocks of 32 weights each, 6-bit scales and mins.
///
/// Nibble layout (matches ggml): each group of 64 weights uses 32 bytes.
/// Low nibble = first 32 weights, high nibble = next 32 weights.
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 144);
    debug_assert!(out.len() >= 256);
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);

    let mut scales_arr = [0u8; 12];
    scales_arr.copy_from_slice(&block[4..16]);
    let (sc, mn) = extract_q4k_scales_mins(&scales_arr);

    // 4 groups of 64 weights, each group = 2 sub-blocks of 32
    for j in 0..4 {
        let qs = &block[16 + 32 * j..];

        // sub-block 2*j: low nibbles
        let sb0 = 2 * j;
        let scale0 = d * sc[sb0] as f32;
        let min0 = dmin * mn[sb0] as f32;
        for l in 0..32 {
            out[64 * j + l] = (qs[l] & 0x0F) as f32 * scale0 - min0;
        }

        // sub-block 2*j+1: high nibbles
        let sb1 = 2 * j + 1;
        let scale1 = d * sc[sb1] as f32;
        let min1 = dmin * mn[sb1] as f32;
        for l in 0..32 {
            out[64 * j + 32 + l] = (qs[l] >> 4) as f32 * scale1 - min1;
        }
    }
}

/// Q5_K: 256 weights, 5-bit asymmetric. 176 bytes per super-block.
/// Layout: [d: f16 (2B)] [dmin: f16 (2B)] [scales: 12B] [qh: 32B] [qs: 128B]
/// Same scale packing as Q4_K. Grouped nibble layout like Q4_K:
/// groups of 64 weights, low nibble = first 32, high nibble = next 32.
/// 5th bit from qh with shifting mask (u1, u2 shift left by 2 per group).
pub fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 176);
    debug_assert!(out.len() >= 256);
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);

    let mut scales_arr = [0u8; 12];
    scales_arr.copy_from_slice(&block[4..16]);
    let (sc, mn) = extract_q4k_scales_mins(&scales_arr);

    let qh = &block[16..48]; // 32 bytes of high bits
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in 0..4 {
        let ql = &block[48 + 32 * j..]; // 32 bytes of qs per group

        // sub-block is+0: low nibbles
        let d1 = d * sc[is] as f32;
        let m1 = dmin * mn[is] as f32;
        for l in 0..32 {
            let q = (ql[l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
            out[64 * j + l] = q as f32 * d1 - m1;
        }

        // sub-block is+1: high nibbles
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * mn[is + 1] as f32;
        for l in 0..32 {
            let q = (ql[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            out[64 * j + 32 + l] = q as f32 * d2 - m2;
        }

        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Q6_K: 256 weights, 6-bit symmetric. 210 bytes per super-block.
/// Layout: [ql: 128B] [qh: 64B] [scales: 16B int8] [d: f16 (2B)]
///
/// Matches ggml layout: processes 128 weights per half.
/// Each group of 32 ql bytes + 32 qh bytes produces 128 6-bit weights:
///   weights[0..31]:  (ql[l]    & 0xF) | ((qh[l] >> 0) & 3) << 4
///   weights[32..63]: (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
///   weights[64..95]: (ql[l]    >> 4)  | ((qh[l] >> 4) & 3) << 4
///   weights[96..127]:(ql[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 210);
    debug_assert!(out.len() >= 256);
    let d = read_f16(block, 208);
    let sc = &block[192..208]; // 16 int8 scales

    // two halves of 128 weights each
    for half in 0..2 {
        let ql = &block[half * 64..]; // 64 bytes of ql per half
        let qh = &block[128 + half * 32..]; // 32 bytes of qh per half
        let out_base = half * 128;
        let sc_base = half * 8;

        for l in 0..32 {
            let is = l / 16; // ggml: is = l/16
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

            out[out_base + l] = d * sc[sc_base + is] as i8 as f32 * q1 as f32;
            out[out_base + l + 32] = d * sc[sc_base + is + 2] as i8 as f32 * q2 as f32;
            out[out_base + l + 64] = d * sc[sc_base + is + 4] as i8 as f32 * q3 as f32;
            out[out_base + l + 96] = d * sc[sc_base + is + 6] as i8 as f32 * q4 as f32;
        }
    }
}

/// Q8_K: 256 weights, 8-bit symmetric super-block. 292 bytes.
/// Layout: [d: f32 (4B)] [qs: 256B int8] [bsums: 32B int16]
pub fn dequant_q8_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 292);
    debug_assert!(out.len() >= 256);
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    for i in 0..256 {
        let q = block[4 + i] as i8;
        out[i] = q as f32 * d;
    }
}

/// IQ4_NL: 32 weights, non-linear 4-bit. 18 bytes per block.
/// Layout: [d: f16 (2B)] [qs: 16B nibbles]
/// Dequant: w_i = codebook[nibble_i] * d
pub fn dequant_iq4_nl(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 18);
    debug_assert!(out.len() >= 32);
    let d = read_f16(block, 0);
    for i in 0..32 {
        let byte = block[2 + i / 2];
        let nibble = if i % 2 == 0 {
            (byte & 0x0F) as usize
        } else {
            (byte >> 4) as usize
        };
        out[i] = IQ4_NL_CODEBOOK[nibble] as f32 * d;
    }
}

// ─── Prefetch intrinsic for dequant kernels ─────────────────────────
// Prefetches the next weight block into L1 cache while the current block
// is being processed. On non-x86 architectures this is a no-op.

/// Prefetch `data[offset..]` into L1 cache (T0 temporal hint).
/// Used inside `vec_dot_*` loops to overlap memory latency with compute.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn prefetch_block(data: &[u8], offset: usize) {
    if offset < data.len() {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        unsafe {
            let ptr = data.as_ptr().add(offset) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
            // Second cache line for blocks >64B (Q4_K=144B, Q6_K=210B)
            if offset + 64 < data.len() {
                _mm_prefetch(ptr.add(64), _MM_HINT_T0);
            }
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn prefetch_block(_data: &[u8], _offset: usize) {}

// ─── Fused dot-product kernels (scalar) ──────────────────────────────
// These compute dot(weights_block, activations) without materializing
// dequantized weights. This is the hot path for GEMV.

/// Q4_0 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q4_0(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 18);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let mut sum = 0.0f32;
    for i in 0..32 {
        let byte = block[2 + i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        sum += (nibble as f32 - 8.0) * act[i];
    }
    sum * d
}

/// Q8_0 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q8_0(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 34);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let mut sum = 0.0f32;
    for i in 0..32 {
        let q = block[2 + i] as i8;
        sum += q as f32 * act[i];
    }
    sum * d
}

/// Q4_K fused dot product: dot(super-block of 256, act[0..256]) → f32
pub fn vec_dot_q4_k(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 144);
    debug_assert!(act.len() >= 256);
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);

    let mut scales_arr = [0u8; 12];
    scales_arr.copy_from_slice(&block[4..16]);
    let (sc, mn) = extract_q4k_scales_mins(&scales_arr);

    let mut sum = 0.0f32;
    for j in 0..4 {
        // Prefetch the next group's qs data while computing this group
        let next_offset = 16 + 32 * (j + 1);
        prefetch_block(block, next_offset);

        let qs = &block[16 + 32 * j..];

        // sub-block 2*j: low nibbles, weights at act[64*j .. 64*j+32]
        let sb0 = 2 * j;
        let scale0 = d * sc[sb0] as f32;
        let min0 = dmin * mn[sb0] as f32;
        let mut sb_sum = 0.0f32;
        let mut min_sum = 0.0f32;
        for l in 0..32 {
            sb_sum += (qs[l] & 0x0F) as f32 * act[64 * j + l];
            min_sum += act[64 * j + l];
        }
        sum += scale0 * sb_sum - min0 * min_sum;

        // sub-block 2*j+1: high nibbles, weights at act[64*j+32 .. 64*j+64]
        let sb1 = 2 * j + 1;
        let scale1 = d * sc[sb1] as f32;
        let min1 = dmin * mn[sb1] as f32;
        sb_sum = 0.0;
        min_sum = 0.0;
        for l in 0..32 {
            sb_sum += (qs[l] >> 4) as f32 * act[64 * j + 32 + l];
            min_sum += act[64 * j + 32 + l];
        }
        sum += scale1 * sb_sum - min1 * min_sum;
    }
    sum
}

/// Q6_K fused dot product: dot(super-block of 256, act[0..256]) → f32
/// Matches dequant_q6_k layout: two halves of 128, ql/qh extraction, is=l/16 scales.
pub fn vec_dot_q6_k(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 210);
    debug_assert!(act.len() >= 256);
    let d = read_f16(block, 208);
    let sc = &block[192..208];
    let mut sum = 0.0f32;

    for half in 0..2 {
        // Prefetch the second half's ql and qh data while computing the first
        if half == 0 {
            prefetch_block(block, 64);   // next half's ql
            prefetch_block(block, 160);  // next half's qh (128 + 32)
        }

        let ql = &block[half * 64..];
        let qh = &block[128 + half * 32..];
        let act_base = half * 128;
        let sc_base = half * 8;

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

            sum += d * sc[sc_base + is] as i8 as f32 * q1 as f32 * act[act_base + l];
            sum += d * sc[sc_base + is + 2] as i8 as f32 * q2 as f32 * act[act_base + l + 32];
            sum += d * sc[sc_base + is + 4] as i8 as f32 * q3 as f32 * act[act_base + l + 64];
            sum += d * sc[sc_base + is + 6] as i8 as f32 * q4 as f32 * act[act_base + l + 96];
        }
    }
    sum
}

/// Q2_K fused dot product: dot(super-block of 256, act[0..256]) → f32
/// 16 sub-blocks of 16 weights, 2-bit asymmetric with 4-bit scale/min per sub-block.
pub fn vec_dot_q2_k(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 84);
    debug_assert!(act.len() >= 256);
    let d = read_f16(block, 80);
    let dmin = read_f16(block, 82);
    let mut sum = 0.0f32;

    for sb in 0..16 {
        // Prefetch next sub-block's qs bytes
        let next_qs_offset = 16 + (sb + 1) * 16 / 4;
        prefetch_block(block, next_qs_offset);

        let scale = (block[sb] & 0x0F) as f32;
        let min = ((block[sb] >> 4) & 0x0F) as f32;
        let mut dot = 0.0f32;
        let mut act_sum = 0.0f32;

        for k in 0..16 {
            let wi = sb * 16 + k;
            let byte_idx = 16 + wi / 4;
            let bit_shift = 2 * (wi % 4);
            let q = ((block[byte_idx] >> bit_shift) & 0x03) as f32;
            dot += q * act[wi];
            act_sum += act[wi];
        }
        sum += d * scale * dot - dmin * min * act_sum;
    }
    sum
}

/// Q3_K fused dot product: dot(super-block of 256, act[0..256]) → f32
/// 16 sub-blocks of 16 weights, 3-bit symmetric with 6-bit signed scales.
pub fn vec_dot_q3_k(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 110);
    debug_assert!(act.len() >= 256);
    let d = read_f16(block, 108);

    // extract 16 6-bit scales (same as dequant_q3_k)
    let mut scales = [0i8; 16];
    for j in 0..16 {
        let lo = if j < 8 {
            block[96 + j] & 0x0F
        } else {
            (block[96 + j - 8] >> 4) & 0x0F
        };
        let hi_byte = 96 + 8 + j / 4;
        let hi_shift = 2 * (j % 4);
        let hi = (block[hi_byte] >> hi_shift) & 0x03;
        let raw = lo | (hi << 4);
        scales[j] = raw as i8 - 32;
    }

    let mut sum = 0.0f32;
    for sb in 0..16 {
        // Prefetch next sub-block's qs and hmask bytes
        let next_qs_offset = 32 + (sb + 1) * 16 / 4;
        prefetch_block(block, next_qs_offset);

        let sc = scales[sb] as f32;
        let mut dot = 0.0f32;
        for k in 0..16 {
            let wi = sb * 16 + k;
            let qs_byte = block[32 + wi / 4];
            let lo = (qs_byte >> (2 * (wi % 4))) & 0x03;
            let hi = (block[wi / 8] >> (wi % 8)) & 0x01;
            let q = (lo | (hi << 2)) as i8 - 4;
            dot += q as f32 * act[wi];
        }
        sum += d * sc * dot;
    }
    sum
}

/// Q5_K fused dot product: dot(super-block of 256, act[0..256]) → f32
/// Same layout as Q4_K + 5th bit from qh. 8 sub-blocks of 32 weights.
pub fn vec_dot_q5_k(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 176);
    debug_assert!(act.len() >= 256);
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);

    let mut scales_arr = [0u8; 12];
    scales_arr.copy_from_slice(&block[4..16]);
    let (sc, mn) = extract_q4k_scales_mins(&scales_arr);

    let qh = &block[16..48];
    let mut sum = 0.0f32;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in 0..4 {
        let ql = &block[48 + 32 * j..];

        // sub-block is: low nibbles + 5th bit
        let scale0 = d * sc[is] as f32;
        let min0 = dmin * mn[is] as f32;
        let mut dot0 = 0.0f32;
        let mut asum0 = 0.0f32;
        for l in 0..32 {
            let q = (ql[l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
            dot0 += q as f32 * act[64 * j + l];
            asum0 += act[64 * j + l];
        }
        sum += scale0 * dot0 - min0 * asum0;

        // sub-block is+1: high nibbles + 5th bit
        let scale1 = d * sc[is + 1] as f32;
        let min1 = dmin * mn[is + 1] as f32;
        let mut dot1 = 0.0f32;
        let mut asum1 = 0.0f32;
        for l in 0..32 {
            let q = (ql[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            dot1 += q as f32 * act[64 * j + 32 + l];
            asum1 += act[64 * j + 32 + l];
        }
        sum += scale1 * dot1 - min1 * asum1;

        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    sum
}

/// Q4_1 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q4_1(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 20);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let mut sum = 0.0f32;
    let mut act_sum = 0.0f32;
    for i in 0..32 {
        let byte = block[4 + i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        sum += nibble as f32 * act[i];
        act_sum += act[i];
    }
    d * sum + m * act_sum
}

/// Q5_0 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q5_0(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 22);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let mut sum = 0.0f32;
    for i in 0..32 {
        let byte = block[6 + i / 2];
        let lo = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let hi = ((qh >> i) & 1) as u8;
        let q = (hi << 4) | lo;
        sum += (q as f32 - 16.0) * act[i];
    }
    sum * d
}

/// Q5_1 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q5_1(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 24);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let mut sum = 0.0f32;
    let mut act_sum = 0.0f32;
    for i in 0..32 {
        let byte = block[8 + i / 2];
        let lo = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let hi = ((qh >> i) & 1) as u8;
        let q = (hi << 4) | lo;
        sum += q as f32 * act[i];
        act_sum += act[i];
    }
    d * sum + m * act_sum
}

/// Q8_1 fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_q8_1(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 36);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let mut sum = 0.0f32;
    for i in 0..32 {
        let q = block[4 + i] as i8;
        sum += q as f32 * act[i];
    }
    sum * d
}

/// IQ4_NL fused dot product: dot(block, act[0..32]) → f32
pub fn vec_dot_iq4_nl(block: &[u8], act: &[f32]) -> f32 {
    debug_assert!(block.len() >= 18);
    debug_assert!(act.len() >= 32);
    let d = read_f16(block, 0);
    let mut sum = 0.0f32;
    for i in 0..32 {
        let byte = block[2 + i / 2];
        let nibble = if i % 2 == 0 {
            (byte & 0x0F) as usize
        } else {
            (byte >> 4) as usize
        };
        sum += IQ4_NL_CODEBOOK[nibble] as f32 * act[i];
    }
    sum * d
}

// ─── Dispatch by DType ───────────────────────────────────────────────

/// Dequantize one block of quantized data into f32 values.
/// `block` must be exactly `dtype.size_of_block()` bytes.
/// `out` must have at least `dtype.block_size()` elements.
pub fn dequant_block(dtype: DType, block: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        DType::Q4_0 => dequant_q4_0(block, out),
        DType::Q4_1 => dequant_q4_1(block, out),
        DType::Q5_0 => dequant_q5_0(block, out),
        DType::Q5_1 => dequant_q5_1(block, out),
        DType::Q8_0 => dequant_q8_0(block, out),
        DType::Q8_1 => dequant_q8_1(block, out),
        DType::Q2_K => dequant_q2_k(block, out),
        DType::Q3_K => dequant_q3_k(block, out),
        DType::Q4_K => dequant_q4_k(block, out),
        DType::Q5_K => dequant_q5_k(block, out),
        DType::Q6_K => dequant_q6_k(block, out),
        DType::Q8_K => dequant_q8_k(block, out),
        DType::IQ4_NL => dequant_iq4_nl(block, out),
        DType::F16 => {
            out[0] = read_f16(block, 0);
        }
        DType::BF16 => {
            let bits = u16::from_le_bytes([block[0], block[1]]);
            out[0] = bf16_to_f32(bits);
        }
        DType::F32 => {
            out[0] = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        }
        DType::IQ1_S
        | DType::IQ1_M
        | DType::IQ2_XXS
        | DType::IQ2_XS
        | DType::IQ2_S
        | DType::IQ3_XXS
        | DType::IQ3_S
        | DType::IQ4_XS => {
            return Err(YuleError::Inference(format!(
                "I-quant dequant for {:?} requires codebook tables (not yet implemented)",
                dtype
            )));
        }
        DType::TQ1_0 | DType::TQ2_0 => {
            return Err(YuleError::Inference(format!(
                "ternary dequant for {:?} not yet implemented",
                dtype
            )));
        }
        _ => {
            return Err(YuleError::Inference(format!(
                "dequant not yet implemented for {:?}",
                dtype
            )));
        }
    }
    Ok(())
}

/// Fused dot product: compute dot(quantized_block, activations).
/// Routes through SIMD dispatch (AVX2 when available, scalar fallback).
pub fn vec_dot_block(dtype: DType, block: &[u8], act: &[f32]) -> Result<f32> {
    crate::simd::vec_dot(dtype, block, act)
        .ok_or_else(|| YuleError::Inference(format!("vec_dot not implemented for {:?}", dtype)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 1.0 in f16 = 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -1.0 in f16 = 0xBC00
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0.5 in f16 = 0x3800
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }

    #[test]
    fn test_bf16_conversion() {
        // 1.0 in bf16 = 0x3F80
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        // -1.0 in bf16 = 0xBF80
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn test_q4_0_dequant_roundtrip() {
        // Build a Q4_0 block where d=1.0 and all nibbles are 8 (= zero weight)
        let mut block = [0u8; 18];
        // d = 1.0 in f16 = 0x3C00 LE = [0x00, 0x3C]
        block[0] = 0x00;
        block[1] = 0x3C;
        // All nibbles = 8: byte = 0x88
        for i in 0..16 {
            block[2 + i] = 0x88;
        }

        let mut out = [0.0f32; 32];
        dequant_q4_0(&block, &mut out);

        // (8 - 8) * 1.0 = 0.0
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_q4_0_dequant_nonzero() {
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // First byte: low nibble = 0, high nibble = 15
        block[2] = 0xF0;

        let mut out = [0.0f32; 32];
        dequant_q4_0(&block, &mut out);

        // weight 0: (0 - 8) * 1.0 = -8.0
        assert!((out[0] - (-8.0)).abs() < 1e-6);
        // weight 1: (15 - 8) * 1.0 = 7.0
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_q8_0_dequant() {
        let mut block = [0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 127; // max positive int8
        block[3] = 0x80u8; // -128 as i8

        let mut out = [0.0f32; 32];
        dequant_q8_0(&block, &mut out);

        assert!((out[0] - 127.0).abs() < 1e-6);
        assert!((out[1] - (-128.0)).abs() < 1e-6);
    }

    #[test]
    fn test_q4_0_vec_dot() {
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // All nibbles = 9 → weight = (9-8)*1.0 = 1.0
        for i in 0..16 {
            block[2 + i] = 0x99;
        }

        let act = [1.0f32; 32];
        let dot = vec_dot_q4_0(&block, &act);
        // 32 * 1.0 * 1.0 = 32.0
        assert!((dot - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_iq4_nl_dequant() {
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // First byte: low nibble = 0 → codebook[-127], high = 15 → codebook[113]
        block[2] = 0xF0;

        let mut out = [0.0f32; 32];
        dequant_iq4_nl(&block, &mut out);

        assert!((out[0] - (-127.0)).abs() < 1e-6);
        assert!((out[1] - 113.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_dispatch() {
        let mut block = [0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 42;

        let mut out = [0.0f32; 32];
        dequant_block(DType::Q8_0, &block, &mut out).unwrap();
        assert!((out[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_q4_1_dequant() {
        // Q4_1: w = d * nibble + m
        let mut block = [0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x00;
        block[3] = 0x3C; // m = 1.0
        // First byte: low nibble = 0, high nibble = 15
        block[4] = 0xF0;
        // Rest zeros

        let mut out = [0.0f32; 32];
        dequant_q4_1(&block, &mut out);

        // weight 0: 0 * 1.0 + 1.0 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-6);
        // weight 1: 15 * 1.0 + 1.0 = 16.0
        assert!((out[1] - 16.0).abs() < 1e-6);
        // weight 2: 0 * 1.0 + 1.0 = 1.0 (rest are zero nibbles)
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_q4_1_vec_dot() {
        let mut block = [0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x00;
        block[3] = 0x3C; // m = 1.0
        // All nibbles = 0 → weight = 0*1 + 1 = 1.0
        // (block[4..20] already zeros)

        let act = [1.0f32; 32];
        let dot = vec_dot_q4_1(&block, &act);
        // 32 * 1.0 * 1.0 = 32.0
        assert!((dot - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_q5_0_dequant() {
        // Q5_0: w = (q5 - 16) * d
        let mut block = [0u8; 22];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // qh at offset 2..6: bit 0 = 1, rest = 0
        block[2] = 0x01;
        block[3] = 0x00;
        block[4] = 0x00;
        block[5] = 0x00;
        // qs at offset 6: first byte low nibble = 0, high nibble = 0
        block[6] = 0x00;

        let mut out = [0.0f32; 32];
        dequant_q5_0(&block, &mut out);

        // weight 0: lo=0, hi=1 (qh bit 0 set) → q5 = 16 → (16 - 16) * 1.0 = 0.0
        assert!((out[0] - 0.0).abs() < 1e-6);
        // weight 1: lo=0, hi=0 (qh bit 1 not set) → q5 = 0 → (0 - 16) * 1.0 = -16.0
        assert!((out[1] - (-16.0)).abs() < 1e-6);

        // Test with non-zero nibble: set qs byte at index 6 to 0x5A
        block[6] = 0x5A;
        // Also set qh bit 0 = 0 for this test
        block[2] = 0x02; // bit 1 set, bit 0 clear
        dequant_q5_0(&block, &mut out);
        // weight 0: lo=0xA=10, hi=0 → q5=10 → (10-16)*1 = -6.0
        assert!((out[0] - (-6.0)).abs() < 1e-6);
        // weight 1: lo=0x5=5, hi=1 → q5=16+5=21 → (21-16)*1 = 5.0
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_q5_0_vec_dot() {
        let mut block = [0u8; 22];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // qh = 0 (all high bits zero)
        // qs: all nibbles = 0 → q5 = 0 → weight = -16.0
        let act = [1.0f32; 32];
        let dot = vec_dot_q5_0(&block, &act);
        // 32 * (-16.0) * 1.0 = -512.0
        assert!((dot - (-512.0)).abs() < 1e-4);
    }

    #[test]
    fn test_q3_k_dequant() {
        // Q3_K: 110 bytes, 256 weights
        // Layout: hmask[0..32] + qs[32..96] + scales[96..108] + d[108..110]
        let mut block = [0u8; 110];
        // d = 1.0 at offset 108
        block[108] = 0x00;
        block[109] = 0x3C;

        // Set all scales to raw value 32 → signed = 32 - 32 = 0
        // With scale = 0, all weights should be 0 regardless of q values
        // scales[12] at offset 96..108: leave as zeros
        // raw = (lo | (hi << 4)), and with all zeros: raw = 0, scale = 0 - 32 = -32
        // Actually let's set a known scale: raw = 33 → scale = 1
        // For j=0: lo = block[96] & 0x0F, hi = (block[104] >> 0) & 0x03
        // Set lo = 1 (block[96] & 0x0F = 1), hi = 2 → raw = 1 | (2<<4) = 33 → scale = 1
        block[96] = 0x01; // lo nibble for scale[0] = 1
        block[104] = 0x02; // hi bits for scale[0] = 2 (at shift 0)

        // Set first weight (wi=0): qs 2 low bits = 1, hmask high bit = 0
        // qs at offset 32 + 0/4 = 32, shift = 0 → set (block[32] >> 0) & 3 = 1
        block[32] = 0x01;
        // hmask bit 0 = 0 (already zero)
        // q3 = (1 | (0 << 2)) - 4 = -3
        // weight 0 = 1.0 * 1.0 * (-3.0) = -3.0

        let mut out = [0.0f32; 256];
        dequant_q3_k(&block, &mut out);
        assert!(
            (out[0] - (-3.0)).abs() < 1e-6,
            "expected -3.0, got {}",
            out[0]
        );

        // Set hmask bit 0 to 1: q3 = (1 | (1<<2)) - 4 = 5 - 4 = 1
        block[0] = 0x01; // hmask byte 0, bit 0
        dequant_q3_k(&block, &mut out);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn test_q5_k_dequant() {
        // Q5_K: 176 bytes, 256 weights
        // Layout: d[0..2] + dmin[2..4] + scales[4..16] + qh[16..48] + qs[48..176]
        let mut block = [0u8; 176];
        // d = 1.0
        block[0] = 0x00;
        block[1] = 0x3C;
        // dmin = 0.0 (simplify test)
        block[2] = 0x00;
        block[3] = 0x00;

        // scales: set sc[0] = 1 via extract_q4k_scales_mins
        // For j=0: sc[0] = q[0] & 63 → set block[4] = 1
        block[4] = 0x01;

        // qs at offset 48: first byte low nibble = 5
        block[48] = 0x05;
        // qh at offset 16: bit for weight 0 = 0 (u1 starts at 1, qh[0] & 1)
        // Leave qh = 0 → 5th bit = 0

        // weight 0: q = (5 & 0xF) + 0 = 5, w = 1.0 * 1.0 * 5 - 0 = 5.0
        let mut out = [0.0f32; 256];
        dequant_q5_k(&block, &mut out);
        assert!(
            (out[0] - 5.0).abs() < 1e-6,
            "expected 5.0, got {}",
            out[0]
        );

        // Set qh bit 0: q = 5 + 16 = 21
        block[16] = 0x01;
        dequant_q5_k(&block, &mut out);
        assert!(
            (out[0] - 21.0).abs() < 1e-6,
            "expected 21.0, got {}",
            out[0]
        );
    }

    #[test]
    fn test_iq4_nl_dequant_all_entries() {
        // Verify the full codebook mapping
        let mut block = [0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0

        // Set nibbles to cover indices 0..15 in sequence
        for i in 0..8 {
            let lo = (2 * i) as u8;
            let hi = (2 * i + 1) as u8;
            block[2 + i] = lo | (hi << 4);
        }

        let mut out = [0.0f32; 32];
        dequant_iq4_nl(&block, &mut out);

        let expected: [f32; 16] = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0,
            53.0, 69.0, 89.0, 113.0,
        ];
        for i in 0..16 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "index {}: expected {}, got {}",
                i,
                expected[i],
                out[i]
            );
        }
    }

    #[test]
    fn test_q8_1_dequant() {
        let mut block = [0u8; 36];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        // s at offset 2..4 (ignored for dequant)
        block[4] = 42; // qs[0] = 42 as i8
        block[5] = 0x80u8; // qs[1] = -128 as i8

        let mut out = [0.0f32; 32];
        dequant_q8_1(&block, &mut out);

        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[1] - (-128.0)).abs() < 1e-6);
    }

    #[test]
    fn test_iquant_errors() {
        let block = [0u8; 256];
        let mut out = [0.0f32; 256];
        assert!(dequant_block(DType::IQ2_XXS, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ2_XS, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ2_S, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ3_XXS, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ3_S, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ4_XS, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ1_S, &block, &mut out).is_err());
        assert!(dequant_block(DType::IQ1_M, &block, &mut out).is_err());
        assert!(dequant_block(DType::TQ1_0, &block, &mut out).is_err());
        assert!(dequant_block(DType::TQ2_0, &block, &mut out).is_err());
    }
}
