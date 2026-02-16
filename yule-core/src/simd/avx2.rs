#![allow(clippy::needless_range_loop)]
use crate::dtype::DType;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// # Safety
/// Caller must verify AVX2+FMA support before calling.
pub unsafe fn vec_dot_dispatch(dtype: DType, block: &[u8], act: &[f32]) -> Option<f32> {
    unsafe {
        match dtype {
            DType::Q8_0 => Some(vec_dot_q8_0_avx2(block, act)),
            DType::Q4_0 => Some(vec_dot_q4_0_avx2(block, act)),
            DType::Q2_K => Some(vec_dot_q2_k_avx2(block, act)),
            DType::Q3_K => Some(vec_dot_q3_k_avx2(block, act)),
            DType::Q4_K => Some(vec_dot_q4_k_avx2(block, act)),
            DType::Q5_K => Some(vec_dot_q5_k_avx2(block, act)),
            DType::Q6_K => Some(vec_dot_q6_k_avx2(block, act)),
            _ => None,
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        let mut acc = _mm256_setzero_ps();

        for g in 0..4 {
            let off = g * 8;
            let qi = _mm_loadl_epi64(qs.as_ptr().add(off) as *const __m128i);
            let i32s = _mm256_cvtepi8_epi32(qi);
            let floats = _mm256_cvtepi32_ps(i32s);
            let a = _mm256_loadu_ps(act.as_ptr().add(off));
            acc = _mm256_fmadd_ps(floats, a, acc);
        }

        hsum_avx2(acc) * d
    }
}

/// Q4_0: interleaved nibbles. byte[i] low = weight 2i, high = weight 2i+1.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        let bias = _mm256_set1_ps(8.0);
        let mut acc = _mm256_setzero_ps();

        for g in 0..4 {
            let off = g * 4;
            let mut w = [0.0f32; 8];
            for i in 0..4 {
                let byte = qs[off + i];
                w[2 * i] = (byte & 0x0F) as f32;
                w[2 * i + 1] = (byte >> 4) as f32;
            }
            let weights = _mm256_loadu_ps(w.as_ptr());
            let weights = _mm256_sub_ps(weights, bias);
            let a = _mm256_loadu_ps(act.as_ptr().add(g * 8));
            acc = _mm256_fmadd_ps(weights, a, acc);
        }

        hsum_avx2(acc) * d
    }
}

/// Q4_K: 256-weight super-block. 8 sub-blocks of 32, grouped nibbles.
/// Accumulates scale*dot - min*act_sum per sub-block using FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_k_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = crate::dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let mut scales_arr = [0u8; 12];
        scales_arr.copy_from_slice(&block[4..16]);
        let (sc, mn) = crate::dequant::extract_q4k_scales_mins(&scales_arr);

        let mut sum = 0.0f32;

        for j in 0..4 {
            let qs = &block[16 + 32 * j..];

            // sub-block 2*j: low nibbles → act[64*j .. 64*j+32]
            let sb0 = 2 * j;
            let scale0 = d * sc[sb0] as f32;
            let min0 = dmin * mn[sb0] as f32;
            let mut dot_acc = _mm256_setzero_ps();
            let mut sum_acc = _mm256_setzero_ps();

            for g in 0..4 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for i in 0..8 {
                    w[i] = (qs[off + i] & 0x0F) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(64 * j + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
                sum_acc = _mm256_add_ps(sum_acc, a);
            }
            sum += scale0 * hsum_avx2(dot_acc) - min0 * hsum_avx2(sum_acc);

            // sub-block 2*j+1: high nibbles → act[64*j+32 .. 64*j+64]
            let sb1 = 2 * j + 1;
            let scale1 = d * sc[sb1] as f32;
            let min1 = dmin * mn[sb1] as f32;
            dot_acc = _mm256_setzero_ps();
            sum_acc = _mm256_setzero_ps();

            for g in 0..4 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for i in 0..8 {
                    w[i] = (qs[off + i] >> 4) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(64 * j + 32 + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
                sum_acc = _mm256_add_ps(sum_acc, a);
            }
            sum += scale1 * hsum_avx2(dot_acc) - min1 * hsum_avx2(sum_acc);
        }

        sum
    }
}

/// Q2_K: 256-weight super-block. 16 sub-blocks of 16, 2-bit asymmetric.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q2_k_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = crate::dequant::f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut sum = 0.0f32;

        for sb in 0..16 {
            let scale = (block[sb] & 0x0F) as f32;
            let min = ((block[sb] >> 4) & 0x0F) as f32;
            let mut dot_acc = _mm256_setzero_ps();
            let mut sum_acc = _mm256_setzero_ps();

            // 16 weights in groups of 8
            for g in 0..2 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for k in 0..8 {
                    let wi = sb * 16 + off + k;
                    let byte_idx = 16 + wi / 4;
                    let bit_shift = 2 * (wi % 4);
                    w[k] = ((block[byte_idx] >> bit_shift) & 0x03) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(sb * 16 + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
                sum_acc = _mm256_add_ps(sum_acc, a);
            }
            sum += d * scale * hsum_avx2(dot_acc) - dmin * min * hsum_avx2(sum_acc);
        }
        sum
    }
}

/// Q3_K: 256-weight super-block. 16 sub-blocks of 16, 3-bit symmetric.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q3_k_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        // extract 16 6-bit scales
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
            let sc = scales[sb] as f32;
            let mut dot_acc = _mm256_setzero_ps();

            for g in 0..2 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for k in 0..8 {
                    let wi = sb * 16 + off + k;
                    let qs_byte = block[32 + wi / 4];
                    let lo = (qs_byte >> (2 * (wi % 4))) & 0x03;
                    let hi = (block[wi / 8] >> (wi % 8)) & 0x01;
                    w[k] = ((lo | (hi << 2)) as i8 - 4) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(sb * 16 + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
            }
            sum += d * sc * hsum_avx2(dot_acc);
        }
        sum
    }
}

/// Q5_K: 256-weight super-block. 8 sub-blocks of 32, 5-bit asymmetric.
/// Same scale packing as Q4_K + 5th bit from qh.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_k_avx2(block: &[u8], act: &[f32]) -> f32 {
    unsafe {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = crate::dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let mut scales_arr = [0u8; 12];
        scales_arr.copy_from_slice(&block[4..16]);
        let (sc, mn) = crate::dequant::extract_q4k_scales_mins(&scales_arr);

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
            let mut dot_acc = _mm256_setzero_ps();
            let mut sum_acc = _mm256_setzero_ps();

            for g in 0..4 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for l in 0..8 {
                    let idx = off + l;
                    w[l] = ((ql[idx] & 0xF) + if qh[idx] & u1 != 0 { 16 } else { 0 }) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(64 * j + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
                sum_acc = _mm256_add_ps(sum_acc, a);
            }
            sum += scale0 * hsum_avx2(dot_acc) - min0 * hsum_avx2(sum_acc);

            // sub-block is+1: high nibbles + 5th bit
            let scale1 = d * sc[is + 1] as f32;
            let min1 = dmin * mn[is + 1] as f32;
            dot_acc = _mm256_setzero_ps();
            sum_acc = _mm256_setzero_ps();

            for g in 0..4 {
                let off = g * 8;
                let mut w = [0.0f32; 8];
                for l in 0..8 {
                    let idx = off + l;
                    w[l] = ((ql[idx] >> 4) + if qh[idx] & u2 != 0 { 16 } else { 0 }) as f32;
                }
                let wv = _mm256_loadu_ps(w.as_ptr());
                let a = _mm256_loadu_ps(act.as_ptr().add(64 * j + 32 + off));
                dot_acc = _mm256_fmadd_ps(wv, a, dot_acc);
                sum_acc = _mm256_add_ps(sum_acc, a);
            }
            sum += scale1 * hsum_avx2(dot_acc) - min1 * hsum_avx2(sum_acc);

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum
    }
}

/// Q6_K: 256-weight super-block. Two halves of 128, ql/qh 6-bit extraction.
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6_k_avx2(block: &[u8], act: &[f32]) -> f32 {
    // Q6_K's ql/qh interleaving makes full SIMD extraction complex.
    // Fused scalar extraction + accumulation still beats dequant+dot fallback.
    {
        let d = crate::dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let sc = &block[192..208];
        let mut sum = 0.0f32;

        for half in 0..2 {
            let ql = &block[half * 64..];
            let qh = &block[128 + half * 32..];
            let act_base = half * 128;
            let sc_base = half * 8;

            // Process in groups of 8 for each of the 4 sub-groups of 32
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn test_q8_0_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 42u64;

        for _ in 0..10_000 {
            let mut block = [0u8; 34];
            block[0] = 0x00;
            block[1] = 0x38;
            for i in 0..32 {
                block[2 + i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            let act: Vec<f32> = (0..32)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q8_0(&block, &act);
            let simd = unsafe { vec_dot_q8_0_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-2,
                "q8_0: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q4_0_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 123u64;

        for _ in 0..10_000 {
            let mut block = [0u8; 18];
            block[0] = 0x00;
            block[1] = 0x38;
            for i in 0..16 {
                block[2 + i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            let act: Vec<f32> = (0..32)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q4_0(&block, &act);
            let simd = unsafe { vec_dot_q4_0_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-2,
                "q4_0: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q4_k_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 777u64;

        for _ in 0..1_000 {
            let mut block = vec![0u8; 144];
            // d and dmin
            block[0] = 0x00;
            block[1] = 0x38;
            block[2] = 0x66;
            block[3] = 0x2E;
            for i in 4..144 {
                block[i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            // keep scales reasonable (6-bit values)
            for i in 4..16 {
                block[i] = (xorshift(&mut s) % 64) as u8;
            }
            let act: Vec<f32> = (0..256)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q4_k(&block, &act);
            let simd = unsafe { vec_dot_q4_k_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-1,
                "q4_k: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q2_k_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 314u64;

        for _ in 0..1_000 {
            let mut block = vec![0u8; 84];
            // scales at 0..16
            for i in 0..16 {
                block[i] = (xorshift(&mut s) % 256) as u8;
            }
            // qs at 16..80
            for i in 16..80 {
                block[i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            // d at 80, dmin at 82
            block[80] = 0x00;
            block[81] = 0x38; // d = 0.5
            block[82] = 0x66;
            block[83] = 0x2E; // dmin small
            let act: Vec<f32> = (0..256)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q2_k(&block, &act);
            let simd = unsafe { vec_dot_q2_k_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-1,
                "q2_k: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q3_k_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 555u64;

        for _ in 0..1_000 {
            let mut block = vec![0u8; 110];
            for i in 0..108 {
                block[i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            // d at 108
            block[108] = 0x00;
            block[109] = 0x38;
            // keep scales reasonable
            for i in 96..108 {
                block[i] = (xorshift(&mut s) % 64) as u8;
            }
            let act: Vec<f32> = (0..256)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q3_k(&block, &act);
            let simd = unsafe { vec_dot_q3_k_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-1,
                "q3_k: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q5_k_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 888u64;

        for _ in 0..1_000 {
            let mut block = vec![0u8; 176];
            // d and dmin
            block[0] = 0x00;
            block[1] = 0x38;
            block[2] = 0x66;
            block[3] = 0x2E;
            for i in 4..176 {
                block[i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            // keep scales reasonable
            for i in 4..16 {
                block[i] = (xorshift(&mut s) % 64) as u8;
            }
            let act: Vec<f32> = (0..256)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q5_k(&block, &act);
            let simd = unsafe { vec_dot_q5_k_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-1,
                "q5_k: scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn test_q6_k_avx2_vs_scalar() {
        if !has_avx2() {
            return;
        }
        let mut s = 999u64;

        for _ in 0..1_000 {
            let mut block = vec![0u8; 210];
            for i in 0..210 {
                block[i] = (xorshift(&mut s) & 0xFF) as u8;
            }
            // d at offset 208
            block[208] = 0x00;
            block[209] = 0x38;
            // scales at 192..208 as small signed values
            for i in 192..208 {
                block[i] = (xorshift(&mut s) % 20) as u8;
            }
            let act: Vec<f32> = (0..256)
                .map(|_| (xorshift(&mut s) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let scalar = crate::dequant::vec_dot_q6_k(&block, &act);
            let simd = unsafe { vec_dot_q6_k_avx2(&block, &act) };
            assert!(
                (scalar - simd).abs() < 1e-1,
                "q6_k: scalar={scalar} simd={simd}"
            );
        }
    }
}
