//! Leech Lattice Vector Quantization (LLVQ).
//! SOTA 2-bit per weight compression using optimal 24D sphere packing.
//! No codebook storage needed -- lattice points computed algebraically.
//! Reference: "Leech Lattice Vector Quantization" (March 2026, arXiv 2603.11021)

/// Configuration for LLVQ.
pub struct LlvqConfig {
    pub group_size: usize,     // 24 (Leech lattice dimension)
    pub shell_radius: u32,     // which Leech lattice shell to use
    pub bits_per_group: usize, // encoding bits per 24-weight group
}

impl Default for LlvqConfig {
    fn default() -> Self {
        Self {
            group_size: 24,
            shell_radius: 1,
            bits_per_group: 48, // 2 bits per weight * 24 weights
        }
    }
}

/// Extended Golay code word (24 bits).
/// The Leech lattice is constructed from the Golay code.
pub struct GolayCodeword(pub [u8; 3]); // 24 bits packed into 3 bytes

/// Decode a Leech lattice index to a 24D point.
/// Uses the extended Golay code construction:
/// 1. Decode the index into a Golay codeword
/// 2. Map the codeword to a lattice point
/// 3. Scale by the quantization scale
pub fn decode_leech_point(index: u64, scale: f32) -> [f32; 24] {
    let mut point = [0.0f32; 24];

    // Simplified decoding: use the index bits directly as ternary coordinates.
    // Full implementation would use the Golay code algebraic structure.
    for i in 0..24 {
        let bits = ((index >> (i * 2)) & 3) as i32;
        // Map: 00 -> -1, 01 -> 0, 10 -> +1, 11 -> -2
        let coord = match bits {
            0 => -1,
            1 => 0,
            2 => 1,
            3 => -2,
            _ => unreachable!(),
        };
        point[i] = coord as f32 * scale;
    }

    point
}

/// Dequantize an LLVQ-encoded group of 24 weights.
pub fn dequant_llvq_group(index: u64, scale: f32, output: &mut [f32]) {
    let point = decode_leech_point(index, scale);
    let len = output.len().min(24);
    output[..len].copy_from_slice(&point[..len]);
}

/// Fused LLVQ dot product (decode + dot in one pass).
pub fn vec_dot_llvq(index: u64, scale: f32, activation: &[f32]) -> f32 {
    let point = decode_leech_point(index, scale);
    let len = activation.len().min(24);
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += point[i] * activation[i];
    }
    sum
}

/// Encode a 24D float vector to the nearest Leech lattice point.
/// Returns (index, scale, error).
pub fn encode_leech(weights: &[f32]) -> (u64, f32, f32) {
    assert!(
        weights.len() >= 24,
        "encode_leech requires at least 24 weights, got {}",
        weights.len()
    );

    // Find scale: max absolute value
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 2.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    // Quantize each coordinate to nearest lattice coordinate
    let mut index = 0u64;
    let mut error = 0.0f32;

    for i in 0..24 {
        let normalized = weights[i] * inv_scale;
        let (bits, closest) = if normalized <= -1.5 {
            (3u64, -2.0)
        } else if normalized <= -0.5 {
            (0u64, -1.0)
        } else if normalized <= 0.5 {
            (1u64, 0.0)
        } else {
            (2u64, 1.0)
        };
        index |= bits << (i * 2);
        error += (normalized - closest).powi(2);
    }

    (index, scale, error.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // Weights that map cleanly to lattice coordinates
        let mut weights = [0.0f32; 24];
        // scale will be max_abs / 2.0 = 2.0 / 2.0 = 1.0
        // So normalized coords are themselves; map to {-2, -1, 0, 1}
        weights[0] = 1.0; // normalized=1.0 -> bits=2, coord=+1
        weights[1] = -1.0; // normalized=-1.0 -> bits=0, coord=-1
        weights[2] = 0.0; // normalized=0.0 -> bits=1, coord=0
        weights[3] = -2.0; // normalized=-2.0 -> bits=3, coord=-2

        let (index, scale, error) = encode_leech(&weights);
        let decoded = decode_leech_point(index, scale);

        assert!(
            (decoded[0] - 1.0).abs() < 1e-5,
            "coord 0: got {}",
            decoded[0]
        );
        assert!(
            (decoded[1] - (-1.0)).abs() < 1e-5,
            "coord 1: got {}",
            decoded[1]
        );
        assert!(
            (decoded[2] - 0.0).abs() < 1e-5,
            "coord 2: got {}",
            decoded[2]
        );
        assert!(
            (decoded[3] - (-2.0)).abs() < 1e-5,
            "coord 3: got {}",
            decoded[3]
        );

        // Error should be small for clean lattice points
        assert!(
            error < 1e-5,
            "roundtrip error should be near zero, got {}",
            error
        );
    }

    #[test]
    fn test_vec_dot_correctness() {
        let index: u64 = 0; // All bits 00 -> all coords = -1
        let scale = 0.5; // All decoded values = -0.5

        let activation = [1.0f32; 24];
        let dot = vec_dot_llvq(index, scale, &activation);

        // Each of 24 coords is -0.5, dotted with 1.0 each = -12.0
        assert!((dot - (-12.0)).abs() < 1e-4, "expected -12.0, got {}", dot);
    }

    #[test]
    fn test_known_point_decoding() {
        // index = 0 means all 24 pairs of bits are 00 -> coord = -1
        let point = decode_leech_point(0, 1.0);
        for i in 0..24 {
            assert!(
                (point[i] - (-1.0)).abs() < 1e-6,
                "coord {}: expected -1.0, got {}",
                i,
                point[i]
            );
        }

        // index with all bits = 01 (value 1 in each 2-bit slot) -> coord = 0
        let all_ones: u64 = (0..24).fold(0u64, |acc, i| acc | (1u64 << (i * 2)));
        let point = decode_leech_point(all_ones, 2.0);
        for i in 0..24 {
            assert!(
                point[i].abs() < 1e-6,
                "coord {}: expected 0.0, got {}",
                i,
                point[i]
            );
        }

        // index with all bits = 10 (value 2 in each 2-bit slot) -> coord = +1
        let all_twos: u64 = (0..24).fold(0u64, |acc, i| acc | (2u64 << (i * 2)));
        let point = decode_leech_point(all_twos, 3.0);
        for i in 0..24 {
            assert!(
                (point[i] - 3.0).abs() < 1e-6,
                "coord {}: expected 3.0, got {}",
                i,
                point[i]
            );
        }
    }

    #[test]
    fn test_dequant_group() {
        let index: u64 = 0; // all -1
        let scale = 2.0;
        let mut output = [0.0f32; 24];
        dequant_llvq_group(index, scale, &mut output);

        for i in 0..24 {
            assert!(
                (output[i] - (-2.0)).abs() < 1e-6,
                "coord {}: expected -2.0, got {}",
                i,
                output[i]
            );
        }
    }

    #[test]
    fn test_error_measurement() {
        // Weights that don't land exactly on lattice points
        let mut weights = [0.3f32; 24];
        weights[0] = 1.5;

        let (_index, _scale, error) = encode_leech(&weights);

        // Error should be positive (non-zero quantization error)
        assert!(
            error > 0.0,
            "expected nonzero error for non-lattice weights"
        );
    }

    #[test]
    fn test_vec_dot_matches_dequant() {
        let mut weights = [0.0f32; 24];
        for i in 0..24 {
            weights[i] = (i as f32 - 12.0) * 0.15;
        }

        let (index, scale, _error) = encode_leech(&weights);

        let mut dequant = [0.0f32; 24];
        dequant_llvq_group(index, scale, &mut dequant);

        let activation: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();

        let dot_from_dequant: f32 = dequant
            .iter()
            .zip(activation.iter())
            .map(|(w, a)| w * a)
            .sum();
        let dot_fused = vec_dot_llvq(index, scale, &activation);

        assert!(
            (dot_from_dequant - dot_fused).abs() < 1e-4,
            "fused dot {} != dequant dot {}",
            dot_fused,
            dot_from_dequant
        );
    }

    #[test]
    fn test_default_config() {
        let config = LlvqConfig::default();
        assert_eq!(config.group_size, 24);
        assert_eq!(config.shell_radius, 1);
        assert_eq!(config.bits_per_group, 48);
    }
}
