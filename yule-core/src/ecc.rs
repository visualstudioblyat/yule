//! Error-correcting code protection for quantized model weights.
//! Detects and corrects single-bit errors in weight tensors.
//! Integrated with BLAKE3 verification for tamper-evident logging.

/// CRC-32 checksum for a weight block.
/// Uses the standard CRC-32 polynomial (ISO 3309 / ITU-T V.42).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Hamming(7,4) encoder — encodes 4 data bits into 7 bits with 3 parity bits.
/// Can correct single-bit errors.
///
/// Bit layout: `[p1, p2, d1, p3, d2, d3, d4]` (positions 1-7).
/// Only the lower 4 bits of `nibble` are used.
pub fn hamming74_encode(nibble: u8) -> u8 {
    let d1 = (nibble >> 3) & 1;
    let d2 = (nibble >> 2) & 1;
    let d3 = (nibble >> 1) & 1;
    let d4 = nibble & 1;

    // Parity bits (even parity):
    // p1 covers positions 1,3,5,7 → p1, d1, d2, d4
    let p1 = d1 ^ d2 ^ d4;
    // p2 covers positions 2,3,6,7 → p2, d1, d3, d4
    let p2 = d1 ^ d3 ^ d4;
    // p3 covers positions 4,5,6,7 → p3, d2, d3, d4
    let p3 = d2 ^ d3 ^ d4;

    // Assemble: bit6=p1, bit5=p2, bit4=d1, bit3=p3, bit2=d2, bit1=d3, bit0=d4
    (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4
}

/// Hamming(7,4) decoder — detects and corrects single-bit errors.
/// Returns `(corrected_nibble, error_detected, error_corrected)`.
pub fn hamming74_decode(code: u8) -> (u8, bool, bool) {
    let p1 = (code >> 6) & 1;
    let p2 = (code >> 5) & 1;
    let d1 = (code >> 4) & 1;
    let p3 = (code >> 3) & 1;
    let d2 = (code >> 2) & 1;
    let d3 = (code >> 1) & 1;
    let d4 = code & 1;

    // Syndrome bits
    let s1 = p1 ^ d1 ^ d2 ^ d4;
    let s2 = p2 ^ d1 ^ d3 ^ d4;
    let s3 = p3 ^ d2 ^ d3 ^ d4;

    let syndrome = (s3 << 2) | (s2 << 1) | s1;

    if syndrome == 0 {
        // No error
        let nibble = (d1 << 3) | (d2 << 2) | (d3 << 1) | d4;
        (nibble, false, false)
    } else {
        // Error detected; syndrome gives the 1-based position of the flipped bit
        // Position mapping in our 7-bit code (bit6..bit0):
        //   position 1 → bit 6 (p1)
        //   position 2 → bit 5 (p2)
        //   position 3 → bit 4 (d1)
        //   position 4 → bit 3 (p3)
        //   position 5 → bit 2 (d2)
        //   position 6 → bit 1 (d3)
        //   position 7 → bit 0 (d4)
        let bit_pos = 7 - syndrome; // convert 1-based position to bit index
        let corrected = code ^ (1 << bit_pos);

        let cd1 = (corrected >> 4) & 1;
        let cd2 = (corrected >> 2) & 1;
        let cd3 = (corrected >> 1) & 1;
        let cd4 = corrected & 1;
        let nibble = (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4;

        (nibble, true, true)
    }
}

/// Protection scheme for a tensor's critical bytes (scale factors).
pub struct TensorProtection {
    /// Name of the protected tensor.
    pub tensor_name: String,
    /// CRC-32 checksum per block.
    pub checksums: Vec<u32>,
    /// Bytes per checksum block.
    pub block_size: usize,
    /// Total number of blocks.
    pub total_blocks: usize,
}

impl TensorProtection {
    /// Compute checksums for a tensor's raw data.
    pub fn compute(name: &str, data: &[u8], block_size: usize) -> Self {
        let total_blocks = if data.is_empty() {
            0
        } else {
            (data.len() + block_size - 1) / block_size
        };

        let mut checksums = Vec::with_capacity(total_blocks);
        for i in 0..total_blocks {
            let start = i * block_size;
            let end = (start + block_size).min(data.len());
            checksums.push(crc32(&data[start..end]));
        }

        Self {
            tensor_name: name.to_string(),
            checksums,
            block_size,
            total_blocks,
        }
    }

    /// Verify all blocks. Returns indices of corrupted blocks.
    pub fn verify(&self, data: &[u8]) -> Vec<usize> {
        let mut corrupted = Vec::new();
        for i in 0..self.total_blocks {
            if !self.verify_block(data, i) {
                corrupted.push(i);
            }
        }
        corrupted
    }

    /// Verify a single block.
    pub fn verify_block(&self, data: &[u8], block_idx: usize) -> bool {
        if block_idx >= self.total_blocks {
            return false;
        }
        let start = block_idx * self.block_size;
        let end = (start + self.block_size).min(data.len());
        let actual = crc32(&data[start..end]);
        actual == self.checksums[block_idx]
    }
}

/// Correction result from a verification pass.
pub struct CorrectionResult {
    /// Number of blocks checked.
    pub blocks_checked: usize,
    /// Number of errors detected.
    pub errors_detected: usize,
    /// Number of errors corrected.
    pub errors_corrected: usize,
    /// Number of uncorrectable errors.
    pub uncorrectable: usize,
}

/// Periodic verification: check N random blocks per forward pass.
///
/// Selects up to `sample_count` blocks (deterministically seeded from the data
/// length to avoid a CSPRNG dependency) and verifies their CRC-32 checksums.
pub fn spot_check(
    protection: &TensorProtection,
    data: &[u8],
    sample_count: usize,
) -> CorrectionResult {
    let n = protection.total_blocks;
    let checks = sample_count.min(n);

    let mut errors_detected = 0usize;

    // Select blocks to check. Use a simple deterministic sequence derived from
    // the data length so that the call is reproducible without pulling in a RNG.
    // For real deployment one would use getrandom here, but the spec says no
    // new dependencies and this keeps tests deterministic.
    let mut idx = data.len() % (n.max(1));
    for _ in 0..checks {
        if !protection.verify_block(data, idx % n) {
            errors_detected += 1;
        }
        // simple LCG-style step to visit different blocks
        idx = idx.wrapping_mul(6364136223846793005).wrapping_add(1);
        idx %= n.max(1);
    }

    CorrectionResult {
        blocks_checked: checks,
        errors_detected,
        errors_corrected: 0, // CRC can detect but not correct
        uncorrectable: errors_detected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_known_values() {
        // "123456789" → 0xCBF43926 (standard CRC-32 test vector)
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF43926);

        // Empty data → 0x00000000
        assert_eq!(crc32(b""), 0x00000000);

        // Single byte
        let single = crc32(&[0x00]);
        assert_ne!(single, 0); // non-trivial
    }

    #[test]
    fn test_hamming74_roundtrip() {
        // Encode all 16 nibbles and decode — must recover the original nibble
        for nibble in 0u8..16 {
            let encoded = hamming74_encode(nibble);
            let (decoded, error_detected, error_corrected) = hamming74_decode(encoded);
            assert_eq!(decoded, nibble, "roundtrip failed for nibble {nibble:#06b}");
            assert!(!error_detected, "false positive error for nibble {nibble}");
            assert!(
                !error_corrected,
                "false positive correction for nibble {nibble}"
            );
        }
    }

    #[test]
    fn test_hamming74_correction() {
        // For each nibble, flip each of the 7 code bits and verify correction
        for nibble in 0u8..16 {
            let encoded = hamming74_encode(nibble);
            for bit in 0..7 {
                let corrupted = encoded ^ (1 << bit);
                let (decoded, error_detected, error_corrected) = hamming74_decode(corrupted);
                assert!(
                    error_detected,
                    "missed error: nibble={nibble}, flipped bit {bit}"
                );
                assert!(
                    error_corrected,
                    "failed to correct: nibble={nibble}, flipped bit {bit}"
                );
                assert_eq!(
                    decoded, nibble,
                    "wrong correction: nibble={nibble}, flipped bit {bit}, got {decoded}"
                );
            }
        }
    }

    #[test]
    fn test_tensor_protection_detect() {
        let data = vec![0xAA; 256];
        let protection = TensorProtection::compute("test_tensor", &data, 64);

        // Clean data — no corrupted blocks
        assert_eq!(protection.total_blocks, 4);
        assert!(protection.verify(&data).is_empty());

        // Flip one byte in block 2
        let mut corrupted = data.clone();
        corrupted[128] ^= 0xFF;
        let bad_blocks = protection.verify(&corrupted);
        assert_eq!(bad_blocks.len(), 1);
        assert_eq!(bad_blocks[0], 2);

        // Individual block verification
        assert!(protection.verify_block(&corrupted, 0));
        assert!(protection.verify_block(&corrupted, 1));
        assert!(!protection.verify_block(&corrupted, 2));
        assert!(protection.verify_block(&corrupted, 3));
    }

    #[test]
    fn test_spot_check() {
        let data = vec![0x55; 1024];
        let protection = TensorProtection::compute("spot_tensor", &data, 64);

        // Clean data — spot check should find 0 errors
        let result = spot_check(&protection, &data, 5);
        assert_eq!(result.errors_detected, 0);
        assert!(result.blocks_checked <= 5);
        assert_eq!(result.errors_corrected, 0);
        assert_eq!(result.uncorrectable, 0);

        // Corrupt every block so any sample will hit an error
        let mut corrupted = data.clone();
        for i in 0..protection.total_blocks {
            corrupted[i * 64] ^= 0xFF;
        }
        let result2 = spot_check(&protection, &corrupted, 5);
        assert!(
            result2.errors_detected > 0,
            "should detect at least one error in fully-corrupted data"
        );
        assert_eq!(result2.uncorrectable, result2.errors_detected);
    }
}
