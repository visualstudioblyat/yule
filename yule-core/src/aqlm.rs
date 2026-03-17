//! AQLM: Additive Quantization for Language Models.
//! Groups of 8 weights are encoded as indices into M codebooks.
//! Inference = codebook lookup + accumulation (no multiply).
//! Reference: "Extreme Compression of Large Language Models" (ICML 2024)

/// A single codebook with 256 entries of `entry_dim` floats each.
pub struct Codebook {
    pub entries: Vec<f32>, // [256 * entry_dim] flattened
    pub entry_dim: usize,  // typically 8 (group size)
}

impl Codebook {
    pub fn new(entries: Vec<f32>, entry_dim: usize) -> Self {
        assert!(
            entries.len() == 256 * entry_dim,
            "codebook must have exactly 256 * entry_dim entries, got {} (expected {})",
            entries.len(),
            256 * entry_dim
        );
        Self { entries, entry_dim }
    }

    /// Look up an entry by index.
    pub fn lookup(&self, index: u8) -> &[f32] {
        let offset = index as usize * self.entry_dim;
        &self.entries[offset..offset + self.entry_dim]
    }
}

/// AQLM configuration for a weight tensor.
pub struct AqlmConfig {
    pub num_codebooks: usize,  // M (typically 1 or 2)
    pub group_size: usize,     // typically 8
    pub bits_per_index: usize, // typically 8 (256-entry codebook)
}

/// Dequantize an AQLM-encoded weight group.
/// Each group of `group_size` weights = sum of codebook lookups.
pub fn dequant_aqlm_group(
    codebooks: &[Codebook],
    indices: &[u8], // one index per codebook
    scale: f32,
    output: &mut [f32],
) {
    let dim = output.len();
    output.fill(0.0);
    for (cb_idx, &index) in indices.iter().enumerate() {
        let entry = codebooks[cb_idx].lookup(index);
        for d in 0..dim.min(entry.len()) {
            output[d] += entry[d];
        }
    }
    for v in output.iter_mut() {
        *v *= scale;
    }
}

/// Fused AQLM dot product: codebook_lookup . activation (no intermediate dequant).
pub fn vec_dot_aqlm(codebooks: &[Codebook], indices: &[u8], scale: f32, activation: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let group_size = activation.len().min(codebooks[0].entry_dim);

    for (cb_idx, &index) in indices.iter().enumerate() {
        let entry = codebooks[cb_idx].lookup(index);
        for d in 0..group_size {
            sum += entry[d] * activation[d];
        }
    }
    sum * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_codebook(entry_dim: usize) -> Codebook {
        // Fill codebook: entry[i] has all values = (i + 1) as f32
        let mut entries = vec![0.0f32; 256 * entry_dim];
        for i in 0..256 {
            for d in 0..entry_dim {
                entries[i * entry_dim + d] = (i + 1) as f32 + d as f32 * 0.1;
            }
        }
        Codebook::new(entries, entry_dim)
    }

    #[test]
    fn test_codebook_lookup() {
        let cb = make_codebook(8);
        let entry0 = cb.lookup(0);
        assert_eq!(entry0.len(), 8);
        assert!((entry0[0] - 1.0).abs() < 1e-6);
        assert!((entry0[1] - 1.1).abs() < 1e-6);

        let entry255 = cb.lookup(255);
        assert!((entry255[0] - 256.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_group_single_codebook() {
        let cb = make_codebook(8);
        let codebooks = vec![cb];
        let indices = vec![0u8]; // look up entry 0
        let scale = 2.0;
        let mut output = vec![0.0f32; 8];

        dequant_aqlm_group(&codebooks, &indices, scale, &mut output);

        // Entry 0 = [1.0, 1.1, 1.2, ..., 1.7], scaled by 2.0
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_group_two_codebooks() {
        let cb1 = make_codebook(8);
        let cb2 = make_codebook(8);
        let codebooks = vec![cb1, cb2];
        let indices = vec![0u8, 1u8]; // entry 0 from cb1 + entry 1 from cb2
        let scale = 1.0;
        let mut output = vec![0.0f32; 8];

        dequant_aqlm_group(&codebooks, &indices, scale, &mut output);

        // Entry 0 = [1.0, 1.1, ...] + Entry 1 = [2.0, 2.1, ...] = [3.0, 3.2, ...]
        assert!((output[0] - 3.0).abs() < 1e-6);
        assert!((output[1] - 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_vec_dot_matches_dequant() {
        let cb1 = make_codebook(4);
        let cb2 = make_codebook(4);
        let codebooks = vec![cb1, cb2];
        let indices = vec![2u8, 5u8];
        let scale = 0.5;
        let activation = vec![1.0, 2.0, 3.0, 4.0];

        // Compute via dequant + manual dot
        let mut dequant = vec![0.0f32; 4];
        dequant_aqlm_group(&codebooks, &indices, scale, &mut dequant);
        let expected: f32 = dequant
            .iter()
            .zip(activation.iter())
            .map(|(w, a)| w * a)
            .sum();

        // Compute via fused dot
        let actual = vec_dot_aqlm(&codebooks, &indices, scale, &activation);

        assert!(
            (expected - actual).abs() < 1e-4,
            "fused dot {} != dequant dot {}",
            actual,
            expected
        );
    }
}
