//! Per-tensor entropy analysis for detecting over-provisioned quantization.
//!
//! Based on Shannon entropy: H = -Σ p(x) log₂ p(x)
//! If a tensor's entropy is significantly below its allocated bits per weight,
//! it could be stored at lower precision without information loss.

use crate::dtype::DType;
use crate::tensor::TensorInfo;

/// Result of entropy analysis for a single tensor.
pub struct EntropyAnalysis {
    pub tensor_name: String,
    /// Shannon entropy of the raw byte stream, in bits (0.0 .. 8.0).
    pub byte_entropy: f64,
    /// Bits per weight allocated by the current DType.
    pub bits_allocated: f64,
    /// Wasted bits: `bits_allocated - byte_entropy`.
    pub waste_bits: f64,
    /// Compression ratio: `bits_allocated / byte_entropy` (infinite if entropy is zero).
    pub compression_ratio: f64,
    /// Suggested lower-precision DType, or `None` if current is already appropriate.
    pub recommended_dtype: Option<DType>,
}

/// Summary statistics for a model's entropy profile.
pub struct ModelEntropyProfile {
    pub total_tensors: usize,
    pub avg_entropy: f64,
    pub avg_waste: f64,
    pub max_waste: f64,
    pub total_bytes_current: u64,
    /// Estimated total bytes if every tensor used its recommended DType.
    pub total_bytes_optimal: u64,
    /// Potential size reduction as a percentage of `total_bytes_current`.
    pub potential_savings_pct: f64,
}

/// Analyze byte-level Shannon entropy of tensor data.
pub fn analyze_tensor_entropy(info: &TensorInfo, data: &[u8]) -> EntropyAnalysis {
    let byte_entropy = byte_entropy(data);
    let bits_allocated = info.dtype.bits_per_weight() as f64;
    let waste_bits = bits_allocated - byte_entropy;
    let compression_ratio = if byte_entropy > 0.0 {
        bits_allocated / byte_entropy
    } else {
        f64::INFINITY
    };
    let recommended_dtype = recommend_dtype(info.dtype, byte_entropy);

    EntropyAnalysis {
        tensor_name: info.name.clone(),
        byte_entropy,
        bits_allocated,
        waste_bits,
        compression_ratio,
        recommended_dtype,
    }
}

/// Analyze all tensors and return results sorted by waste (most wasteful first).
pub fn analyze_model_entropy(tensors: &[(TensorInfo, &[u8])]) -> Vec<EntropyAnalysis> {
    let mut analyses: Vec<EntropyAnalysis> = tensors
        .iter()
        .map(|(info, data)| analyze_tensor_entropy(info, data))
        .collect();
    analyses.sort_by(|a, b| {
        b.waste_bits
            .partial_cmp(&a.waste_bits)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    analyses
}

/// Recommend a DType based on byte-level entropy.
///
/// Rules:
/// - If entropy < 2.0 and current is F32/F16: recommend Q2_K
/// - If entropy < 3.0 and current is F32/F16: recommend Q3_K
/// - If entropy < 4.0 and current is F32: recommend Q4_K
/// - If entropy < 5.0 and current is F32: recommend Q5_K
/// - If entropy < 6.0 and current is F32: recommend Q6_K
/// - If entropy < 8.0 and current is F32: recommend Q8_0
/// - Otherwise: keep current
fn recommend_dtype(current: DType, entropy: f64) -> Option<DType> {
    match current {
        DType::F32 | DType::F16 => {
            if entropy < 2.0 {
                Some(DType::Q2_K)
            } else if entropy < 3.0 {
                Some(DType::Q3_K)
            } else if current == DType::F32 {
                // remaining thresholds only apply to F32
                if entropy < 4.0 {
                    Some(DType::Q4_K)
                } else if entropy < 5.0 {
                    Some(DType::Q5_K)
                } else if entropy < 6.0 {
                    Some(DType::Q6_K)
                } else if entropy < 8.0 {
                    Some(DType::Q8_0)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Compute summary statistics for a set of entropy analyses.
pub fn compute_model_profile(analyses: &[EntropyAnalysis]) -> ModelEntropyProfile {
    if analyses.is_empty() {
        return ModelEntropyProfile {
            total_tensors: 0,
            avg_entropy: 0.0,
            avg_waste: 0.0,
            max_waste: 0.0,
            total_bytes_current: 0,
            total_bytes_optimal: 0,
            potential_savings_pct: 0.0,
        };
    }

    let total_tensors = analyses.len();
    let sum_entropy: f64 = analyses.iter().map(|a| a.byte_entropy).sum();
    let sum_waste: f64 = analyses.iter().map(|a| a.waste_bits).sum();
    let max_waste = analyses
        .iter()
        .map(|a| a.waste_bits)
        .fold(f64::NEG_INFINITY, f64::max);

    let avg_entropy = sum_entropy / total_tensors as f64;
    let avg_waste = sum_waste / total_tensors as f64;

    // Estimate byte totals using bits_allocated as a proxy.
    // For each tensor we compute a notional size proportional to its BPW.
    let total_bytes_current: u64 = analyses.iter().map(|a| a.bits_allocated as u64).sum();
    let total_bytes_optimal: u64 = analyses
        .iter()
        .map(|a| {
            let bpw = match &a.recommended_dtype {
                Some(dt) => dt.bits_per_weight() as f64,
                None => a.bits_allocated,
            };
            bpw as u64
        })
        .sum();

    let potential_savings_pct = if total_bytes_current > 0 {
        (1.0 - total_bytes_optimal as f64 / total_bytes_current as f64) * 100.0
    } else {
        0.0
    };

    ModelEntropyProfile {
        total_tensors,
        avg_entropy,
        avg_waste,
        max_waste,
        total_bytes_current,
        total_bytes_optimal,
        potential_savings_pct,
    }
}

/// Compute Shannon entropy (in bits) of a byte stream.
fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut entropy = 0.0f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorInfo;

    fn make_info(name: &str, dtype: DType, size_bytes: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype,
            shape: vec![size_bytes],
            offset: 0,
            size_bytes,
        }
    }

    #[test]
    fn test_uniform_entropy() {
        // All bytes identical → entropy = 0.0
        let data = vec![0x55u8; 1024];
        let info = make_info("uniform", DType::F32, data.len() as u64);
        let analysis = analyze_tensor_entropy(&info, &data);
        assert!(
            (analysis.byte_entropy - 0.0).abs() < 1e-10,
            "expected entropy 0.0, got {}",
            analysis.byte_entropy
        );
    }

    #[test]
    fn test_maximum_entropy() {
        // Every byte value appears exactly the same number of times → entropy = 8.0
        let mut data = Vec::with_capacity(256 * 100);
        for _ in 0..100 {
            for b in 0u8..=255 {
                data.push(b);
            }
        }
        let info = make_info("max_entropy", DType::F32, data.len() as u64);
        let analysis = analyze_tensor_entropy(&info, &data);
        assert!(
            (analysis.byte_entropy - 8.0).abs() < 1e-10,
            "expected entropy 8.0, got {}",
            analysis.byte_entropy
        );
    }

    #[test]
    fn test_recommendation() {
        // F32 tensor with entropy 3.5 → should recommend Q4_K
        // We don't need real data for this test — just call recommend_dtype directly.
        let rec = recommend_dtype(DType::F32, 3.5);
        assert_eq!(rec, Some(DType::Q4_K));
    }

    #[test]
    fn test_profile_computation() {
        let info_a = make_info("a", DType::F32, 1024);
        let info_b = make_info("b", DType::F32, 1024);

        // Tensor A: uniform data → entropy ~0, huge waste
        let data_a = vec![0x00u8; 1024];
        // Tensor B: max-entropy data → entropy ~8, minimal waste
        let mut data_b = Vec::with_capacity(256 * 4);
        for _ in 0..4 {
            for b in 0u8..=255 {
                data_b.push(b);
            }
        }

        let analyses =
            analyze_model_entropy(&[(info_a, data_a.as_slice()), (info_b, data_b.as_slice())]);

        // Should be sorted by waste descending: uniform first
        assert_eq!(analyses[0].tensor_name, "a");
        assert_eq!(analyses[1].tensor_name, "b");

        let profile = compute_model_profile(&analyses);
        assert_eq!(profile.total_tensors, 2);
        assert!(profile.avg_entropy > 0.0);
        assert!(profile.max_waste > 0.0);
        assert!(profile.potential_savings_pct > 0.0);
    }
}
