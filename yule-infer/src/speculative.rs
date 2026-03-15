//! Speculative decoding for accelerated inference.
//!
//! Generates multiple draft tokens cheaply, then verifies them in one forward pass.
//! Accepted tokens are kept; rejected tokens trigger a resample from the corrected
//! distribution. Net effect: multiple tokens per forward pass at full-model quality.
//!
//! # Strategies
//!
//! - **Self-speculation (LayerSkip):** Uses the same model with layers skipped for
//!   draft generation. No additional VRAM cost. ~1.4-1.8x speedup.
//! - **EAGLE-3 (future):** Lightweight prediction head on hidden states. Tree attention
//!   for parallel verification. ~2.4-2.8x speedup.
//! - **Medusa heads (future):** Multiple linear projection heads for parallel next-token
//!   prediction. ~2.3x average acceptance length.

use yule_core::error::{Result, YuleError};

use crate::sampler::Sampler;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate before verification.
    pub draft_length: u32,
    /// Which layers to skip during draft generation (for self-speculation).
    /// E.g., for a 32-layer model, skipping layers 20-31 uses only the first 20.
    pub skip_layers_start: u32,
    /// Strategy to use for draft generation.
    pub strategy: SpeculativeStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeculativeStrategy {
    /// Self-speculation: skip trailing layers for draft, use full model for verify.
    LayerSkip,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 4,
            skip_layers_start: 0, // must be set based on model layer count
            strategy: SpeculativeStrategy::LayerSkip,
        }
    }
}

/// Result of a speculative decode step: the accepted tokens and the final logits.
pub struct SpeculativeResult {
    /// Tokens accepted from the draft (may be 0 to draft_length).
    pub accepted_tokens: Vec<u32>,
    /// Logits from the verification pass for the next token after accepted ones.
    pub next_logits: Vec<f32>,
    /// Number of draft tokens that were accepted.
    pub acceptance_count: u32,
    /// Total draft tokens generated (for stats).
    pub draft_count: u32,
}

/// Rejection sampling: accept draft token with probability min(1, p_target / p_draft).
/// If rejected, resample from norm(max(0, p_target - p_draft)).
///
/// This guarantees the output distribution exactly matches the target model.
pub fn rejection_sample(
    target_probs: &[f32],
    draft_probs: &[f32],
    draft_token: u32,
    sampler: &Sampler,
) -> Result<(bool, Option<u32>)> {
    let idx = draft_token as usize;
    if idx >= target_probs.len() || idx >= draft_probs.len() {
        return Err(YuleError::Inference("token index out of range".into()));
    }

    let p_target = target_probs[idx];
    let p_draft = draft_probs[idx];

    // acceptance probability = min(1, p_target / p_draft)
    let accept_prob = if p_draft > 0.0 {
        (p_target / p_draft).min(1.0)
    } else if p_target > 0.0 {
        1.0 // draft assigned 0 probability but target didn't — always accept
    } else {
        0.0 // both zero — reject
    };

    // draw random value
    let r = csprng_f32()?;

    if r < accept_prob {
        Ok((true, None)) // accepted
    } else {
        // rejected: resample from norm(max(0, p_target - p_draft))
        let mut residual: Vec<f32> = target_probs
            .iter()
            .zip(draft_probs.iter())
            .map(|(&pt, &pd)| (pt - pd).max(0.0))
            .collect();

        let sum: f32 = residual.iter().sum();
        if sum <= 0.0 {
            // fallback: sample from target distribution
            let token = sampler.sample(target_probs)?;
            return Ok((false, Some(token)));
        }

        // normalize residual
        let inv_sum = 1.0 / sum;
        for p in &mut residual {
            *p *= inv_sum;
        }

        // sample from residual distribution
        let r2 = csprng_f32()?;
        let mut cumulative = 0.0f32;
        let mut selected = (residual.len() - 1) as u32;
        for (i, &p) in residual.iter().enumerate() {
            cumulative += p;
            if cumulative > r2 {
                selected = i as u32;
                break;
            }
        }

        Ok((false, Some(selected)))
    }
}

/// Convert logits to probability distribution (softmax).
pub fn logits_to_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
    let mut scaled = logits.to_vec();

    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for l in &mut scaled {
            *l *= inv_t;
        }
    }

    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum;
    for p in &mut probs {
        *p *= inv_sum;
    }
    probs
}

/// Generate a random f32 in [0, 1) using OS CSPRNG.
fn csprng_f32() -> Result<f32> {
    let mut bytes = [0u8; 4];
    getrandom::fill(&mut bytes)
        .map_err(|e| YuleError::Inference(format!("CSPRNG failed: {e}")))?;
    let u = u32::from_le_bytes(bytes);
    Ok((u >> 8) as f32 / (1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_to_probs_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = logits_to_probs(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    #[test]
    fn logits_to_probs_temperature_sharpens() {
        let logits = vec![1.0, 2.0, 3.0];
        let warm = logits_to_probs(&logits, 2.0);
        let cold = logits_to_probs(&logits, 0.1);
        // cold should concentrate more on the max
        assert!(cold[2] > warm[2], "cold should be sharper");
    }

    #[test]
    fn rejection_sample_always_accepts_identical_distributions() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let sampler = Sampler::new(crate::SamplingParams {
            temperature: 1.0,
            ..Default::default()
        });

        let mut accepted = 0;
        for _ in 0..100 {
            let (ok, _) = rejection_sample(&probs, &probs, 0, &sampler).unwrap();
            if ok {
                accepted += 1;
            }
        }
        // with identical distributions, acceptance rate should be 100%
        assert_eq!(accepted, 100);
    }

    #[test]
    fn rejection_sample_returns_valid_token_on_reject() {
        // target concentrates on token 2, draft concentrates on token 0
        let target = vec![0.01, 0.01, 0.97, 0.01];
        let draft = vec![0.97, 0.01, 0.01, 0.01];
        let sampler = Sampler::new(crate::SamplingParams {
            temperature: 1.0,
            ..Default::default()
        });

        // drafting token 0 (which target doesn't want) should usually reject
        let mut rejected = 0;
        for _ in 0..100 {
            let (ok, resampled) = rejection_sample(&target, &draft, 0, &sampler).unwrap();
            if !ok {
                rejected += 1;
                assert!(resampled.is_some());
                assert!((resampled.unwrap() as usize) < target.len());
            }
        }
        assert!(rejected > 80, "expected mostly rejections, got {rejected}/100");
    }
}
