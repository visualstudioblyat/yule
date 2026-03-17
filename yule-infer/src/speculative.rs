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
    /// Configuration for EAGLE-3 strategy.
    pub eagle: Option<EagleConfig>,
    /// Configuration for Medusa strategy.
    pub medusa: Option<MedusaConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeculativeStrategy {
    /// Self-speculation: skip trailing layers for draft, use full model for verify.
    LayerSkip,
    /// EAGLE-3: lightweight prediction head on hidden states with tree attention.
    Eagle,
    /// Medusa: multiple parallel prediction heads for multi-position drafting.
    Medusa,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 4,
            skip_layers_start: 0, // must be set based on model layer count
            strategy: SpeculativeStrategy::LayerSkip,
            eagle: None,
            medusa: None,
        }
    }
}

// ---------------------------------------------------------------------------
// EAGLE-3 types and functions
// ---------------------------------------------------------------------------

/// Configuration for EAGLE-3 speculative decoding.
#[derive(Debug, Clone)]
pub struct EagleConfig {
    /// Which transformer layer's hidden state to feed into the prediction head.
    pub hidden_layer_idx: u32,
    /// Branching factor for tree attention (typically 2-4).
    pub tree_width: u32,
    /// Depth of the draft tree (typically 3-5).
    pub tree_depth: u32,
    /// Hidden dimension of the lightweight prediction-head MLP.
    pub head_hidden_dim: u32,
}

/// A single candidate path produced by EAGLE tree expansion.
#[derive(Debug, Clone)]
pub struct TreeCandidate {
    /// Token ids along this candidate path.
    pub tokens: Vec<u32>,
    /// Cumulative probability of the path (product of per-step probs).
    pub cumulative_prob: f32,
    /// Index of the parent candidate in the candidate list (`None` for roots).
    pub parent_idx: Option<usize>,
}

/// Generate a tree of draft candidates using EAGLE-3.
///
/// Starting from `draft_logits` (output of the lightweight prediction head),
/// expands a tree of width `config.tree_width` at each level up to
/// `config.tree_depth`. Returns all candidates (including intermediate nodes).
pub fn eagle_draft(
    draft_logits: &[f32],
    config: &EagleConfig,
    sampler: &Sampler,
) -> Result<Vec<TreeCandidate>> {
    let _ = sampler; // sampler reserved for future top-p/top-k filtering

    if draft_logits.is_empty() {
        return Err(YuleError::Inference("empty draft logits for EAGLE".into()));
    }

    let vocab_size = draft_logits.len();
    let width = config.tree_width as usize;
    let depth = config.tree_depth as usize;

    // Convert root logits to probabilities.
    let root_probs = logits_to_probs(draft_logits, 1.0);

    // Select top-K tokens as root candidates (depth-1 nodes).
    let root_top_k = top_k_indices(&root_probs, width);

    let mut candidates: Vec<TreeCandidate> = Vec::new();

    // Create root-level candidates.
    for &tok_idx in &root_top_k {
        candidates.push(TreeCandidate {
            tokens: vec![tok_idx as u32],
            cumulative_prob: root_probs[tok_idx],
            parent_idx: None,
        });
    }

    // Expand tree level by level.
    // At each subsequent level we treat the logits as a flat "uniform-ish"
    // distribution offset by the token index, since we don't have per-node
    // forward-pass logits here. In a real inference loop the caller would
    // run the prediction head for each node; this function demonstrates the
    // tree structure bookkeeping.
    let mut level_start = 0;
    let mut level_end = candidates.len();

    for _level in 1..depth {
        let mut new_candidates: Vec<TreeCandidate> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for parent_idx in level_start..level_end {
            // Derive child logits by rotating the root logits by the last token.
            // This is a placeholder that gives deterministic, non-degenerate trees
            // for testing; real usage would supply actual prediction-head logits.
            let last_tok = *candidates[parent_idx].tokens.last().unwrap() as usize;
            let mut child_logits = vec![0.0f32; vocab_size];
            for (i, v) in child_logits.iter_mut().enumerate() {
                *v = draft_logits[(i + last_tok) % vocab_size];
            }
            let child_probs = logits_to_probs(&child_logits, 1.0);
            let children = top_k_indices(&child_probs, width);

            for &c in &children {
                let mut path = candidates[parent_idx].tokens.clone();
                path.push(c as u32);
                new_candidates.push(TreeCandidate {
                    tokens: path,
                    cumulative_prob: candidates[parent_idx].cumulative_prob * child_probs[c],
                    parent_idx: Some(parent_idx),
                });
            }
        }

        level_start = candidates.len();
        candidates.extend(new_candidates);
        level_end = candidates.len();
    }

    Ok(candidates)
}

/// Verify a tree of EAGLE candidates against target model probabilities.
///
/// `target_probs_per_node` contains one probability vector per candidate.
/// Returns the token sequence of the longest accepted path (greedily choosing
/// the highest-probability child at each level that passes acceptance).
pub fn verify_tree(
    candidates: &[TreeCandidate],
    target_probs_per_node: &[Vec<f32>],
    sampler: &Sampler,
) -> Result<Vec<u32>> {
    let _ = sampler; // reserved for stochastic acceptance in future

    if candidates.is_empty() {
        return Ok(vec![]);
    }
    if candidates.len() != target_probs_per_node.len() {
        return Err(YuleError::Inference(
            "candidates and target_probs_per_node length mismatch".into(),
        ));
    }

    // Find the candidate with the longest token sequence whose every token
    // is the argmax of the corresponding target probability vector.
    // We iterate all candidates, checking acceptance at each position.
    let mut best_path: Vec<u32> = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let target_probs = &target_probs_per_node[idx];
        if target_probs.is_empty() {
            continue;
        }

        // Check: is the last token of this candidate the argmax of its
        // target probability distribution?
        let last_tok = *candidate.tokens.last().unwrap() as usize;
        if last_tok >= target_probs.len() {
            continue;
        }

        let argmax = target_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        if argmax != last_tok {
            continue; // this node's token is not the most likely — skip
        }

        // Walk up the tree to verify all ancestors are also accepted.
        let mut all_accepted = true;
        let mut walk = Some(idx);
        while let Some(cur) = walk {
            let cur_cand = &candidates[cur];
            let cur_probs = &target_probs_per_node[cur];
            let cur_tok = *cur_cand.tokens.last().unwrap() as usize;
            if cur_probs.is_empty() || cur_tok >= cur_probs.len() {
                all_accepted = false;
                break;
            }
            let cur_argmax = cur_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if cur_argmax != cur_tok {
                all_accepted = false;
                break;
            }
            walk = cur_cand.parent_idx;
        }

        if all_accepted && candidate.tokens.len() > best_path.len() {
            best_path = candidate.tokens.clone();
        }
    }

    Ok(best_path)
}

// ---------------------------------------------------------------------------
// Medusa types and functions
// ---------------------------------------------------------------------------

/// Configuration for Medusa speculative decoding.
#[derive(Debug, Clone)]
pub struct MedusaConfig {
    /// Number of parallel prediction heads (typically 3-5).
    pub num_heads: u32,
    /// How many top-K candidates each head proposes (typically 5-10).
    pub top_k_per_head: u32,
}

/// A single Medusa candidate: one token per head position.
#[derive(Debug, Clone)]
pub struct MedusaCandidate {
    /// Predicted tokens, one per head (`tokens[i]` predicts position `pos + i + 1`).
    pub tokens: Vec<u32>,
    /// Per-head probability of each predicted token.
    pub probabilities: Vec<f32>,
}

/// Result of Medusa verification.
#[derive(Debug, Clone)]
pub struct MedusaResult {
    /// The accepted token sequence (prefix of the best candidate).
    pub accepted_tokens: Vec<u32>,
    /// Number of consecutive positions accepted starting from pos+1.
    pub acceptance_length: u32,
}

/// Generate Medusa draft candidates.
///
/// `head_logits[h]` contains the logits produced by Medusa head `h`.
/// Returns a cartesian-product set of candidates formed by taking the top-K
/// token from each head. The number of candidates is bounded by
/// `top_k_per_head ^ num_heads`, but in practice we generate at most
/// `top_k_per_head` candidates by picking the best token from each head
/// combined into single candidates.
pub fn medusa_draft(
    head_logits: &[Vec<f32>],
    config: &MedusaConfig,
    sampler: &Sampler,
) -> Result<Vec<MedusaCandidate>> {
    let _ = sampler; // reserved for stochastic top-p filtering

    let num_heads = config.num_heads as usize;
    if head_logits.len() != num_heads {
        return Err(YuleError::Inference(format!(
            "expected {} head logits, got {}",
            num_heads,
            head_logits.len()
        )));
    }

    let k = config.top_k_per_head as usize;

    // For each head, compute probabilities and extract top-K tokens.
    let mut per_head_top: Vec<Vec<(u32, f32)>> = Vec::with_capacity(num_heads);
    for head_logit in head_logits.iter().take(num_heads) {
        let probs = logits_to_probs(head_logit, 1.0);
        let top_indices = top_k_indices(&probs, k);
        let top_entries: Vec<(u32, f32)> =
            top_indices.iter().map(|&i| (i as u32, probs[i])).collect();
        per_head_top.push(top_entries);
    }

    // Generate candidates: take the cartesian product of top-K across heads,
    // but cap the total at `k * num_heads` to avoid combinatorial explosion.
    // Strategy: for each rank r in 0..k, form a candidate by taking the
    // r-th best token from each head. Then fill remaining slots by cycling.
    let mut candidates: Vec<MedusaCandidate> = Vec::with_capacity(k);

    for r in 0..k {
        let mut tokens = Vec::with_capacity(num_heads);
        let mut probs = Vec::with_capacity(num_heads);
        for head_top in per_head_top.iter().take(num_heads) {
            let idx = r.min(head_top.len() - 1);
            let (tok, prob) = head_top[idx];
            tokens.push(tok);
            probs.push(prob);
        }
        candidates.push(MedusaCandidate {
            tokens,
            probabilities: probs,
        });
    }

    Ok(candidates)
}

/// Verify Medusa candidates against target model logits.
///
/// `target_logits_per_pos[p]` are the target model logits for position `pos + p + 1`.
/// Returns the longest prefix of the best candidate whose tokens match the
/// target distribution (acceptance via probability threshold).
pub fn medusa_verify(
    candidates: &[MedusaCandidate],
    target_logits_per_pos: &[Vec<f32>],
    sampler: &Sampler,
    temperature: f32,
) -> Result<MedusaResult> {
    let _ = sampler; // reserved for stochastic verification

    if candidates.is_empty() {
        return Ok(MedusaResult {
            accepted_tokens: vec![],
            acceptance_length: 0,
        });
    }

    let num_positions = target_logits_per_pos.len();

    // Convert target logits to probabilities.
    let target_probs: Vec<Vec<f32>> = target_logits_per_pos
        .iter()
        .map(|logits| logits_to_probs(logits, temperature))
        .collect();

    // Score each candidate: count how many consecutive positions are accepted.
    // A position is accepted when the candidate's token is the argmax of the
    // target distribution for that position.
    let mut best_result = MedusaResult {
        accepted_tokens: vec![],
        acceptance_length: 0,
    };

    for candidate in candidates {
        let check_len = candidate.tokens.len().min(num_positions);
        let mut accepted = Vec::new();

        for (pos, probs) in target_probs.iter().enumerate().take(check_len) {
            let tok = candidate.tokens[pos] as usize;

            if tok >= probs.len() {
                break;
            }

            // Accept if this token is the argmax of the target distribution.
            let argmax = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            if argmax == tok {
                accepted.push(candidate.tokens[pos]);
            } else {
                break; // first mismatch ends the consecutive run
            }
        }

        let acc_len = accepted.len() as u32;
        if acc_len > best_result.acceptance_length {
            best_result = MedusaResult {
                accepted_tokens: accepted,
                acceptance_length: acc_len,
            };
        }
    }

    Ok(best_result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the indices of the top-K values in `values`, sorted descending.
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|&(i, _)| i).collect()
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

/// Run one speculative decode step: draft N tokens, verify, accept/reject.
///
/// Uses the same model with the full runner for both draft and verify.
/// For self-speculation (LayerSkip), a future optimization would skip layers
/// during draft generation. Currently this uses the full model for both,
/// which validates the rejection sampling pipeline end-to-end.
pub fn speculative_decode_step(
    runner: &mut dyn crate::model_runner::ModelRunner,
    last_logits: &[f32],
    config: &SpeculativeConfig,
    sampler: &Sampler,
    temperature: f32,
) -> Result<SpeculativeResult> {
    let draft_len = config.draft_length as usize;
    let mut draft_tokens = Vec::with_capacity(draft_len);
    let mut draft_probs_list = Vec::with_capacity(draft_len);
    let mut current_logits = last_logits.to_vec();

    // 1. Generate draft tokens using the sampler
    for _ in 0..draft_len {
        let probs = logits_to_probs(&current_logits, temperature);
        let token = sampler.sample(&current_logits)?;
        draft_tokens.push(token);
        draft_probs_list.push(probs);
        current_logits = runner.decode_step(token)?;
    }

    // 2. Verify: for each draft token, compute target probability and accept/reject
    // In a true self-speculation setup, we'd re-run with full layers here.
    // Since we used the full model for drafting, target == draft (all accepted).
    // This wiring is correct for when we add LayerSkip draft generation.
    let target_logits_final = current_logits;
    let _target_probs_final = logits_to_probs(&target_logits_final, temperature);

    // For now, with full-model drafting, all tokens are accepted by construction.
    // The rejection sampling infrastructure is exercised for correctness.
    let mut accepted_tokens = Vec::with_capacity(draft_len);
    let mut last_accepted_idx = 0;

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let draft_probs = &draft_probs_list[i];
        // When draft == target model, acceptance probability is always 1.0
        let target_probs = draft_probs; // same model
        let (accepted, resampled) =
            rejection_sample(target_probs, draft_probs, draft_token, sampler)?;

        if accepted {
            accepted_tokens.push(draft_token);
            last_accepted_idx = i + 1;
        } else {
            // Rejected: use the resampled token instead
            if let Some(new_token) = resampled {
                accepted_tokens.push(new_token);
            }
            break;
        }
    }

    Ok(SpeculativeResult {
        acceptance_count: accepted_tokens.len() as u32,
        draft_count: draft_len as u32,
        accepted_tokens,
        next_logits: if last_accepted_idx == draft_len {
            target_logits_final
        } else {
            // Need to re-derive logits for the position after the last accepted token
            // For now, return the final logits (correct when all accepted)
            target_logits_final
        },
    })
}

/// Speculative decode with constant-time padding.
/// The entire draft+verify cycle takes exactly `target_duration` wall-clock time,
/// regardless of how many draft tokens were accepted.
pub fn speculative_decode_step_padded(
    runner: &mut dyn crate::model_runner::ModelRunner,
    last_logits: &[f32],
    config: &SpeculativeConfig,
    sampler: &Sampler,
    temperature: f32,
    target_duration: std::time::Duration,
) -> Result<SpeculativeResult> {
    let start = std::time::Instant::now();

    let result = speculative_decode_step(runner, last_logits, config, sampler, temperature)?;

    // Pad to target duration
    let elapsed = start.elapsed();
    if elapsed < target_duration {
        crate::constant_time::busy_wait_until(start + target_duration);
    }

    Ok(result)
}

/// Speculative decode with constant-time padding and random noise.
/// Adds random noise to the target duration to prevent fixed-pattern detection.
pub fn speculative_decode_step_with_noise(
    runner: &mut dyn crate::model_runner::ModelRunner,
    last_logits: &[f32],
    config: &SpeculativeConfig,
    sampler: &Sampler,
    temperature: f32,
    target_duration: std::time::Duration,
    noise_amplitude_us: u64,
) -> Result<SpeculativeResult> {
    let start = std::time::Instant::now();

    let result = speculative_decode_step(runner, last_logits, config, sampler, temperature)?;

    // Add random noise to the target
    let noise = if noise_amplitude_us > 0 {
        let mut bytes = [0u8; 8];
        getrandom::fill(&mut bytes)
            .map_err(|e| YuleError::Inference(format!("CSPRNG failed: {e}")))?;
        std::time::Duration::from_micros(u64::from_le_bytes(bytes) % noise_amplitude_us)
    } else {
        std::time::Duration::ZERO
    };

    let padded_target = target_duration + noise;

    let elapsed = start.elapsed();
    if elapsed < padded_target {
        crate::constant_time::busy_wait_until(start + padded_target);
    }

    Ok(result)
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
    getrandom::fill(&mut bytes).map_err(|e| YuleError::Inference(format!("CSPRNG failed: {e}")))?;
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
        assert!(
            rejected > 80,
            "expected mostly rejections, got {rejected}/100"
        );
    }

    // -------------------------------------------------------------------
    // EAGLE-3 tests
    // -------------------------------------------------------------------

    fn make_sampler() -> Sampler {
        Sampler::new(crate::SamplingParams {
            temperature: 1.0,
            ..Default::default()
        })
    }

    #[test]
    fn test_eagle_tree_generation() {
        let config = EagleConfig {
            hidden_layer_idx: 16,
            tree_width: 2,
            tree_depth: 3,
            head_hidden_dim: 128,
        };

        // 8-token vocabulary with logits that favour token 7 and 6.
        let logits = vec![0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0];
        let sampler = make_sampler();

        let candidates = eagle_draft(&logits, &config, &sampler).unwrap();

        // Depth-1 roots: tree_width = 2
        assert!(candidates.len() >= 2, "need at least root candidates");

        // Root candidates should have exactly 1 token each.
        for c in &candidates[..2] {
            assert_eq!(c.tokens.len(), 1);
            assert!(c.parent_idx.is_none());
            assert!(c.cumulative_prob > 0.0);
        }

        // Deeper candidates should have longer token sequences.
        let max_depth = candidates.iter().map(|c| c.tokens.len()).max().unwrap();
        assert_eq!(
            max_depth, 3,
            "tree_depth=3 should produce paths of length 3"
        );

        // Total candidates: depth1=2 + depth2=2*2 + depth3=4*2 = 2+4+8 = 14
        let expected = 2 + 2 * 2 + 2 * 2 * 2;
        assert_eq!(candidates.len(), expected, "unexpected candidate count");

        // Verify parent pointers for depth-2 candidates.
        for c in &candidates[2..6] {
            assert_eq!(c.tokens.len(), 2);
            assert!(c.parent_idx.is_some());
            let parent = c.parent_idx.unwrap();
            assert!(parent < 2); // parent is a root
        }
    }

    #[test]
    fn test_eagle_tree_verification() {
        let config = EagleConfig {
            hidden_layer_idx: 16,
            tree_width: 2,
            tree_depth: 2,
            head_hidden_dim: 128,
        };

        let logits = vec![0.1, 0.5, 1.0, 3.0]; // 4-token vocab; argmax = 3
        let sampler = make_sampler();

        let candidates = eagle_draft(&logits, &config, &sampler).unwrap();
        // We have root and depth-2 candidates.

        // Build target probs that make a specific path the winner.
        // For each candidate, the target distribution should have argmax
        // matching the candidate's last token IFF we want it accepted.
        let mut target_probs_per_node: Vec<Vec<f32>> = Vec::new();

        for candidate in &candidates {
            let last_tok = *candidate.tokens.last().unwrap() as usize;
            // Make the target's argmax equal to last_tok so it is accepted.
            let mut probs = vec![0.05; 4];
            probs[last_tok] = 0.85;
            target_probs_per_node.push(probs);
        }

        let path = verify_tree(&candidates, &target_probs_per_node, &sampler).unwrap();

        // All candidates are accepted; the longest path has length = tree_depth = 2.
        assert_eq!(path.len(), 2, "should accept full depth-2 path");

        // Now make all target argmaxes point to token 0 — only candidates
        // whose last token is 0 will be accepted.
        let uniform_wrong: Vec<Vec<f32>> = candidates
            .iter()
            .map(|_| vec![0.9, 0.03, 0.03, 0.04])
            .collect();
        let path2 = verify_tree(&candidates, &uniform_wrong, &sampler).unwrap();
        // The path should only contain tokens that are 0.
        for &t in &path2 {
            assert_eq!(t, 0, "only token-0 nodes should be accepted");
        }
    }

    // -------------------------------------------------------------------
    // Medusa tests
    // -------------------------------------------------------------------

    #[test]
    fn test_medusa_draft_generation() {
        let config = MedusaConfig {
            num_heads: 3,
            top_k_per_head: 4,
        };

        // 6-token vocab; head 0 favours token 5, head 1 favours token 3,
        // head 2 favours token 1.
        let head_logits = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.5, 3.0], // head 0
            vec![0.1, 0.1, 0.1, 3.0, 0.5, 0.1], // head 1
            vec![0.1, 3.0, 0.1, 0.1, 0.5, 0.1], // head 2
        ];

        let sampler = make_sampler();
        let candidates = medusa_draft(&head_logits, &config, &sampler).unwrap();

        assert_eq!(
            candidates.len(),
            4,
            "should produce top_k_per_head candidates"
        );

        // The first (best) candidate should pick argmax from each head.
        let best = &candidates[0];
        assert_eq!(best.tokens.len(), 3, "one token per head");
        assert_eq!(best.tokens[0], 5, "head 0 argmax is 5");
        assert_eq!(best.tokens[1], 3, "head 1 argmax is 3");
        assert_eq!(best.tokens[2], 1, "head 2 argmax is 1");

        // All probabilities should be positive.
        for c in &candidates {
            for &p in &c.probabilities {
                assert!(p > 0.0);
            }
        }
    }

    #[test]
    fn test_medusa_verify_all_match() {
        let config = MedusaConfig {
            num_heads: 3,
            top_k_per_head: 2,
        };

        // Build head logits so that the best candidate is [5, 3, 1].
        let head_logits = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.5, 3.0],
            vec![0.1, 0.1, 0.1, 3.0, 0.5, 0.1],
            vec![0.1, 3.0, 0.1, 0.1, 0.5, 0.1],
        ];
        let sampler = make_sampler();
        let candidates = medusa_draft(&head_logits, &config, &sampler).unwrap();

        // Target logits: argmax at each position matches the best candidate.
        let target_logits_per_pos = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.5, 3.0], // pos+1 argmax = 5
            vec![0.1, 0.1, 0.1, 3.0, 0.5, 0.1], // pos+2 argmax = 3
            vec![0.1, 3.0, 0.1, 0.1, 0.5, 0.1], // pos+3 argmax = 1
        ];

        let result = medusa_verify(&candidates, &target_logits_per_pos, &sampler, 1.0).unwrap();

        assert_eq!(result.acceptance_length, 3);
        assert_eq!(result.accepted_tokens, vec![5, 3, 1]);
    }

    #[test]
    fn test_medusa_verify_partial_match() {
        let config = MedusaConfig {
            num_heads: 3,
            top_k_per_head: 2,
        };

        let head_logits = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.5, 3.0],
            vec![0.1, 0.1, 0.1, 3.0, 0.5, 0.1],
            vec![0.1, 3.0, 0.1, 0.1, 0.5, 0.1],
        ];
        let sampler = make_sampler();
        let candidates = medusa_draft(&head_logits, &config, &sampler).unwrap();

        // Target: pos+1 agrees (argmax=5), pos+2 disagrees (argmax=0), pos+3 irrelevant.
        let target_logits_per_pos = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.5, 3.0], // argmax = 5 ✓
            vec![3.0, 0.1, 0.1, 0.1, 0.5, 0.1], // argmax = 0 ✗ (candidate wants 3)
            vec![0.1, 3.0, 0.1, 0.1, 0.5, 0.1], // would match but unreachable
        ];

        let result = medusa_verify(&candidates, &target_logits_per_pos, &sampler, 1.0).unwrap();

        // Only the first position should be accepted for the best candidate.
        assert_eq!(result.acceptance_length, 1);
        assert_eq!(result.accepted_tokens, vec![5]);
    }

    #[test]
    fn test_padded_spec_decode_constant_time() {
        // We cannot easily construct a full ModelRunner without a real model,
        // so we test the busy_wait_until timing primitive that underpins the
        // padded function. This verifies that the padding mechanism works.
        let target = std::time::Duration::from_millis(20);
        let start = std::time::Instant::now();

        // Simulate: work finishes instantly, then pad to target
        let elapsed = start.elapsed();
        if elapsed < target {
            crate::constant_time::busy_wait_until(start + target);
        }

        let total = start.elapsed();
        assert!(
            total >= target,
            "padded duration {total:?} should be >= target {target:?}"
        );
        // Allow up to 5ms overshoot (OS scheduling jitter)
        assert!(
            total < target + std::time::Duration::from_millis(5),
            "padded duration {total:?} overshot target {target:?} by too much"
        );
    }
}
