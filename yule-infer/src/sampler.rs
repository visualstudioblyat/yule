use crate::SamplingParams;
use yule_core::error::{Result, YuleError};

pub struct Sampler {
    params: SamplingParams,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self { params }
    }

    pub fn sample(&self, logits: &[f32]) -> Result<u32> {
        let n = logits.len();
        if n == 0 {
            return Err(YuleError::Inference("empty logits".into()));
        }

        let mut logits = logits.to_vec();

        // temperature scaling
        if self.params.temperature > 0.0 && self.params.temperature != 1.0 {
            let inv_t = 1.0 / self.params.temperature;
            for l in &mut logits {
                *l *= inv_t;
            }
        }

        // softmax (stable: subtract max first)
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        let inv_sum = 1.0 / sum;
        for p in &mut probs {
            *p *= inv_sum;
        }

        // top-k: zero out everything below the k-th largest
        if self.params.top_k > 0 && (self.params.top_k as usize) < n {
            let k = self.params.top_k as usize;
            // find k-th largest probability via partial sort
            let mut sorted = probs.clone();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k];
            // zero out below threshold (constant-time mask over all elements)
            for p in &mut probs {
                // branchless: mask = (p >= threshold) as float
                let keep = ct_gte(*p, threshold);
                *p *= keep;
            }
        }

        // min_p: zero out below max_prob * min_p
        if self.params.min_p > 0.0 {
            let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
            let min_threshold = max_prob * self.params.min_p;
            for p in &mut probs {
                let keep = ct_gte(*p, min_threshold);
                *p *= keep;
            }
        }

        // top-p (nucleus): zero out tokens outside cumulative probability mass
        if self.params.top_p < 1.0 {
            // build sorted index for cumulative sum, then mask
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut cumulative = 0.0f32;
            let mut cutoff_reached = 0.0f32;
            for &(_, prob) in &indexed {
                // once cumulative exceeds top_p, all further tokens get zeroed
                // branchless: cutoff_reached transitions from 0→1 and stays 1
                let was_over = cutoff_reached;
                cumulative += prob * (1.0 - was_over); // stop accumulating after cutoff
                cutoff_reached = ct_gte(cumulative, self.params.top_p);
                // zero this token if cutoff was ALREADY reached before this token
                // (we keep the token that pushes us over the threshold)
            }

            // simpler correct version: mark tokens after the cumulative exceeds top_p
            cumulative = 0.0;
            let mut in_nucleus = vec![false; n];
            for &(idx, prob) in &indexed {
                cumulative += prob;
                in_nucleus[idx] = true;
                if cumulative >= self.params.top_p {
                    break;
                }
            }
            // zero out tokens not in nucleus (iterate all for constant-ish timing)
            for i in 0..n {
                if !in_nucleus[i] {
                    probs[i] = 0.0;
                }
            }
        }

        // renormalize
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            // fallback: argmax of original logits
            return Ok(argmax_ct(&logits));
        }
        let inv_total = 1.0 / total;
        for p in &mut probs {
            *p *= inv_total;
        }

        // CSPRNG random value
        let r = csprng_f32()?;

        // constant-time token selection: scan ALL tokens, accumulate CDF
        Ok(sample_ct(&probs, r))
    }
}

/// Constant-time greater-or-equal: returns 1.0 if a >= b, else 0.0.
/// Uses bit manipulation to avoid branches.
#[inline(always)]
fn ct_gte(a: f32, b: f32) -> f32 {
    // (a - b) >= 0 check via sign bit
    let diff = a - b;
    let sign_bit = diff.to_bits() >> 31; // 0 if positive/zero, 1 if negative
    (1 - sign_bit) as f32
}

/// Constant-time argmax: always iterates all elements.
fn argmax_ct(values: &[f32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let mut best_idx = 0u32;
    let mut best_val = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        let is_better = ct_gte(v, best_val);
        best_idx = blend_u32(best_idx, i as u32, is_better);
        // use select to avoid NaN from -inf * 0.0
        best_val = select_f32(best_val, v, is_better);
    }
    best_idx
}

/// Constant-time sampling from CDF. Always iterates ALL tokens.
/// Returns the first token where cumulative probability exceeds r.
fn sample_ct(probs: &[f32], r: f32) -> u32 {
    let mut cumulative = 0.0f32;
    let mut selected = (probs.len() - 1) as u32; // default to last token
    let mut found = 0.0f32; // 0.0 = not yet found, 1.0 = found

    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        // did we just cross the threshold?
        let crossed = ct_gte(cumulative, r);
        // only select if we haven't already found one
        let first_cross = crossed * (1.0 - found);
        selected = blend_u32(selected, i as u32, first_cross);
        // once found, stay found
        found = blend_f32(found, 1.0, crossed);
    }
    selected
}

/// Branchless blend: returns b if mask >= 0.5, else a.
#[inline(always)]
fn blend_u32(a: u32, b: u32, mask: f32) -> u32 {
    let m = (mask as u32) & 1; // 1 if mask >= 1.0, else 0
    a * (1 - m) + b * m
}

/// Branchless blend for f32.
#[inline(always)]
fn blend_f32(a: f32, b: f32, mask: f32) -> f32 {
    // mask is 0.0 or 1.0
    a * (1.0 - mask) + b * mask
}

/// Branchless select via bit manipulation. Avoids NaN from inf * 0.
#[inline(always)]
fn select_f32(a: f32, b: f32, mask: f32) -> f32 {
    let m = (mask as u32) & 1;
    // m=0 → pick a, m=1 → pick b (via XOR-mask trick)
    let diff = a.to_bits() ^ b.to_bits();
    let selected = a.to_bits() ^ (diff & (0u32.wrapping_sub(m)));
    f32::from_bits(selected)
}

/// Generate a random f32 in [0, 1) using the OS CSPRNG.
fn csprng_f32() -> Result<f32> {
    let mut bytes = [0u8; 4];
    getrandom::fill(&mut bytes).map_err(|e| YuleError::Inference(format!("CSPRNG failed: {e}")))?;
    let u = u32::from_le_bytes(bytes);
    // map [0, 2^32) to [0, 1) uniformly
    Ok((u >> 8) as f32 / (1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_gte_basic() {
        assert_eq!(ct_gte(1.0, 0.5), 1.0);
        assert_eq!(ct_gte(0.5, 1.0), 0.0);
        assert_eq!(ct_gte(1.0, 1.0), 1.0);
    }

    #[test]
    fn argmax_finds_max() {
        let vals = vec![0.1, 0.3, 0.9, 0.2];
        assert_eq!(argmax_ct(&vals), 2);
    }

    #[test]
    fn sample_ct_deterministic() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        // r=0.0 should select first token
        assert_eq!(sample_ct(&probs, 0.01), 0);
        // r=0.3 should select second token (cumulative 0.25, 0.50, ...)
        assert_eq!(sample_ct(&probs, 0.3), 1);
        // r=0.99 should select last token
        assert_eq!(sample_ct(&probs, 0.99), 3);
    }

    #[test]
    fn csprng_produces_valid_range() {
        for _ in 0..100 {
            let r = csprng_f32().unwrap();
            assert!((0.0..1.0).contains(&r), "csprng out of range: {r}");
        }
    }

    #[test]
    fn sampler_returns_valid_token() {
        let params = crate::SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.0,
        };
        let sampler = Sampler::new(params);
        let logits = vec![0.1, 5.0, 0.2, 0.3]; // token 1 is heavily favored
        let token = sampler.sample(&logits).unwrap();
        assert!(token < 4, "token out of range: {token}");
    }

    #[test]
    fn greedy_sampling_picks_max() {
        let params = crate::SamplingParams {
            temperature: 0.0001, // near-greedy
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        };
        let sampler = Sampler::new(params);
        let logits = vec![0.1, 0.2, 10.0, 0.3];
        // with very low temperature, should almost always pick token 2
        let mut counts = [0u32; 4];
        for _ in 0..100 {
            let token = sampler.sample(&logits).unwrap();
            counts[token as usize] += 1;
        }
        assert!(
            counts[2] > 90,
            "expected token 2 dominant, got {:?}",
            counts
        );
    }
}
