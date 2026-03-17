//! Mixture-of-Depths: dynamically skip transformer layers for easy tokens.
//! Reference: "Mixture-of-Depths" (2024), "Mixture-of-Recursions" (NeurIPS 2025)
//!
//! A lightweight router at each layer decides: compute full attention+MLP,
//! or just pass through via residual connection. Up to 2x inference throughput.

/// Configuration for layer skipping.
pub struct MoDConfig {
    /// Minimum activation norm to process a layer (skip if below this).
    pub skip_threshold: f32,
    /// Number of warmup tokens before enabling skipping.
    pub warmup_tokens: u32,
    /// Maximum fraction of layers that can be skipped per token.
    pub max_skip_ratio: f32,
}

impl Default for MoDConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 0.1,
            warmup_tokens: 10,
            max_skip_ratio: 0.5,
        }
    }
}

/// Track which layers were skipped and their activation norms.
pub struct MoDStats {
    /// Per-layer flag: true = computed, false = skipped.
    pub layers_computed: Vec<bool>,
    /// Norm of intermediate activation per layer.
    pub activation_norms: Vec<f32>,
    /// Total tokens processed.
    pub total_tokens: u64,
    /// Total layer computations performed.
    pub total_layers_computed: u64,
    /// Total layer computations skipped.
    pub total_layers_skipped: u64,
}

impl MoDStats {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers_computed: vec![true; n_layers],
            activation_norms: vec![0.0; n_layers],
            total_tokens: 0,
            total_layers_computed: 0,
            total_layers_skipped: 0,
        }
    }

    /// Record a decision for one layer.
    pub fn record(&mut self, layer: usize, computed: bool, norm: f32) {
        self.layers_computed[layer] = computed;
        self.activation_norms[layer] = norm;
        if computed {
            self.total_layers_computed += 1;
        } else {
            self.total_layers_skipped += 1;
        }
    }

    /// Start a new token.
    pub fn new_token(&mut self) {
        self.total_tokens += 1;
    }

    /// Fraction of computation saved.
    pub fn skip_ratio(&self) -> f64 {
        let total = self.total_layers_computed + self.total_layers_skipped;
        if total == 0 {
            return 0.0;
        }
        self.total_layers_skipped as f64 / total as f64
    }
}

/// Decide whether to skip a layer based on activation norm.
///
/// Returns `true` if the layer should be skipped (residual pass-through only).
/// Never skips during warmup, and never skips the first or last layer.
pub fn should_skip_layer(
    config: &MoDConfig,
    hidden: &[f32],
    layer: usize,
    n_layers: usize,
    tokens_seen: u64,
    layers_skipped_this_token: usize,
) -> bool {
    // Don't skip during warmup
    if tokens_seen < config.warmup_tokens as u64 {
        return false;
    }

    // Don't skip first or last layer (they're always important)
    if layer == 0 || layer == n_layers - 1 {
        return false;
    }

    // Don't exceed max skip ratio
    let max_skips = (n_layers as f32 * config.max_skip_ratio) as usize;
    if layers_skipped_this_token >= max_skips {
        return false;
    }

    // Compute activation norm
    let norm = activation_norm(hidden);

    // Skip if norm is below threshold (activation is near-zero, layer would do little)
    norm < config.skip_threshold
}

/// Compute RMS norm of a vector (for activation analysis).
pub fn activation_norm(hidden: &[f32]) -> f32 {
    let ss: f32 = hidden.iter().map(|x| x * x).sum();
    (ss / hidden.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_during_warmup() {
        let config = MoDConfig {
            warmup_tokens: 10,
            ..Default::default()
        };
        // Even with zero-norm hidden state, should NOT skip during warmup
        let hidden = vec![0.0f32; 64];
        for tokens_seen in 0..10 {
            assert!(
                !should_skip_layer(&config, &hidden, 2, 12, tokens_seen, 0),
                "should not skip at token {tokens_seen} (warmup)"
            );
        }
    }

    #[test]
    fn test_skip_first_last_layer() {
        let config = MoDConfig::default();
        let hidden = vec![0.0f32; 64]; // zero norm, would normally be skipped
        let n_layers = 12;
        let tokens_seen = 100;
        // Layer 0 must not be skipped
        assert!(!should_skip_layer(
            &config,
            &hidden,
            0,
            n_layers,
            tokens_seen,
            0
        ));
        // Last layer must not be skipped
        assert!(!should_skip_layer(
            &config,
            &hidden,
            n_layers - 1,
            n_layers,
            tokens_seen,
            0
        ));
    }

    #[test]
    fn test_skip_low_norm() {
        let config = MoDConfig {
            skip_threshold: 0.1,
            warmup_tokens: 0,
            max_skip_ratio: 0.5,
        };
        // Very small activations -> should skip
        let hidden = vec![0.001f32; 64];
        assert!(should_skip_layer(&config, &hidden, 3, 12, 100, 0));

        // Large activations -> should NOT skip
        let hidden_big = vec![5.0f32; 64];
        assert!(!should_skip_layer(&config, &hidden_big, 3, 12, 100, 0));
    }

    #[test]
    fn test_max_skip_ratio() {
        let config = MoDConfig {
            skip_threshold: 0.1,
            warmup_tokens: 0,
            max_skip_ratio: 0.5,
        };
        let n_layers = 12;
        let max_skips = (n_layers as f32 * config.max_skip_ratio) as usize; // 6
        let hidden = vec![0.0f32; 64]; // zero norm, would be skipped

        // At max_skips already skipped, should NOT skip more
        assert!(!should_skip_layer(
            &config, &hidden, 3, n_layers, 100, max_skips
        ));
        // Below max_skips, should skip
        assert!(should_skip_layer(
            &config,
            &hidden,
            3,
            n_layers,
            100,
            max_skips - 1
        ));
    }

    #[test]
    fn test_stats_tracking() {
        let mut stats = MoDStats::new(4);
        stats.new_token();

        // Record: layers 0,3 computed; layers 1,2 skipped
        stats.record(0, true, 1.0);
        stats.record(1, false, 0.01);
        stats.record(2, false, 0.02);
        stats.record(3, true, 0.8);

        assert_eq!(stats.total_layers_computed, 2);
        assert_eq!(stats.total_layers_skipped, 2);
        assert!((stats.skip_ratio() - 0.5).abs() < 1e-10);
        assert_eq!(stats.total_tokens, 1);

        // Verify per-layer state
        assert!(stats.layers_computed[0]);
        assert!(!stats.layers_computed[1]);
        assert!(!stats.layers_computed[2]);
        assert!(stats.layers_computed[3]);
    }
}
