//! Dynamic runtime quantization: per-layer precision switching at inference time.
//! Reference: Adaptive Layer-Wise Quantization (2025), BinaryMoS

/// Per-layer precision configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerPrecision {
    Full, // use the model's native quantization (Q4_K, Q8_0, etc.)
    Half, // skip every other block (2x faster, lower quality)
    Skip, // skip the entire layer (residual pass-through)
}

/// Dynamic quantization controller.
pub struct DynamicQuantController {
    n_layers: usize,
    precision: Vec<LayerPrecision>,
    activation_history: Vec<f64>, // EMA of activation norms per layer
    warmup_tokens: u32,
    tokens_seen: u32,
    // Thresholds
    skip_threshold: f64, // below this -> Skip
    half_threshold: f64, // below this -> Half, above -> Full
}

impl DynamicQuantController {
    pub fn new(n_layers: usize) -> Self {
        Self {
            n_layers,
            precision: vec![LayerPrecision::Full; n_layers],
            activation_history: vec![0.0; n_layers],
            warmup_tokens: 32,
            tokens_seen: 0,
            skip_threshold: 0.1,
            half_threshold: 0.5,
        }
    }

    /// Record activation norm and update precision for this layer.
    pub fn update(&mut self, layer: usize, activation_norm: f64) {
        let alpha = 0.1;
        self.activation_history[layer] =
            (1.0 - alpha) * self.activation_history[layer] + alpha * activation_norm;

        if self.tokens_seen <= self.warmup_tokens {
            self.precision[layer] = LayerPrecision::Full;
            return;
        }

        let norm = self.activation_history[layer];
        if norm < self.skip_threshold {
            self.precision[layer] = LayerPrecision::Skip;
        } else if norm < self.half_threshold {
            self.precision[layer] = LayerPrecision::Half;
        } else {
            self.precision[layer] = LayerPrecision::Full;
        }
    }

    /// Get current precision for a layer.
    pub fn precision(&self, layer: usize) -> LayerPrecision {
        self.precision[layer]
    }

    /// Start a new token.
    pub fn new_token(&mut self) {
        self.tokens_seen += 1;
    }

    /// Stats: what fraction of layers are at each precision.
    pub fn stats(&self) -> (usize, usize, usize) {
        let full = self
            .precision
            .iter()
            .filter(|&&p| p == LayerPrecision::Full)
            .count();
        let half = self
            .precision
            .iter()
            .filter(|&&p| p == LayerPrecision::Half)
            .count();
        let skip = self
            .precision
            .iter()
            .filter(|&&p| p == LayerPrecision::Skip)
            .count();
        (full, half, skip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_always_full() {
        let mut ctrl = DynamicQuantController::new(4);
        // During warmup (tokens_seen < 32), all layers stay Full regardless of norm
        for token in 0..32 {
            ctrl.new_token();
            for layer in 0..4 {
                ctrl.update(layer, 0.001); // very low norm
            }
            for layer in 0..4 {
                assert_eq!(
                    ctrl.precision(layer),
                    LayerPrecision::Full,
                    "layer {} should be Full during warmup (token {})",
                    layer,
                    token
                );
            }
        }
    }

    #[test]
    fn skip_when_low_norm() {
        let mut ctrl = DynamicQuantController::new(4);
        // Push past warmup
        for _ in 0..33 {
            ctrl.new_token();
            for layer in 0..4 {
                ctrl.update(layer, 0.001);
            }
        }
        // After warmup with very low activation norms, layers should be Skip
        assert_eq!(ctrl.precision(0), LayerPrecision::Skip);
    }

    #[test]
    fn half_when_medium_norm() {
        let mut ctrl = DynamicQuantController::new(4);
        // Push past warmup with medium norms
        for _ in 0..33 {
            ctrl.new_token();
            for layer in 0..4 {
                ctrl.update(layer, 0.3);
            }
        }
        // Medium activation norm -> Half precision
        assert_eq!(ctrl.precision(0), LayerPrecision::Half);
    }

    #[test]
    fn full_when_high_norm() {
        let mut ctrl = DynamicQuantController::new(4);
        // Push past warmup with high norms
        for _ in 0..33 {
            ctrl.new_token();
            for layer in 0..4 {
                ctrl.update(layer, 1.0);
            }
        }
        assert_eq!(ctrl.precision(0), LayerPrecision::Full);
    }

    #[test]
    fn stats_correct() {
        let mut ctrl = DynamicQuantController::new(6);
        // Push past warmup
        for _ in 0..33 {
            ctrl.new_token();
            // Set different norms: layers 0,1 low, layers 2,3 medium, layers 4,5 high
            ctrl.update(0, 0.001);
            ctrl.update(1, 0.001);
            ctrl.update(2, 0.3);
            ctrl.update(3, 0.3);
            ctrl.update(4, 1.0);
            ctrl.update(5, 1.0);
        }
        let (full, half, skip) = ctrl.stats();
        assert_eq!(full, 2, "expected 2 full layers");
        assert_eq!(half, 2, "expected 2 half layers");
        assert_eq!(skip, 2, "expected 2 skip layers");
        assert_eq!(full + half + skip, 6);
    }
}
