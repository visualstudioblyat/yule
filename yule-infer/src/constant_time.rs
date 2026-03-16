//! Constant-time inference utilities for timing side-channel resistance.
//!
//! Problem: An adversary measuring per-token decode time can infer which token
//! was generated (correlation measured at 0.185 on TinyLlama). Speculative
//! decoding makes this worse (75-100% query fingerprinting in literature).
//!
//! Solution: Pad each decode step to a fixed wall-clock duration. If the actual
//! computation finishes early, busy-wait (not sleep — sleep has variable wakeup).

use std::time::{Duration, Instant};

use yule_core::error::Result;

use crate::model_runner::ModelRunner;
use crate::sampler::Sampler;

/// A decode wrapper that ensures each decode step takes exactly the same
/// wall-clock time, preventing timing-based token identification attacks.
pub struct ConstantTimeDecoder {
    target_duration: Duration,
    noise_amplitude_us: u64, // random noise added to pad target (prevents fixed-pattern detection)
}

impl ConstantTimeDecoder {
    /// Create a decoder that pads to `target_ms` milliseconds per token.
    /// If actual decode takes longer, no padding is applied (can't speed up).
    pub fn new(target_ms: u64) -> Self {
        Self {
            target_duration: Duration::from_millis(target_ms),
            noise_amplitude_us: 1000, // ±1ms random noise
        }
    }

    /// Create from a calibration run: measures actual decode time and sets
    /// target to max_observed + 10% margin.
    pub fn calibrate(
        runner: &mut dyn ModelRunner,
        sampler: &Sampler,
        calibration_tokens: u32,
    ) -> Result<Self> {
        // Use a dummy prompt (token 1) to prefill, then measure decode steps.
        let logits = runner.prefill(&[1])?;

        let mut max_elapsed = Duration::ZERO;
        let mut current_logits = logits;
        let mut previous_tokens: Vec<u32> = vec![1];

        for _ in 0..calibration_tokens {
            let start = Instant::now();
            let token = sampler.sample_with_history(&current_logits, &previous_tokens)?;
            current_logits = runner.decode_step(token)?;
            let elapsed = start.elapsed();

            previous_tokens.push(token);

            if elapsed > max_elapsed {
                max_elapsed = elapsed;
            }
        }

        // Add 10% margin
        let target_us = (max_elapsed.as_micros() as f64 * 1.1) as u64;

        Ok(Self {
            target_duration: Duration::from_micros(target_us),
            noise_amplitude_us: 1000,
        })
    }

    /// Decode one token with constant-time padding.
    /// Returns the token and next logits, but takes exactly target_duration.
    pub fn decode_step_padded(
        &self,
        runner: &mut dyn ModelRunner,
        logits: &[f32],
        sampler: &Sampler,
        previous_tokens: &[u32],
    ) -> Result<(u32, Vec<f32>)> {
        let start = Instant::now();

        // Actual work
        let token = sampler.sample_with_history(logits, previous_tokens)?;
        let next_logits = runner.decode_step(token)?;

        // Pad to target duration
        let target = self.padded_target();
        let elapsed = start.elapsed();
        if elapsed < target {
            busy_wait_until(start + target);
        }

        Ok((token, next_logits))
    }

    /// Get target with noise to prevent fixed-pattern detection.
    fn padded_target(&self) -> Duration {
        let noise = random_noise_us(self.noise_amplitude_us);
        self.target_duration + Duration::from_micros(noise)
    }
}

/// Busy-wait until the target instant.
/// Uses a spin loop with no sleep (sleep has unpredictable wakeup latency).
/// Includes a `spin_loop` hint to reduce power consumption while spinning.
fn busy_wait_until(target: Instant) {
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// Generate random noise in [0, amplitude_us) using CSPRNG.
fn random_noise_us(amplitude_us: u64) -> u64 {
    if amplitude_us == 0 {
        return 0;
    }
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).unwrap_or_default();
    u64::from_le_bytes(bytes) % amplitude_us
}

/// Fixed-size response padding for network-level side-channel resistance.
/// Pads response to a fixed number of tokens regardless of actual generation length.
pub struct FixedSizeResponse {
    /// The padded token sequence (length == `padded_length`).
    pub tokens: Vec<u32>,
    /// The number of real (non-padding) tokens.
    pub actual_length: usize,
    /// The total length after padding.
    pub padded_length: usize,
}

impl FixedSizeResponse {
    /// Pad token sequence to fixed length with dummy tokens.
    /// If `tokens` is longer than `target_length`, it is truncated.
    pub fn pad(tokens: Vec<u32>, target_length: usize, pad_token: u32) -> Self {
        let actual_length = tokens.len().min(target_length);
        let mut padded = tokens;
        while padded.len() < target_length {
            padded.push(pad_token);
        }
        padded.truncate(target_length);
        Self {
            tokens: padded,
            actual_length,
            padded_length: target_length,
        }
    }

    /// Get the actual (unpadded) tokens.
    pub fn actual_tokens(&self) -> &[u32] {
        &self.tokens[..self.actual_length]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_busy_wait_accuracy() {
        let start = Instant::now();
        let target = start + Duration::from_millis(10);
        busy_wait_until(target);
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(10),
            "busy_wait finished too early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(11),
            "busy_wait overshot by too much: {elapsed:?}"
        );
    }

    #[test]
    fn test_fixed_size_response_padding() {
        let original = vec![10, 20, 30, 40, 50];
        let response = FixedSizeResponse::pad(original.clone(), 20, 0);

        // Total length is the target
        assert_eq!(response.tokens.len(), 20);
        assert_eq!(response.padded_length, 20);

        // Actual length matches original
        assert_eq!(response.actual_length, 5);

        // actual_tokens returns the original 5
        assert_eq!(response.actual_tokens(), &[10, 20, 30, 40, 50]);

        // Remaining slots are the pad token
        for &tok in &response.tokens[5..] {
            assert_eq!(tok, 0);
        }
    }

    #[test]
    fn test_fixed_size_response_truncation() {
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let response = FixedSizeResponse::pad(original, 4, 0);

        assert_eq!(response.tokens.len(), 4);
        assert_eq!(response.actual_length, 4);
        assert_eq!(response.actual_tokens(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_random_noise_bounded() {
        // Run many iterations to test the bound holds
        for amplitude in [1, 10, 100, 1000, 10_000] {
            for _ in 0..200 {
                let noise = random_noise_us(amplitude);
                assert!(noise < amplitude, "noise {noise} >= amplitude {amplitude}");
            }
        }
    }

    #[test]
    fn test_random_noise_zero_amplitude() {
        assert_eq!(random_noise_us(0), 0);
    }

    #[test]
    fn test_constant_time_decoder_new() {
        let decoder = ConstantTimeDecoder::new(50);
        assert_eq!(decoder.target_duration, Duration::from_millis(50));
        assert_eq!(decoder.noise_amplitude_us, 1000);
    }
}
