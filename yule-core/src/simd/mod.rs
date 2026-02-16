pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;

use crate::dtype::DType;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

static LEVEL: AtomicU8 = AtomicU8::new(0);

const SCALAR: u8 = 1;
const AVX2: u8 = 2;

fn dispatch_level() -> u8 {
    let l = LEVEL.load(Relaxed);
    if l != 0 {
        return l;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let detected = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        AVX2
    } else {
        SCALAR
    };

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let detected = SCALAR;

    LEVEL.store(detected, Relaxed);
    detected
}

/// Fused dot product with SIMD dispatch. Falls back to scalar if no
/// accelerated kernel exists for the given dtype.
pub fn vec_dot(dtype: DType, block: &[u8], act: &[f32]) -> Option<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if dispatch_level() == AVX2 {
        if let Some(v) = unsafe { avx2::vec_dot_dispatch(dtype, block, act) } {
            return Some(v);
        }
    }

    scalar::vec_dot_dispatch(dtype, block, act)
}
