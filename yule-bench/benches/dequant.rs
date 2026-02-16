#![allow(clippy::needless_range_loop)]
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use yule_core::dequant;

fn q4_0_block() -> [u8; 18] {
    let mut b = [0u8; 18];
    b[0] = 0x00;
    b[1] = 0x38; // d = 0.5 f16
    for i in 0..16 {
        b[2 + i] = 0x5A;
    }
    b
}

fn q8_0_block() -> [u8; 34] {
    let mut b = [0u8; 34];
    b[0] = 0x00;
    b[1] = 0x34; // d = 0.25 f16
    for i in 0..32 {
        b[2 + i] = ((i as i8).wrapping_mul(7)) as u8;
    }
    b
}

fn q4_k_block() -> Vec<u8> {
    let mut b = vec![0u8; 144];
    b[0] = 0x00;
    b[1] = 0x38; // d
    b[2] = 0x66;
    b[3] = 0x2E; // dmin
    for i in 0..12 {
        b[4 + i] = 0x05;
    }
    for i in 0..128 {
        b[16 + i] = 0x73;
    }
    b
}

fn q6_k_block() -> Vec<u8> {
    let mut b = vec![0u8; 210];
    b[208] = 0x00;
    b[209] = 0x38; // d
    for i in 0..16 {
        b[192 + i] = 3;
    }
    for i in 0..128 {
        b[i] = 0xA5;
    }
    for i in 0..64 {
        b[128 + i] = 0x55;
    }
    b
}

fn activations(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32) * 0.01 - (n as f32 * 0.005))
        .collect()
}

fn bench_vec_dot(c: &mut Criterion) {
    let q4_0 = q4_0_block();
    let q8_0 = q8_0_block();
    let q4_k = q4_k_block();
    let q6_k = q6_k_block();
    let act32 = activations(32);
    let act256 = activations(256);

    c.bench_function("vec_dot_q4_0 (scalar)", |b| {
        b.iter(|| dequant::vec_dot_q4_0(black_box(&q4_0), black_box(&act32)))
    });
    c.bench_function("vec_dot_q8_0 (scalar)", |b| {
        b.iter(|| dequant::vec_dot_q8_0(black_box(&q8_0), black_box(&act32)))
    });
    c.bench_function("vec_dot_q4_k (scalar)", |b| {
        b.iter(|| dequant::vec_dot_q4_k(black_box(&q4_k), black_box(&act256)))
    });
    c.bench_function("vec_dot_q6_k (scalar)", |b| {
        b.iter(|| dequant::vec_dot_q6_k(black_box(&q6_k), black_box(&act256)))
    });

    // SIMD dispatch (uses AVX2 when available)
    use yule_core::dtype::DType;
    use yule_core::simd;

    c.bench_function("vec_dot_q4_0 (dispatch)", |b| {
        b.iter(|| simd::vec_dot(DType::Q4_0, black_box(&q4_0), black_box(&act32)))
    });
    c.bench_function("vec_dot_q8_0 (dispatch)", |b| {
        b.iter(|| simd::vec_dot(DType::Q8_0, black_box(&q8_0), black_box(&act32)))
    });
    c.bench_function("vec_dot_q4_k (dispatch)", |b| {
        b.iter(|| simd::vec_dot(DType::Q4_K, black_box(&q4_k), black_box(&act256)))
    });
    c.bench_function("vec_dot_q6_k (dispatch)", |b| {
        b.iter(|| simd::vec_dot(DType::Q6_K, black_box(&q6_k), black_box(&act256)))
    });
}

criterion_group!(benches, bench_vec_dot);
criterion_main!(benches);
