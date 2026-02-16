use criterion::{Criterion, criterion_group, criterion_main};

fn merkle_tree_benchmark(c: &mut Criterion) {
    let data = vec![0u8; 1024 * 1024 * 100]; // 100MB
    c.bench_function("merkle_tree_100mb", |b| {
        b.iter(|| {
            let tree = yule_verify::merkle::MerkleTree::new();
            tree.build(&data)
        })
    });
}

fn softmax_benchmark(c: &mut Criterion) {
    let logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001).collect();
    c.bench_function("softmax_32k_vocab", |b| {
        b.iter(|| {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let _probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        })
    });
}

criterion_group!(benches, merkle_tree_benchmark, softmax_benchmark);
criterion_main!(benches);
