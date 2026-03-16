use std::time::Instant;

pub struct BenchResult {
    pub name: String,
    pub iterations: u64,
    pub total_ns: u64,
    pub avg_ns: u64,
    pub throughput_mb_s: Option<f64>,
}

impl BenchResult {
    pub fn avg_ms(&self) -> f64 {
        self.avg_ns as f64 / 1_000_000.0
    }
}

pub fn bench_fn<F: FnMut()>(name: &str, iterations: u64, mut f: F) -> BenchResult {
    // warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total_ns = start.elapsed().as_nanos() as u64;

    BenchResult {
        name: name.to_string(),
        iterations,
        total_ns,
        avg_ns: total_ns / iterations,
        throughput_mb_s: None,
    }
}

pub fn bench_throughput<F: FnMut()>(
    name: &str,
    iterations: u64,
    bytes_per_iter: u64,
    f: F,
) -> BenchResult {
    let mut result = bench_fn(name, iterations, f);
    let total_bytes = bytes_per_iter * iterations;
    let seconds = result.total_ns as f64 / 1_000_000_000.0;
    result.throughput_mb_s = Some(total_bytes as f64 / (1024.0 * 1024.0) / seconds);
    result
}
