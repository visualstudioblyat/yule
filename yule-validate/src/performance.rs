use std::collections::HashMap;
use std::time::Instant;

use crate::model::{LoadedTestModel, TestRunner};
use crate::report::{Category, TestResult, compute_stats};

pub fn run_all(
    model: &LoadedTestModel,
    runner: &mut TestRunner,
    model_path: &std::path::Path,
    should_run: &dyn Fn(u32) -> bool,
) -> Vec<TestResult> {
    let mut results = Vec::new();
    let cat = Category::Performance;

    // Test 22: GGUF parse time
    if should_run(22) {
        results.push(test_gguf_parse_time(model_path, cat));
    }

    // Test 23: Tokenization throughput
    if should_run(23) {
        results.push(test_tokenization_throughput(model, cat));
    }

    // Test 24: Prefill throughput
    if should_run(24) {
        results.push(test_prefill_throughput(model, runner, cat));
    }

    // Test 25: Decode throughput
    if should_run(25) {
        results.push(test_decode_throughput(model, runner, cat));
    }

    // Test 26: Memory estimate
    if should_run(26) {
        results.push(test_memory_estimate(model, cat));
    }

    // Test 27: Dequant throughput per format
    if should_run(27) {
        results.push(test_dequant_throughput(cat));
    }

    // Test 28: Huge pages vs regular mmap
    if should_run(28) {
        results.push(test_huge_pages_mmap(model_path, cat));
    }

    // Test 29: Tiled matmul vs naive
    if should_run(29) {
        results.push(test_tiled_matmul_vs_naive(cat));
    }

    // Test 30: Flash attention vs naive
    if should_run(30) {
        results.push(test_flash_attention_vs_naive(cat));
    }

    // Test 31: Prefetch metric
    if should_run(31) {
        results.push(test_prefetch_metric(model, runner, cat));
    }

    results
}

fn test_gguf_parse_time(model_path: &std::path::Path, cat: Category) -> TestResult {
    let start = Instant::now();

    let data = match std::fs::read(model_path) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::fail(22, "GGUF parse time", cat, &format!("read error: {e}"), 0.0);
        }
    };

    let parser = yule_core::gguf::GgufParser::new();
    let iters = 5;
    let mut times_ms = Vec::with_capacity(iters);

    for _ in 0..iters {
        let t = Instant::now();
        match parser.parse_bytes(&data, data.len() as u64) {
            Ok(_) => {}
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    22,
                    "GGUF parse time",
                    cat,
                    &format!("parse error: {e}"),
                    dur,
                );
            }
        }
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let (mean, p50, p99, min, max) = compute_stats(&times_ms);

    let mut metrics = HashMap::new();
    metrics.insert("mean_ms".to_string(), mean);
    metrics.insert("p50_ms".to_string(), p50);
    metrics.insert("p99_ms".to_string(), p99);
    metrics.insert("min_ms".to_string(), min);
    metrics.insert("max_ms".to_string(), max);

    let passed = mean < 500.0;
    let msg = format!("mean={mean:.1}ms over {iters} iters (threshold 500ms)");

    if passed {
        TestResult::pass(22, "GGUF parse time", cat, &msg, dur).with_metrics(metrics)
    } else {
        TestResult::fail(22, "GGUF parse time", cat, &msg, dur).with_metrics(metrics)
    }
}

fn test_tokenization_throughput(model: &LoadedTestModel, cat: Category) -> TestResult {
    use yule_core::tokenizer::Tokenizer;

    let start = Instant::now();

    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let iters = 100;

    // Encode throughput
    let enc_start = Instant::now();
    let mut tokens = Vec::new();
    for _ in 0..iters {
        match model.tokenizer.encode(&long_text) {
            Ok(t) => tokens = t,
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    23,
                    "Tokenization throughput",
                    cat,
                    &format!("encode error: {e}"),
                    dur,
                );
            }
        }
    }
    let enc_elapsed = enc_start.elapsed().as_secs_f64();
    let enc_tokens_per_sec = (tokens.len() as f64 * iters as f64) / enc_elapsed;

    // Decode throughput
    let dec_start = Instant::now();
    for _ in 0..iters {
        match model.tokenizer.decode(&tokens) {
            Ok(_) => {}
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    23,
                    "Tokenization throughput",
                    cat,
                    &format!("decode error: {e}"),
                    dur,
                );
            }
        }
    }
    let dec_elapsed = dec_start.elapsed().as_secs_f64();
    let dec_tokens_per_sec = (tokens.len() as f64 * iters as f64) / dec_elapsed;

    let dur = start.elapsed().as_secs_f64() * 1000.0;

    let mut metrics = HashMap::new();
    metrics.insert("encode_tokens_per_sec".to_string(), enc_tokens_per_sec);
    metrics.insert("decode_tokens_per_sec".to_string(), dec_tokens_per_sec);
    metrics.insert("tokens_per_input".to_string(), tokens.len() as f64);

    TestResult::pass(
        23,
        "Tokenization throughput",
        cat,
        &format!(
            "encode={:.0} tok/s, decode={:.0} tok/s ({} tokens/input)",
            enc_tokens_per_sec,
            dec_tokens_per_sec,
            tokens.len()
        ),
        dur,
    )
    .with_metrics(metrics)
}

fn test_prefill_throughput(
    model: &LoadedTestModel,
    runner: &mut TestRunner,
    cat: Category,
) -> TestResult {
    use yule_core::tokenizer::Tokenizer;

    let start = Instant::now();

    // Encode a short prompt to get ~50 tokens
    let prompt = "The quick brown fox jumps over the lazy dog and then continues running \
                  across the wide open field while the sun sets behind the distant mountains.";
    let tokens = match model.tokenizer.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            return TestResult::fail(
                24,
                "Prefill throughput",
                cat,
                &format!("encode error: {e}"),
                0.0,
            );
        }
    };

    let iters = 3;
    let mut times_ms = Vec::with_capacity(iters);

    for _ in 0..iters {
        runner.runner.reset();
        let t = Instant::now();
        match runner.runner.prefill(&tokens) {
            Ok(_) => {}
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    24,
                    "Prefill throughput",
                    cat,
                    &format!("prefill error: {e}"),
                    dur,
                );
            }
        }
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let (mean, p50, _p99, _min, _max) = compute_stats(&times_ms);
    let tokens_per_sec = tokens.len() as f64 / (mean / 1000.0);

    let mut metrics = HashMap::new();
    metrics.insert("mean_ms".to_string(), mean);
    metrics.insert("p50_ms".to_string(), p50);
    metrics.insert("tokens_per_sec".to_string(), tokens_per_sec);
    metrics.insert("num_tokens".to_string(), tokens.len() as f64);

    TestResult::pass(
        24,
        "Prefill throughput",
        cat,
        &format!(
            "{:.0} tok/s ({} tokens, mean={:.1}ms)",
            tokens_per_sec,
            tokens.len(),
            mean,
        ),
        dur,
    )
    .with_metrics(metrics)
}

fn test_decode_throughput(
    model: &LoadedTestModel,
    runner: &mut TestRunner,
    cat: Category,
) -> TestResult {
    use yule_core::tokenizer::Tokenizer;

    let start = Instant::now();

    // Prefill a short prompt first
    let prompt = "Once upon a time";
    let tokens = match model.tokenizer.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            return TestResult::fail(
                25,
                "Decode throughput",
                cat,
                &format!("encode error: {e}"),
                0.0,
            );
        }
    };

    runner.runner.reset();
    match runner.runner.prefill(&tokens) {
        Ok(_) => {}
        Err(e) => {
            let dur = start.elapsed().as_secs_f64() * 1000.0;
            return TestResult::fail(
                25,
                "Decode throughput",
                cat,
                &format!("prefill error: {e}"),
                dur,
            );
        }
    }

    let decode_steps = 20;
    let mut step_times_ms = Vec::with_capacity(decode_steps);
    let mut last_token = tokens.last().copied().unwrap_or(1);

    for _ in 0..decode_steps {
        let t = Instant::now();
        match runner.runner.decode_step(last_token) {
            Ok(logits) => {
                // Greedy: pick argmax
                if !logits.is_empty() {
                    last_token = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                }
            }
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    25,
                    "Decode throughput",
                    cat,
                    &format!("decode_step error: {e}"),
                    dur,
                );
            }
        }
        step_times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let (mean, p50, p99, min, max) = compute_stats(&step_times_ms);
    let tokens_per_sec = 1000.0 / mean;

    let mut metrics = HashMap::new();
    metrics.insert("mean_ms_per_token".to_string(), mean);
    metrics.insert("p50_ms_per_token".to_string(), p50);
    metrics.insert("p99_ms_per_token".to_string(), p99);
    metrics.insert("min_ms_per_token".to_string(), min);
    metrics.insert("max_ms_per_token".to_string(), max);
    metrics.insert("tokens_per_sec".to_string(), tokens_per_sec);

    TestResult::pass(
        25,
        "Decode throughput",
        cat,
        &format!(
            "{:.1} tok/s, {:.1}ms/tok (p50={:.1}ms, {decode_steps} steps)",
            tokens_per_sec, mean, p50,
        ),
        dur,
    )
    .with_metrics(metrics)
}

fn test_memory_estimate(model: &LoadedTestModel, cat: Category) -> TestResult {
    let start = Instant::now();

    let file_size = model.model_info.file_size;
    let meta = &model.model_info.metadata;

    // KV cache estimate: 2 * num_layers * num_kv_heads * head_dim * context_length * 4 bytes
    let head_dim = if meta.head_count > 0 {
        meta.embedding_dim / meta.head_count
    } else {
        128
    };
    let kv_bytes = 2u64
        * meta.layer_count as u64
        * meta.head_count_kv as u64
        * head_dim as u64
        * meta.context_length as u64
        * 4;

    // System RAM via CpuBackend
    let backend = yule_gpu::cpu::CpuBackend::new();
    let info = <yule_gpu::cpu::CpuBackend as yule_gpu::ComputeBackend>::device_info(&backend);
    let system_ram = info.memory_bytes;

    let dur = start.elapsed().as_secs_f64() * 1000.0;

    let total_required = file_size + kv_bytes;
    let passed = system_ram > 0 && total_required < system_ram;

    let mut metrics = HashMap::new();
    metrics.insert(
        "model_file_mb".to_string(),
        file_size as f64 / (1024.0 * 1024.0),
    );
    metrics.insert(
        "kv_cache_mb".to_string(),
        kv_bytes as f64 / (1024.0 * 1024.0),
    );
    metrics.insert(
        "system_ram_mb".to_string(),
        system_ram as f64 / (1024.0 * 1024.0),
    );
    metrics.insert(
        "total_required_mb".to_string(),
        total_required as f64 / (1024.0 * 1024.0),
    );

    let msg = format!(
        "model={:.0}MB + kv={:.0}MB = {:.0}MB, system RAM={:.0}MB",
        file_size as f64 / (1024.0 * 1024.0),
        kv_bytes as f64 / (1024.0 * 1024.0),
        total_required as f64 / (1024.0 * 1024.0),
        system_ram as f64 / (1024.0 * 1024.0),
    );

    if passed {
        TestResult::pass(26, "Memory estimate", cat, &msg, dur).with_metrics(metrics)
    } else {
        TestResult::fail(26, "Memory estimate", cat, &msg, dur).with_metrics(metrics)
    }
}

fn test_dequant_throughput(cat: Category) -> TestResult {
    use yule_core::dequant;
    use yule_core::dtype::DType;

    let start = Instant::now();

    let formats = [
        (DType::Q4_0, "Q4_0"),
        (DType::Q4_K, "Q4_K"),
        (DType::Q6_K, "Q6_K"),
        (DType::Q8_0, "Q8_0"),
    ];

    let target_bytes: usize = 1024 * 1024; // 1MB
    let iters = 50;
    let mut metrics = HashMap::new();

    for (dtype, name) in &formats {
        let block_bytes = dtype.size_of_block();
        let block_elements = dtype.block_size();
        let num_blocks = target_bytes / block_bytes;
        let data_size = num_blocks * block_bytes;

        // Create fake quantized data (zeros are fine for throughput measurement)
        let data = vec![0u8; data_size];
        let mut out = vec![0.0f32; num_blocks * block_elements];

        let t = Instant::now();
        for _ in 0..iters {
            for b in 0..num_blocks {
                let block = &data[b * block_bytes..(b + 1) * block_bytes];
                let out_slice = &mut out[b * block_elements..(b + 1) * block_elements];
                let _ = dequant::dequant_block(*dtype, block, out_slice);
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        let mb_per_sec = (data_size as f64 * iters as f64) / (1024.0 * 1024.0) / elapsed;

        metrics.insert(format!("{name}_mb_per_sec"), mb_per_sec);
    }

    let dur = start.elapsed().as_secs_f64() * 1000.0;

    let summary: Vec<String> = formats
        .iter()
        .map(|(_, name)| {
            let key = format!("{name}_mb_per_sec");
            let val = metrics.get(&key).unwrap_or(&0.0);
            format!("{name}={val:.0}MB/s")
        })
        .collect();

    TestResult::pass(27, "Dequant throughput", cat, &summary.join(", "), dur).with_metrics(metrics)
}

fn test_huge_pages_mmap(model_path: &std::path::Path, cat: Category) -> TestResult {
    use yule_core::mmap::{MmapOptions, mmap_model, mmap_model_with};

    let start = Instant::now();

    // Default mmap (may use huge pages on Linux)
    let t1 = Instant::now();
    let _default_map = match mmap_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            let dur = start.elapsed().as_secs_f64() * 1000.0;
            return TestResult::fail(
                28,
                "Huge pages vs regular mmap",
                cat,
                &format!("default mmap error: {e}"),
                dur,
            );
        }
    };
    let default_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Regular mmap (huge_pages=false)
    let t2 = Instant::now();
    let _regular_map = match mmap_model_with(
        model_path,
        &MmapOptions {
            huge_pages: false,
            prefault: false,
            sequential: true,
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            let dur = start.elapsed().as_secs_f64() * 1000.0;
            return TestResult::fail(
                28,
                "Huge pages vs regular mmap",
                cat,
                &format!("regular mmap error: {e}"),
                dur,
            );
        }
    };
    let regular_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let dur = start.elapsed().as_secs_f64() * 1000.0;

    let mut metrics = HashMap::new();
    metrics.insert("default_ms".to_string(), default_ms);
    metrics.insert("regular_ms".to_string(), regular_ms);

    TestResult::pass(
        28,
        "Huge pages vs regular mmap",
        cat,
        &format!("default={default_ms:.1}ms, regular={regular_ms:.1}ms"),
        dur,
    )
    .with_metrics(metrics)
}

fn test_tiled_matmul_vs_naive(cat: Category) -> TestResult {
    use yule_gpu::ComputeBackend;

    let start = Instant::now();

    let n = 128usize;
    let backend = yule_gpu::cpu::CpuBackend::new();

    // Create deterministic matrices
    let a_data: Vec<f32> = (0..n * n)
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..n * n)
        .map(|i| ((i * 13 + 5) % 100) as f32 * 0.01)
        .collect();

    // Tiled matmul via CpuBackend
    let a_handle = backend.allocate(n * n * 4).unwrap();
    let b_handle = backend.allocate(n * n * 4).unwrap();
    let out_handle = backend.allocate(n * n * 4).unwrap();
    backend
        .copy_to_device(bytemuck::cast_slice(&a_data), &a_handle)
        .unwrap();
    backend
        .copy_to_device(bytemuck::cast_slice(&b_data), &b_handle)
        .unwrap();

    let t_tiled = Instant::now();
    backend
        .matmul(
            &a_handle,
            &b_handle,
            &out_handle,
            n as u32,
            n as u32,
            n as u32,
        )
        .unwrap();
    let tiled_ms = t_tiled.elapsed().as_secs_f64() * 1000.0;

    // Naive triple loop
    let t_naive = Instant::now();
    let mut naive_out = vec![0.0f32; n * n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a_data[i * n + k] * b_data[k * n + j];
            }
            naive_out[i * n + j] = sum;
        }
    }
    let naive_ms = t_naive.elapsed().as_secs_f64() * 1000.0;

    // Suppress unused variable warning
    let _ = &naive_out;

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let ratio = if tiled_ms > 0.0 {
        naive_ms / tiled_ms
    } else {
        f64::NAN
    };

    let mut metrics = HashMap::new();
    metrics.insert("tiled_ms".to_string(), tiled_ms);
    metrics.insert("naive_ms".to_string(), naive_ms);
    metrics.insert("ratio".to_string(), ratio);

    TestResult::pass(
        29,
        "Tiled matmul vs naive",
        cat,
        &format!("tiled={tiled_ms:.2}ms, naive={naive_ms:.2}ms, ratio={ratio:.2}x"),
        dur,
    )
    .with_metrics(metrics)
}

fn test_flash_attention_vs_naive(cat: Category) -> TestResult {
    use yule_gpu::ComputeBackend;
    use yule_infer::attention::{FlashAttention, FlashAttentionParams};

    let start = Instant::now();

    let backend = yule_gpu::cpu::CpuBackend::new();
    let nh = 4usize;
    let nkv = 4usize;
    let sq = 64usize;
    let skv = 64usize;
    let hd = 32usize;

    // Generate data
    let q_data: Vec<f32> = (0..nh * sq * hd).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let k_data: Vec<f32> = (0..nkv * skv * hd)
        .map(|i| (i as f32 * 0.013) - 0.3)
        .collect();
    let v_data: Vec<f32> = (0..nkv * skv * hd)
        .map(|i| (i as f32 * 0.02) + 1.0)
        .collect();

    let q_handle = backend.allocate(q_data.len() * 4).unwrap();
    let k_handle = backend.allocate(k_data.len() * 4).unwrap();
    let v_handle = backend.allocate(v_data.len() * 4).unwrap();

    backend
        .copy_to_device(bytemuck::cast_slice(&q_data), &q_handle)
        .unwrap();
    backend
        .copy_to_device(bytemuck::cast_slice(&k_data), &k_handle)
        .unwrap();
    backend
        .copy_to_device(bytemuck::cast_slice(&v_data), &v_handle)
        .unwrap();

    let flash = FlashAttention::new(hd as u32);
    let params = FlashAttentionParams {
        num_heads: nh as u32,
        num_kv_heads: nkv as u32,
        seq_len_q: sq as u32,
        seq_len_kv: skv as u32,
        head_dim: hd as u32,
        causal: true,
    };

    // Flash attention timing
    let t_flash = Instant::now();
    match flash.forward(&backend, &q_handle, &k_handle, &v_handle, &params) {
        Ok(_) => {}
        Err(e) => {
            let dur = start.elapsed().as_secs_f64() * 1000.0;
            return TestResult::fail(
                30,
                "Flash attention vs naive",
                cat,
                &format!("flash forward error: {e}"),
                dur,
            );
        }
    }
    let flash_ms = t_flash.elapsed().as_secs_f64() * 1000.0;

    // Naive attention timing
    let t_naive = Instant::now();
    for h in 0..nh {
        let kv_h = h / (nh / nkv);
        let q_head = &q_data[h * sq * hd..(h + 1) * sq * hd];
        let k_head = &k_data[kv_h * skv * hd..(kv_h + 1) * skv * hd];
        let v_head = &v_data[kv_h * skv * hd..(kv_h + 1) * skv * hd];
        let _out = naive_attention(q_head, k_head, v_head, sq, skv, hd, true);
    }
    let naive_ms = t_naive.elapsed().as_secs_f64() * 1000.0;

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let ratio = if flash_ms > 0.0 {
        naive_ms / flash_ms
    } else {
        f64::NAN
    };

    let mut metrics = HashMap::new();
    metrics.insert("flash_ms".to_string(), flash_ms);
    metrics.insert("naive_ms".to_string(), naive_ms);
    metrics.insert("ratio".to_string(), ratio);

    TestResult::pass(
        30,
        "Flash attention vs naive",
        cat,
        &format!("flash={flash_ms:.2}ms, naive={naive_ms:.2}ms, ratio={ratio:.2}x (seq_len={sq})"),
        dur,
    )
    .with_metrics(metrics)
}

/// Naive attention for reference: softmax(Q @ K^T / sqrt(hd)) @ V
#[allow(clippy::needless_range_loop)]
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    skv: usize,
    hd: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; sq * hd];

    for i in 0..sq {
        let mut scores = vec![0.0f32; skv];
        for j in 0..skv {
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            scores[j] = dot * scale;
        }

        if causal {
            for j in 0..skv {
                if j > i {
                    scores[j] = f32::NEG_INFINITY;
                }
            }
        }

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut exp_scores = vec![0.0f32; skv];
        for j in 0..skv {
            exp_scores[j] = (scores[j] - max_s).exp();
            sum += exp_scores[j];
        }
        if sum > 0.0 {
            for j in 0..skv {
                exp_scores[j] /= sum;
            }
        }

        for d in 0..hd {
            let mut val = 0.0f32;
            for j in 0..skv {
                val += exp_scores[j] * v[j * hd + d];
            }
            out[i * hd + d] = val;
        }
    }
    out
}

fn test_prefetch_metric(
    model: &LoadedTestModel,
    runner: &mut TestRunner,
    cat: Category,
) -> TestResult {
    use yule_core::tokenizer::Tokenizer;

    let start = Instant::now();

    let prompt = "Hello world";
    let tokens = match model.tokenizer.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            return TestResult::fail(
                31,
                "Prefetch metric",
                cat,
                &format!("encode error: {e}"),
                0.0,
            );
        }
    };

    runner.runner.reset();
    match runner.runner.prefill(&tokens) {
        Ok(_) => {}
        Err(e) => {
            let dur = start.elapsed().as_secs_f64() * 1000.0;
            return TestResult::fail(
                31,
                "Prefetch metric",
                cat,
                &format!("prefill error: {e}"),
                dur,
            );
        }
    }

    let decode_steps = 10;
    let mut step_times_ms = Vec::with_capacity(decode_steps);
    let mut last_token = tokens.last().copied().unwrap_or(1);

    for _ in 0..decode_steps {
        let t = Instant::now();
        match runner.runner.decode_step(last_token) {
            Ok(logits) => {
                if !logits.is_empty() {
                    last_token = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                }
            }
            Err(e) => {
                let dur = start.elapsed().as_secs_f64() * 1000.0;
                return TestResult::fail(
                    31,
                    "Prefetch metric",
                    cat,
                    &format!("decode_step error: {e}"),
                    dur,
                );
            }
        }
        step_times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let dur = start.elapsed().as_secs_f64() * 1000.0;
    let (mean, p50, p99, min, max) = compute_stats(&step_times_ms);

    let mut metrics = HashMap::new();
    metrics.insert("mean_ms_per_token".to_string(), mean);
    metrics.insert("p50_ms_per_token".to_string(), p50);
    metrics.insert("p99_ms_per_token".to_string(), p99);
    metrics.insert("min_ms_per_token".to_string(), min);
    metrics.insert("max_ms_per_token".to_string(), max);

    TestResult::pass(
        31,
        "Prefetch metric",
        cat,
        &format!("{mean:.1}ms/tok (p50={p50:.1}ms, prefetch always active on x86)"),
        dur,
    )
    .with_metrics(metrics)
}
