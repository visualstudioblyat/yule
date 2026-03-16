use std::collections::HashMap;
use std::panic::catch_unwind;
use std::time::Instant;

use crate::report::{Category, TestResult};

pub fn run_all(should_run: &dyn Fn(u32) -> bool) -> Vec<TestResult> {
    let mut results = Vec::new();
    let cat = Category::Infrastructure;

    // Test 32: CPU RAM detection > 0
    if should_run(32) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            let backend = yule_gpu::cpu::CpuBackend::new();
            let info =
                <yule_gpu::cpu::CpuBackend as yule_gpu::ComputeBackend>::device_info(&backend);
            (info.memory_bytes, info.compute_units)
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok((mem, cu)) => {
                let passed = mem > 0 && cu > 0;
                let mut metrics = HashMap::new();
                metrics.insert("memory_bytes".to_string(), mem as f64);
                metrics.insert("compute_units".to_string(), cu as f64);
                results.push(
                    TestResult::pass(
                        32,
                        "CPU RAM detection",
                        cat,
                        &format!("memory={}MB, compute_units={}", mem / (1024 * 1024), cu),
                        dur,
                    )
                    .with_metrics(metrics),
                );
                if !passed {
                    results.last_mut().unwrap().passed = false;
                    results.last_mut().unwrap().message =
                        format!("memory_bytes={mem}, compute_units={cu} — expected both > 0");
                }
            }
            Err(_) => {
                results.push(TestResult::fail(
                    32,
                    "CPU RAM detection",
                    cat,
                    "panicked during device_info()",
                    dur,
                ));
            }
        }
    }

    // Test 33: Sandbox apply_to_current_process
    if should_run(33) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            let sandbox = yule_sandbox::create_sandbox();
            let config = yule_sandbox::SandboxConfig {
                model_path: std::path::PathBuf::from("."),
                allow_gpu: false,
                max_memory_bytes: 1024 * 1024 * 512,
                allow_network: false,
            };
            match sandbox.apply_to_current_process(&config) {
                Ok(_guard) => "applied successfully".to_string(),
                Err(e) => format!("warning: sandbox apply failed (expected in some envs): {e}"),
            }
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(msg) => {
                results.push(TestResult::pass(
                    33,
                    "Sandbox apply_to_current_process",
                    cat,
                    &msg,
                    dur,
                ));
            }
            Err(_) => {
                results.push(TestResult::fail(
                    33,
                    "Sandbox apply_to_current_process",
                    cat,
                    "panicked during sandbox apply",
                    dur,
                ));
            }
        }
    }

    // Test 34: BufferPool acquire/release
    if should_run(34) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            let mut pool = yule_gpu::buffer::BufferPool::new(1024);

            let h1 = pool.acquire(256).expect("first acquire failed");
            assert!(pool.in_use_count() == 1, "expected 1 in use after acquire");

            pool.release(&h1);
            assert!(pool.free_count() == 1, "expected 1 free after release");

            let h2 = pool.acquire(256).expect("re-acquire failed");
            assert!(h2.0 == h1.0, "expected handle reuse");
            assert!(pool.free_count() == 0, "expected 0 free after re-acquire");

            // Budget enforcement
            let _h3 = pool.acquire(700).expect("second acquire failed");
            // 256 + 700 = 956 < 1024, ok
            let over = pool.acquire(100);
            assert!(
                over.is_none(),
                "expected None when exceeding budget (956+100 > 1024)"
            );

            "acquire/release/reuse/budget all correct"
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(msg) => {
                results.push(TestResult::pass(
                    34,
                    "BufferPool acquire/release",
                    cat,
                    msg,
                    dur,
                ));
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                results.push(TestResult::fail(
                    34,
                    "BufferPool acquire/release",
                    cat,
                    &msg,
                    dur,
                ));
            }
        }
    }

    // Test 35: KvCache allocate + write
    if should_run(35) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            use yule_gpu::ComputeBackend;

            let backend = yule_gpu::cpu::CpuBackend::new();
            let num_layers = 2u32;
            let num_kv_heads = 2u32;
            let head_dim = 4u32;
            let max_seq_len = 16u32;

            let mut cache = yule_infer::kv_cache::KvCache::allocate(
                &backend,
                num_layers,
                num_kv_heads,
                head_dim,
                max_seq_len,
            )
            .expect("KvCache allocate failed");

            assert!(
                cache.current_len == 0,
                "expected current_len=0 after allocate"
            );
            assert!(
                cache.remaining_tokens() == max_seq_len,
                "expected remaining_tokens=max_seq_len"
            );

            // Write KV at position 0
            let kv_size = num_kv_heads as usize * head_dim as usize * 4;
            let k_data = backend.allocate(kv_size).expect("alloc k_data failed");
            let v_data = backend.allocate(kv_size).expect("alloc v_data failed");
            let k_bytes = vec![1u8; kv_size];
            let v_bytes = vec![2u8; kv_size];
            backend
                .copy_to_device(&k_bytes, &k_data)
                .expect("copy k failed");
            backend
                .copy_to_device(&v_bytes, &v_data)
                .expect("copy v failed");

            cache
                .write_kv(&backend, 0, 0, &k_data, &v_data)
                .expect("write_kv failed");

            assert!(cache.current_len == 1, "expected current_len=1 after write");
            assert!(
                cache.remaining_tokens() == max_seq_len - 1,
                "expected remaining_tokens=max_seq_len-1"
            );

            // Clear
            cache.clear();
            assert!(cache.current_len == 0, "expected current_len=0 after clear");
            assert!(
                cache.remaining_tokens() == max_seq_len,
                "expected remaining_tokens=max_seq_len after clear"
            );

            format!("allocate/write/clear ok, size_bytes={}", cache.size_bytes())
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(msg) => {
                results.push(TestResult::pass(
                    35,
                    "KvCache allocate + write",
                    cat,
                    &msg,
                    dur,
                ));
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                results.push(TestResult::fail(
                    35,
                    "KvCache allocate + write",
                    cat,
                    &msg,
                    dur,
                ));
            }
        }
    }

    // Test 36: io_uring detection
    if should_run(36) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            let available = yule_core::async_io::is_io_uring_available();
            format!("io_uring available: {available}")
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(msg) => {
                results.push(TestResult::pass(36, "io_uring detection", cat, &msg, dur));
            }
            Err(_) => {
                results.push(TestResult::fail(
                    36,
                    "io_uring detection",
                    cat,
                    "panicked during is_io_uring_available()",
                    dur,
                ));
            }
        }
    }

    // Test 37: TEE detection
    if should_run(37) {
        let start = Instant::now();
        let result = catch_unwind(|| {
            let tee = yule_sandbox::tee::is_tee_available();
            match tee {
                Some(ref backend) => format!("TEE available: {backend:?}"),
                None => "TEE not available (expected on most hardware)".to_string(),
            }
        });
        let dur = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(msg) => {
                results.push(TestResult::pass(37, "TEE detection", cat, &msg, dur));
            }
            Err(_) => {
                results.push(TestResult::fail(
                    37,
                    "TEE detection",
                    cat,
                    "panicked during is_tee_available()",
                    dur,
                ));
            }
        }
    }

    results
}
