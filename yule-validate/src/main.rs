mod correctness;
mod infrastructure;
mod model;
mod performance;
mod report;

use clap::Parser;
use report::{TestResult, print_json, print_report};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "yule-validate", about = "Empirical validation suite for Yule")]
struct Cli {
    #[arg(long, help = "Path to GGUF model file")]
    model_path: Option<String>,

    #[arg(long, help = "Skip performance benchmarks")]
    skip_perf: bool,

    #[arg(long, help = "Only run specific test IDs (comma-separated)")]
    tests: Option<String>,

    #[arg(long, help = "Output results as JSON")]
    json: bool,
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let filter: Option<Vec<u32>> = cli
        .tests
        .as_ref()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());
    let should_run = |id: u32| -> bool { filter.as_ref().is_none_or(|ids| ids.contains(&id)) };

    let mut results: Vec<TestResult> = Vec::new();
    let total_start = Instant::now();

    // Load model
    println!("Resolving model...");
    let model_path = match model::resolve_model_path(cli.model_path.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    println!("Loading model from: {}", model_path.display());

    let loaded = match model::load_model(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Model: {:?}, {} layers, {} vocab",
        loaded.model_info.metadata.architecture,
        loaded.model_info.metadata.layer_count,
        loaded.model_info.metadata.vocab_size,
    );

    // Create runner
    let mut test_runner = match model::create_runner(&model_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to create runner: {e}");
            std::process::exit(1);
        }
    };

    println!();
    println!("Running validation suite...");
    println!();

    // Infrastructure tests (32-37) — don't need model runner
    for r in infrastructure::run_all(&should_run) {
        results.push(r);
    }

    // Correctness tests (1-21) — need model + runner
    for r in correctness::run_all(&loaded, &mut test_runner, &should_run) {
        results.push(r);
    }

    // Performance tests (22-31)
    if !cli.skip_perf {
        for r in performance::run_all(&loaded, &mut test_runner, &model_path, &should_run) {
            results.push(r);
        }
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // Sort by ID
    results.sort_by_key(|r| r.id);

    if cli.json {
        print_json(&results);
    } else {
        print_report(&results);
        println!("Total time: {:.1}s", total_ms / 1000.0);
    }

    let failed = results.iter().any(|r| !r.passed);
    std::process::exit(if failed { 1 } else { 0 });
}
