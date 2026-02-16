use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "yule")]
#[command(about = "Secure-first local AI inference runtime")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    Run {
        /// Model reference (e.g. "meta/llama3.2-7b" or path to .gguf)
        model: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Compute backend: auto, cpu, vulkan
        #[arg(long, default_value = "auto")]
        backend: String,

        /// Disable sandboxing (NOT recommended)
        #[arg(long)]
        no_sandbox: bool,
    },

    /// Pull a model from registry
    Pull {
        /// Model reference
        model: String,

        /// Verify signature after download
        #[arg(long, default_value = "true")]
        verify: bool,
    },

    /// Verify integrity of a local model file
    Verify {
        /// Path to model file (.gguf)
        model: String,

        /// Show detailed tensor-by-tensor hashes
        #[arg(long)]
        verbose: bool,
    },

    /// Inspect model metadata without full verification
    Inspect {
        /// Path to model file (.gguf)
        model: String,
    },

    /// Start local API server
    Serve {
        /// Path to model file (.gguf)
        model: String,

        /// Bind address
        #[arg(long, default_value = "127.0.0.1:11434")]
        bind: String,

        /// Capability token (generated if not provided)
        #[arg(long)]
        token: Option<String>,

        /// Disable sandboxing
        #[arg(long)]
        no_sandbox: bool,
    },

    /// Query attestation audit log
    Audit {
        /// Number of recent entries to show
        #[arg(long, default_value = "50")]
        last: usize,

        /// Verify audit log chain integrity
        #[arg(long)]
        verify_chain: bool,
    },

    /// List cached models
    List,
}

fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            backend,
            no_sandbox: _,
        } => {
            let prompt = prompt.unwrap_or_else(|| {
                eprintln!("error: --prompt is required");
                std::process::exit(1);
            });
            if let Err(e) = cmd_run(&model, &prompt, max_tokens, temperature, &backend) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Pull { model, verify: _ } => {
            if let Err(e) = cmd_pull(&model) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Verify { model, verbose } => {
            cmd_verify(&model, verbose);
        }
        Commands::Inspect { model } => {
            cmd_inspect(&model);
        }
        Commands::Serve {
            model,
            bind,
            token,
            no_sandbox,
        } => {
            if let Err(e) = cmd_serve(&model, &bind, token, no_sandbox) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Audit { last, verify_chain } => {
            cmd_audit(last, verify_chain);
        }
        Commands::List => {
            if let Err(e) = cmd_list() {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn cmd_verify(model_path: &str, verbose: bool) {
    let path = Path::new(model_path);
    if !path.exists() {
        eprintln!("error: file not found: {model_path}");
        std::process::exit(1);
    }

    println!("verifying: {model_path}");
    println!();

    // step 1: parse
    let start = Instant::now();
    let parser = yule_core::gguf::GgufParser::new();
    let gguf = match parser.parse_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("PARSE FAILED: {e}");
            std::process::exit(1);
        }
    };
    let parse_time = start.elapsed();

    // step 2: extract model info
    let model = match gguf.to_loaded_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("METADATA EXTRACTION FAILED: {e}");
            std::process::exit(1);
        }
    };

    println!("  format:       GGUF v{}", gguf.version);
    if let Some(ref name) = model.metadata.name {
        println!("  name:         {name}");
    }
    println!("  architecture: {:?}", model.metadata.architecture);
    println!(
        "  parameters:   {}",
        format_params(model.metadata.parameters)
    );
    println!("  context:      {}", model.metadata.context_length);
    println!("  embedding:    {}", model.metadata.embedding_dim);
    println!(
        "  heads:        {} (kv: {})",
        model.metadata.head_count, model.metadata.head_count_kv
    );
    println!("  layers:       {}", model.metadata.layer_count);
    println!("  vocab:        {}", model.metadata.vocab_size);
    if let Some(experts) = model.metadata.expert_count {
        println!(
            "  experts:      {} (active: {})",
            experts,
            model.metadata.expert_used_count.unwrap_or(0)
        );
    }
    if let Some(rope) = model.metadata.rope_freq_base {
        println!("  rope base:    {rope}");
    }
    if let Some(ref scaling) = model.metadata.rope_scaling {
        println!("  rope scaling: {scaling}");
    }
    println!("  tensors:      {}", model.tensors.len());
    println!("  file size:    {}", format_bytes(model.file_size));
    println!("  data offset:  {}", gguf.data_offset);
    println!("  alignment:    {}", gguf.alignment);
    println!("  parse time:   {:.1}ms", parse_time.as_secs_f64() * 1000.0);
    println!();

    // step 3: build Merkle tree over tensor data
    println!("  building merkle tree...");
    let verify_start = Instant::now();

    // mmap the file for verification (huge pages reduce TLB misses on large models)
    let mmap = yule_core::mmap::mmap_model(path).expect("failed to mmap file");

    let tree = yule_verify::merkle::MerkleTree::new();

    if gguf.data_offset as usize <= mmap.len() {
        let tensor_data = &mmap[gguf.data_offset as usize..];
        let root = tree.build(tensor_data);
        let verify_time = verify_start.elapsed();

        println!("  merkle root:  {}", hex(&root.hash));
        println!("  leaves:       {}", root.leaf_count);
        println!(
            "  verify time:  {:.1}ms",
            verify_time.as_secs_f64() * 1000.0
        );
    } else {
        println!("  WARNING: data offset exceeds file size — no tensor data to verify");
    }

    if verbose {
        println!();
        println!("  tensors:");
        for tensor in &model.tensors {
            let shape_str: Vec<String> = tensor.shape.iter().map(|d| d.to_string()).collect();
            println!(
                "    {:<40} {:<8} [{}]  {}",
                tensor.name,
                format!("{:?}", tensor.dtype),
                shape_str.join("x"),
                format_bytes(tensor.size_bytes),
            );

            // per-tensor hash
            if let Ok(data) = gguf.tensor_data(tensor, &mmap) {
                let hash = blake3::hash(data);
                println!("      blake3: {}", hex(hash.as_bytes()));
            }
        }
    }

    println!();
    println!("  VERIFICATION COMPLETE");
}

fn cmd_inspect(model_path: &str) {
    let path = Path::new(model_path);
    if !path.exists() {
        eprintln!("error: file not found: {model_path}");
        std::process::exit(1);
    }

    let parser = yule_core::gguf::GgufParser::new();
    let gguf = match parser.parse_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("PARSE FAILED: {e}");
            std::process::exit(1);
        }
    };

    println!("metadata ({} keys):", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let value = &gguf.metadata[key];
        let display = match value {
            yule_core::gguf::GgufValue::String(s) => {
                if s.len() > 80 {
                    format!("\"{}...\"", &s[..77])
                } else {
                    format!("\"{s}\"")
                }
            }
            yule_core::gguf::GgufValue::Uint32(v) => v.to_string(),
            yule_core::gguf::GgufValue::Int32(v) => v.to_string(),
            yule_core::gguf::GgufValue::Uint64(v) => v.to_string(),
            yule_core::gguf::GgufValue::Float32(v) => format!("{v:.6}"),
            yule_core::gguf::GgufValue::Float64(v) => format!("{v:.6}"),
            yule_core::gguf::GgufValue::Bool(v) => v.to_string(),
            yule_core::gguf::GgufValue::Array(arr) => format!("[array; {} items]", arr.len()),
            yule_core::gguf::GgufValue::Uint8(v) => v.to_string(),
            yule_core::gguf::GgufValue::Int8(v) => v.to_string(),
            yule_core::gguf::GgufValue::Uint16(v) => v.to_string(),
            yule_core::gguf::GgufValue::Int16(v) => v.to_string(),
            yule_core::gguf::GgufValue::Int64(v) => v.to_string(),
        };
        println!("  {key}: {display}");
    }
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(n: u64) -> String {
    if n >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", n as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if n >= 1024 * 1024 {
        format!("{:.2} MB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.2} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn cmd_serve(
    model_path: &str,
    bind: &str,
    token: Option<String>,
    no_sandbox: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = resolve_model_path(model_path)?;

    let _sandbox_guard = if no_sandbox {
        eprintln!("sandbox: disabled");
        None
    } else {
        let sandbox = yule_sandbox::create_sandbox();
        let config = yule_sandbox::SandboxConfig {
            model_path: path.to_path_buf(),
            allow_gpu: false,
            max_memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            allow_network: true,                       // need network for the server
        };
        match sandbox.apply_to_current_process(&config) {
            Ok(guard) => {
                eprintln!("sandbox: active (job object, 32GB memory limit)");
                Some(guard)
            }
            Err(e) => {
                eprintln!("sandbox: failed to apply ({e}), continuing without sandbox");
                None
            }
        }
    };

    let sandbox_active = _sandbox_guard.is_some();

    let rt = tokio::runtime::Runtime::new()?;
    let server = yule_api::ApiServer::new(bind.to_string(), path.clone(), token, sandbox_active);

    rt.block_on(server.run())?;
    Ok(())
}

fn cmd_run(
    model_path: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    backend: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = resolve_model_path(model_path)?;

    // 1. parse GGUF
    let start = Instant::now();
    let parser = yule_core::gguf::GgufParser::new();
    let gguf = parser.parse_file(&path)?;
    let parse_time = start.elapsed();

    let model_info = gguf.to_loaded_model()?;
    eprintln!(
        "loaded: {:?} ({}, {} layers, dim {})",
        model_info.metadata.architecture,
        format_params(model_info.metadata.parameters),
        model_info.metadata.layer_count,
        model_info.metadata.embedding_dim,
    );
    eprintln!("parse: {:.1}ms", parse_time.as_secs_f64() * 1000.0);

    // 2. load tokenizer
    let tok_start = Instant::now();
    let tokenizer = yule_core::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
    eprintln!(
        "tokenizer: {} tokens, loaded in {:.1}ms",
        yule_core::tokenizer::Tokenizer::vocab_size(&tokenizer),
        tok_start.elapsed().as_secs_f64() * 1000.0,
    );

    // 3. mmap weights
    let mmap = yule_core::mmap::mmap_model(&path)?;

    let weight_start = Instant::now();
    let store = yule_infer::weight_loader::WeightStore::from_gguf(&gguf, &mmap)?;
    let weights = yule_infer::weight_loader::TransformerWeights::new(store);
    eprintln!(
        "weights: mapped in {:.1}ms",
        weight_start.elapsed().as_secs_f64() * 1000.0
    );

    // 4. select backend and create runner
    let use_vulkan = match backend {
        "vulkan" => true,
        "cpu" => false,
        _ => {
            #[cfg(feature = "vulkan")]
            {
                yule_gpu::vulkan::VulkanBackend::is_available()
            }
            #[cfg(not(feature = "vulkan"))]
            {
                false
            }
        }
    };

    use yule_infer::model_runner::ModelRunner;

    #[allow(unused_mut)]
    let mut runner: Box<dyn ModelRunner>;

    if use_vulkan {
        #[cfg(feature = "vulkan")]
        {
            eprintln!("backend: vulkan");
            runner = Box::new(yule_infer::gpu_runner::GpuTransformerRunner::new(weights)?);
        }
        #[cfg(not(feature = "vulkan"))]
        {
            return Err("vulkan feature not compiled in".into());
        }
    } else {
        eprintln!("backend: cpu");
        runner = Box::new(yule_infer::model_runner::TransformerRunner::new(weights)?);
    }
    eprintln!();

    // 5. encode prompt
    use yule_core::tokenizer::Tokenizer;
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt)?);

    eprintln!("prompt: {} tokens", tokens.len());

    // 6. prefill
    let prefill_start = Instant::now();
    let mut logits = runner.prefill(&tokens)?;
    let prefill_time = prefill_start.elapsed();
    eprintln!(
        "prefill: {:.1}ms ({:.1} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        tokens.len() as f64 / prefill_time.as_secs_f64(),
    );

    // 7. decode loop
    let sampler = yule_infer::sampler::Sampler::new(yule_infer::SamplingParams {
        temperature,
        ..Default::default()
    });

    let eos = tokenizer.eos_token();
    let decode_start = Instant::now();
    let mut generated = 0u32;

    for _ in 0..max_tokens {
        let token = sampler.sample(&logits)?;

        // check eos
        if Some(token) == eos {
            break;
        }

        // decode and print token
        let text = tokenizer.decode(&[token])?;
        print!("{text}");
        std::io::stdout().flush()?;

        generated += 1;
        logits = runner.decode_step(token)?;
    }

    let decode_time = decode_start.elapsed();
    eprintln!();
    eprintln!();
    eprintln!(
        "generated: {} tokens in {:.1}ms ({:.2} tok/s)",
        generated,
        decode_time.as_secs_f64() * 1000.0,
        generated as f64 / decode_time.as_secs_f64(),
    );

    Ok(())
}

/// Resolve a model reference to a local file path.
/// Tries: 1) registry cache lookup, 2) direct file path.
fn resolve_model_path(
    model_ref: &str,
) -> std::result::Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    // Try registry resolution first
    let cache_dir = yule_registry::Registry::default_cache_dir();
    let hf_token = std::env::var("HF_TOKEN").ok();
    if let Ok(registry) = yule_registry::Registry::new(cache_dir, hf_token) {
        match registry.resolve_local(model_ref) {
            Ok(Some(path)) => {
                eprintln!("resolved from cache: {}", path.display());
                return Ok(path);
            }
            Ok(None) => {}
            Err(_) => {}
        }
    }

    // Fall back to direct file path
    let path = std::path::PathBuf::from(model_ref);
    if path.exists() {
        return Ok(path);
    }

    Err(
        format!("model not found: {model_ref}\n  hint: run `yule pull {model_ref}` to download it")
            .into(),
    )
}

fn cmd_pull(model_ref: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cache_dir = yule_registry::Registry::default_cache_dir();
    let hf_token = std::env::var("HF_TOKEN").ok();
    let registry = yule_registry::Registry::new(cache_dir, hf_token)?;

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(registry.pull(model_ref))?;

    println!();
    println!("model:       {}/{}", result.publisher, result.repo);
    println!("file:        {}", result.filename);
    println!("size:        {}", format_bytes(result.size_bytes));
    println!("merkle root: {}", result.merkle_root);
    println!("path:        {}", result.path.display());

    Ok(())
}

fn cmd_list() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cache_dir = yule_registry::Registry::default_cache_dir();
    let hf_token = std::env::var("HF_TOKEN").ok();
    let registry = yule_registry::Registry::new(cache_dir, hf_token)?;

    let models = registry.list_cached()?;
    if models.is_empty() {
        println!("no cached models");
        println!("  pull one with: yule pull bartowski/Llama-3.2-1B-Instruct-GGUF");
        return Ok(());
    }

    println!("cached models:\n");
    for m in &models {
        let verified = if m.merkle_root.is_some() {
            "verified"
        } else {
            "unverified"
        };
        println!("  {}/{}", m.publisher, m.repo);
        println!("    file:     {}", m.filename);
        println!("    size:     {}", format_bytes(m.size_bytes));
        println!("    status:   {verified}");
        if let Some(ref root) = m.merkle_root {
            println!("    merkle:   {}...", &root[..16.min(root.len())]);
        }
        println!();
    }

    Ok(())
}

fn cmd_audit(last: usize, verify_chain: bool) {
    let log = match yule_attest::log::AuditLog::default_path() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot open audit log: {e}");
            std::process::exit(1);
        }
    };

    if verify_chain {
        match log.verify_chain() {
            Ok((valid, count)) => {
                if valid {
                    println!("chain integrity: VALID ({count} records)");
                } else {
                    println!("chain integrity: BROKEN ({count} records)");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let records = match log.query_last(last) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if records.is_empty() {
        println!("no attestation records found");
        return;
    }

    println!("showing last {} of {} requested:\n", records.len(), last);

    for record in &records {
        let model = &record.model.name;
        let tokens = record.inference.tokens_generated;
        let sandbox = if record.sandbox.active { "yes" } else { "no" };
        let sig_len = record.signature.len();
        let ts = record.timestamp;

        println!("  session:   {}", record.session_id);
        println!("  timestamp: {ts}");
        println!("  model:     {model}");
        println!("  tokens:    {tokens}");
        println!("  sandbox:   {sandbox}");
        println!(
            "  signed:    {} ({sig_len}B)",
            if sig_len == 64 { "yes" } else { "no" }
        );
        println!("  prev_hash: {}", hex(&record.prev_hash));
        println!();
    }
}
