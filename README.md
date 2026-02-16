<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset=".github/logo-light.svg">
    <img alt="Yule" src=".github/logo.svg" width="80">
  </picture>
</p>

<h1 align="center">Yule</h1>

<p align="center">
  <a href="https://github.com/visualstudioblyat/yule/actions/workflows/ci.yml"><img src="https://github.com/visualstudioblyat/yule/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/yule"><img src="https://img.shields.io/crates/v/yule.svg" alt="crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>
Local AI inference runtime written from scratch in pure Rust. No bindings to llama.cpp, no CUDA dependency, no C++ anywhere in the stack.

**Beta** — 2.5 months of solo development. Correctness-validated on TinyLlama 1.1B (CPU + Vulkan). Actively developing multi-model support and Linux sandboxing.

## What this is

A from-scratch implementation of transformer inference with cryptographic integrity built into every layer:

- **Pure Rust GGUF parser** — no `unsafe` in parsing code, no C++ string handling CVEs
- **11 hand-written Vulkan compute shaders** — raw GLSL 450 compiled to SPIR-V, fused dequant+dot-product directly from quantized weight blocks. Includes a Q4_K shader that unpacks 144-byte super-blocks with 6-bit scale extraction from a sharded 12-byte metadata array. No cuBLAS, no GGML backend, no abstraction layers
- **Merkle verification** — blake3 hash tree over every tensor in the model file. If a single weight is modified, the root hash changes
- **Kernel-level process sandbox** — Windows Job Objects with no child processes, no clipboard access, no UI permissions. The model process cannot touch the host
- **Signed audit trail** — Ed25519-signed, hash-chained attestation records for every inference. Cryptographically tamper-evident
- **8.5× GPU speedup** — hand-tuned Vulkan path vs scalar CPU on Q4_K_M. No NVIDIA lock-in

## Install

```
cargo install yule
```

For GPU acceleration:
```
cargo install yule --features vulkan
```

Requires Rust 1.85+ (2024 edition). Vulkan feature requires Vulkan SDK.

## Usage

### Run inference

```bash
yule run ./model.gguf --prompt "Explain quantum computing in one sentence"
```

Options:
- `--max-tokens 512` — generation limit (default: 512)
- `--temperature 0.7` — sampling temperature (default: 0.7)
- `--backend auto|cpu|vulkan` — compute backend (default: auto)
- `--no-sandbox` — disable process sandbox (not recommended)

### Pull models from HuggingFace

```bash
yule pull TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
yule list
```

### Start API server

```bash
yule serve ./model.gguf
```

Prints a capability token to stderr. All requests require `Authorization: Bearer <token>`.

Options:
- `--bind 127.0.0.1:11434` — listen address
- `--token <token>` — use a specific token instead of generating one
- `--no-sandbox` — disable process sandbox

### API endpoints

**Yule-native** — every response includes integrity proof (merkle root, sandbox status, attestation ID):

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:11434/yule/health
curl -H "Authorization: Bearer $TOKEN" http://localhost:11434/yule/model
curl -H "Authorization: Bearer $TOKEN" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:11434/yule/chat
```

**OpenAI-compatible** — drop-in replacement for tool integration:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:11434/v1/chat/completions
```

### Verify and audit

```bash
yule verify ./model.gguf          # merkle tree + per-tensor hashes
yule inspect ./model.gguf         # raw GGUF metadata
yule audit --last 10              # recent attestation records
yule audit --verify-chain         # validate entire hash chain
```

## Supported models

Architectures: Llama, Mistral, Phi-3, Qwen2, Gemma2.

Quantizations: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, Q2_K, Q3_K, F16, F32, BF16.

AVX2 SIMD: Q4_0 (3.9×), Q8_0 (5.0×), Q4_K (1.9×), Q6_K — with runtime CPU detection and automatic dispatch.

Vulkan GPU: Q4_0, Q4_K, Q8_0 — hand-written compute shaders, no framework dependency.

## Architecture

10-crate Cargo workspace. 98 tests, zero `unsafe` in application code (only in SIMD intrinsics and mmap).

| Crate | What it does |
|-------|-------------|
| `yule-core` | GGUF parser, 11 dequant kernels (scalar + AVX2), BPE tokenizer |
| `yule-infer` | Forward pass (CPU + GPU), GQA attention, KV cache, constant-time sampler |
| `yule-gpu` | ComputeBackend trait, Vulkan backend (ash + gpu-allocator), 11 SPIR-V shaders |
| `yule-verify` | blake3 merkle trees, Ed25519 signatures, model manifests |
| `yule-attest` | Hash-chained attestation records, device key management |
| `yule-sandbox` | Windows Job Object isolation (Linux seccomp, macOS seatbelt planned) |
| `yule-api` | Axum server, capability-token auth, SSE streaming, OpenAI-compat |
| `yule-registry` | HuggingFace model pull, local cache, quant detection |
| `yule-cli` | CLI entry point |
| `yule-bench` | Criterion benchmarks |

### GPU backend

The Vulkan path uploads quantized weights to VRAM at init, then runs the full forward pass as a single command buffer submission. Each quantized matmul shader reads raw quantized bytes from SSBOs and does fused dequant+dot — no VRAM bloat from dequantized weight copies. The Q4_K shader handles 144-byte super-blocks with a non-trivial 6-bit scale/min extraction that matches the CPU path exactly.

Embedding lookup and sampling stay on CPU. Auto-detects discrete GPUs at startup, falls back to CPU if Vulkan is unavailable.

### Inference thread model

```
HTTP request → Axum → mpsc channel → Inference Thread → tokens → channel → SSE/JSON response
```

Model inference runs on a dedicated `std::thread` (mmap refs aren't `Send`). Async HTTP communicates via channels.

### Cryptographic integrity

Every inference session produces a signed attestation record: model merkle root, sandbox status, blake3(prompt), blake3(output), token count, sampling parameters. Records are hash-chained — each includes blake3 of the previous record. The log at `~/.yule/audit.jsonl` is append-only and tamper-evident.

## Development

```bash
cargo build                       # CPU only
cargo build --features vulkan     # with Vulkan GPU
cargo test -j 1                   # all tests
cargo bench -p yule-bench         # dequant benchmarks
```

## License

Apache-2.0
