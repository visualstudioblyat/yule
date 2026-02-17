# Yule — Changelog

---

## v0.3.2

**2026-02-17**

Linux sandbox implementation. The inference process is now sandboxed on both Windows and Linux.

### Added

- **Linux seccomp-BPF** — syscall allowlist with ~60 base syscalls. Conditional networking (for `yule serve`) and GPU ioctl access. Default deny returns EPERM.
- **Linux Landlock** — filesystem restricted to model path (read-only), GPU device nodes, and essential system libraries. Graceful degradation on kernels < 5.13.
- **Linux rlimit** — `RLIMIT_AS` caps virtual memory, matching Windows memory limit behavior.
- **`yule run` sandbox** — the `run` command now applies sandboxing (was previously ignored). Uses `allow_network: false`, GPU access based on backend selection.
- 5 new Linux sandbox tests (rlimit roundtrip, seccomp filter construction, conditional syscall verification). Total: 103 tests.

### Changed

- Sandbox log messages are now platform-agnostic.

---

## v0.3.1

**2026-02-15**

Eliminated all CPU round-trips from the Vulkan forward pass. KV cache stays in VRAM, attention runs on GPU with GQA support, entire forward pass submits as a single command buffer.

### Added

- **Batched command buffer API** — `begin_batch()`, `dispatch_batched()`, `barrier()`, `transfer_barrier()`, `submit_batch()`. The entire forward pass (all layers, attention, FFN, final norm, output projection) records into one Vulkan command buffer and submits once. Down from 330+ individual submits per token.

- **GPU attention with GQA** — `attn_score.comp` and `attn_value.comp` updated with `kv_stride` push constant for correct grouped query attention indexing. Per-head dispatch loop handles arbitrary Q-to-KV head ratios. No more CPU attention fallback.

- **KV cache offset writes** — `copy_buffer_offset` on the `ComputeBackend` trait. KV cache writes use GPU-to-GPU offset copies instead of downloading the entire cache, patching one position, and re-uploading. Eliminates ~2.8GB of PCIe traffic per token.

- **GPU buffer copies** — `copy_buffer` on the `ComputeBackend` trait. Residual connections use `vkCmdCopyBuffer` directly instead of CPU staging round-trips.

- **Multi-head RoPE** — `rope.comp` rewritten to rotate all Q and K heads in a single dispatch. Takes `n_heads_q` and `n_heads_k` as push constants. Interleaved pair layout matching the CPU path.

- **7 new tests** — `copy_buffer`, `copy_buffer_offset` (including KV cache pattern), multi-head RoPE, non-zero position RoPE, full attention pipeline, GQA attention with 2:1 head ratio. Total: 98 tests, 0 failures.

### Changed

- `attn_score.comp` push constants: 16 → 20 bytes (added `kv_stride`)
- `attn_value.comp` push constants: 12 → 20 bytes (added `kv_stride`, `out_offset`)
- `rope.comp` push constants: 12 → 20 bytes (added `n_heads_q`, `n_heads_k`)
- `ComputeBackend::rope()` signature updated with head count parameters
- `gpu_runner.rs` forward pass fully restructured for single-submit batched execution

---

## v0.3.0

**2026-02-15**

Vulkan GPU compute backend. 11 SPIR-V shaders, full forward pass on GPU, `--backend vulkan` flag. The math that was eating 100% of one CPU core now runs on the graphics card.

### Added

- **Vulkan device layer** (`yule-gpu/src/vulkan/`) — ash for raw Vulkan bindings, gpu-allocator for VMA-style memory management. Instance creation, physical device selection (prefer discrete GPU), compute queue allocation. Runtime `is_available()` probe so the CLI can detect Vulkan at startup without crashing.

- **11 GLSL 450 compute shaders** — compiled to SPIR-V at dev time via `glslc`. Pre-compiled `.spv` files loaded with `include_bytes!()`, no runtime shader compilation. Covers the full inference pipeline: `add`, `silu_mul`, `rms_norm`, `rope`, `softmax`, `embed_lookup`, `attn_score`, `attn_value`, `qmv_q4_0`, `qmv_q4_k`, `qmv_q8_0`. Each quantized matmul shader does fused dequant+dot — raw quantized bytes in SSBO, no VRAM bloat from dequantized weights.

- **Q4_K shader** — the most complex shader. 144-byte super-blocks with 6-bit scale extraction from a sharded 12-byte metadata array. Custom `unpack_f16()` function because GLSL 450 has no native f16 load.

- **`quantized_matmul`** — new method on the `ComputeBackend` trait. Fused dequant+dot dispatch. CpuBackend returns an error (it uses the existing `qmv()` path), VulkanBackend dispatches the appropriate shader based on DType.

- **GpuTransformerRunner** (`yule-infer/src/gpu_runner.rs`) — hybrid CPU/GPU forward pass. Uploads all quantized weight tensors to VRAM at init. Per-token forward: CPU embedding lookup → GPU layers (rms_norm, quantized_matmul×3, rope, attention, quantized_matmul output proj, add, rms_norm, quantized_matmul×2 FFN, silu_mul, quantized_matmul down, add) → GPU final norm → GPU output projection → logits back to CPU for sampling. Implements the same `ModelRunner` trait as TransformerRunner.

- **`--backend auto|cpu|vulkan`** — CLI flag on `yule run` and `yule serve`. `auto` probes for Vulkan at startup and falls back to CPU. Feature-gated: `cargo build --features vulkan` to include the Vulkan path, otherwise it compiles out entirely.

- **Feature flag chain** — `yule-cli/vulkan` activates `yule-gpu/vulkan` + `yule-infer/vulkan`. Default build has no Vulkan dependency.

### Architecture

Vulkan infrastructure is 5 modules under `yule-gpu/src/vulkan/`:
- `device.rs` — instance, physical device, logical device, queue
- `memory.rs` — gpu-allocator wrapper, staging buffers, host↔device copies
- `pipeline.rs` — SPIR-V loading, descriptor sets, push constants, pipeline cache
- `commands.rs` — command buffer recording, dispatch, barriers, submit+fence
- `mod.rs` — VulkanBackend struct, ComputeBackend trait impl

Known TODOs: attention is still computed on CPU (GQA grouping makes multi-head GPU dispatch complex), KV cache writes go through CPU roundtrip (should use offset buffer writes), internal buffer copies use CPU staging (should use vkCmdCopyBuffer).

91 tests, 0 failures. Both `--features vulkan` and default configurations compile clean.

---

## v0.2.0

**2026-02-15**

Five milestones in one release. Multi-architecture inference, process sandboxing, cryptographic integrity, full API server, model registry. Yule went from "runs TinyLlama" to "runs any GGUF model with a signed audit trail behind a sandboxed API."

### Added

- **Multi-architecture inference** — unified TransformerRunner replaces the Llama-specific runner. Architecture differences are config flags, not code duplication. RunnerConfig captures sliding_window, partial_rotary_dim, has_qkv_bias, activation type, post-norms, softcap. One runner handles Llama, Mistral, Phi-3, Qwen2, and Gemma2.

- **Sliding window attention** — Mistral and Gemma2 restrict attention to the last W tokens. KV cache wraps around instead of growing unbounded. Detected from GGUF `attention.sliding_window` metadata.

- **Partial RoPE** — Phi-3 only applies rotary embeddings to a subset of head dimensions. RopeTable sized to actual rotary_dim, non-rotated dimensions pass through unchanged.

- **QKV bias** — Qwen2 adds learned bias to Q, K, and V projections. Detected by checking whether `attn_q.bias` tensors exist in the GGUF file.

- **GeGLU activation** — Gemma2 uses GELU(gate) × up instead of SiLU(gate) × up. Activation enum selects the right function.

- **Gemma quirks** — norm weight offset (add 1.0 to all norm weights), embedding scaling (multiply by sqrt(dim)), attention logit softcapping (tanh(score / cap) × cap), final logit softcapping.

- **Post-attention and post-FFN norms** — Gemma2 applies additional RMSNorm layers after attention output projection and after the down projection. Detected from tensor presence.

- **Chat templates** — hardcoded templates for Llama3, Mistral, Phi-3, Qwen2, Gemma2. Each model gets the right `<|begin_of_text|>`, `[INST]`, `<|im_start|>`, etc.

- **Windows Job Object sandbox** — `yule serve` drops the process into a Job Object with memory limits, no child spawning, UI restrictions (clipboard, desktop, display, global atoms), kill-on-close. SandboxGuard with RAII CloseHandle. `--no-sandbox` to disable. Linux/macOS return proper errors instead of panicking.

- **Model manifests** — JSON format with publisher identity, Ed25519 signatures, per-tensor blake3 hashes. Hybrid signature enum with ML-DSA placeholder for post-quantum. `sign_ed25519()`, `verify_signature()`, `verify_merkle()`.

- **Device signing key** — Ed25519 keypair auto-generated on first run via `getrandom` (OS CSPRNG, no `rand` dependency). Persisted at `~/.yule/keys/device.key` + `device.pub`.

- **Publisher trust registry** — `trust_publisher()` saves a publisher's public key. `publisher_key()` loads it. `list_publishers()` enumerates. Stored at `~/.yule/keys/{name}.pub`.

- **Attestation records** — every inference produces a signed record: session ID, model merkle root, sandbox status, blake3(prompt), blake3(output), token count, sampling params. Records are hash-chained (each includes blake3 of the previous record). Append-only JSON-lines log at `~/.yule/audit.jsonl`.

- **Audit CLI** — `yule audit --last 10` shows recent records. `yule audit --verify-chain` walks the entire log validating hash links.

- **API server** — Axum-based with two surfaces. Yule-native: `/yule/health`, `/yule/model`, `/yule/chat`, `/yule/tokenize` — every response includes integrity proof (merkle root, sandbox status, attestation ID). OpenAI-compatible: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`. SSE streaming with typed YuleStreamEvent. Inference runs on a dedicated `std::thread` (mmap refs aren't Send), async HTTP communicates via channels.

- **Capability-token auth** — blake3 CSPRNG token generated at startup, printed to stderr. All API requests require `Authorization: Bearer <token>`. Server stores hash, not plaintext.

- **Model registry** — HuggingFace API client for repo listing and GGUF file discovery. Quantization label extraction from filenames. Async download with progress tracking. Local cache at `~/.yule/models/{publisher}/{repo}/`. `yule pull publisher/repo` downloads and caches. `yule list` shows cached models with size and quant type.

### Architecture

91 tests across 10 crates (up from 80). Key test additions: manifest JSON roundtrip, sign→verify, tampered manifest detection, audit chain verification, sandbox policy construction, HuggingFace API mocks, model resolution, cache operations.

---

## v0.1.0

**2026-02-15**

First public release. Pure Rust local inference runtime with cryptographic integrity from the ground up. 10-crate workspace, 80 tests, zero unsafe in application code (only in SIMD intrinsics and mmap).

### Added

- **GGUF parser** — full binary format support with metadata extraction, tensor layout, mmap-based weight access. Architecture detection for Llama, Mistral, Phi, Qwen, Gemma, Mixtral, Mamba.

- **Inference engine** — complete forward pass: embedding lookup, RoPE positional encoding, GQA with KV cache, SwiGLU FFN, RMSNorm. Runs TinyLlama 1.1B through all 22 layers, validated against Python reference to 6+ decimal places.

- **Dequantization kernels** — Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, Q2_K, Q3_K, F16, F32, BF16. Both scalar and AVX2 SIMD paths with automatic dispatch. Differential testing (1000 random iterations per kernel) confirms AVX2 matches scalar.

- **Constant-time sampler** — temperature, top-p, top-k, repetition penalty. Branchless token selection via XOR-mask bit tricks to prevent timing side-channels. CSPRNG via `getrandom` instead of `SystemTime` seeding.

- **SentencePiece BPE tokenizer** — score-based merge (not greedy longest-match). Matches llama-cpp-python output exactly.

- **AVX2 SIMD kernels** — for Q4_0 (3.9×), Q8_0 (5.0×), Q4_K (1.9×). Runtime CPU detection via AtomicU8, automatic dispatch. 10,000+ differential tests per kernel. Pre-allocated ScratchBuffers eliminate all per-token allocations. Pre-allocated KV cache to max_seq_len. Precomputed RoPE sin/cos lookup table. Criterion benchmarks for dequant throughput.

- **Pre-faulted mmap** — `memmap2::populate()` eliminates soft page faults during inference. Falls back to standard mmap on unsupported platforms.

- **Prefetch intrinsics** — `_mm_prefetch` in the `qmv` hot loop hides memory latency by prefetching the next weight block while computing the current dot product.

- **Merkle verification** — blake3 tree over tensor data (1MB leaves). Computed at model load, root included in every API response. Tamper detection for model files.

- **CLI** — `yule run` (inference with streaming output), `yule verify` (merkle tree + metadata), `yule inspect` (raw GGUF metadata dump).

### Architecture

10 crates: `yule-core` (GGUF, tokenizer, dequant, SIMD), `yule-verify` (merkle, signatures), `yule-attest` (attestation records), `yule-infer` (model runner, sampler, KV cache), `yule-sandbox` (OS-level process isolation), `yule-api` (HTTP server), `yule-gpu` (compute backend abstraction), `yule-registry` (model cache), `yule-bench` (benchmarks), `yule-cli` (binary).
