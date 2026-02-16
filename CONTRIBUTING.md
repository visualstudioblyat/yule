# Contributing to Yule

## Getting Started

```bash
git clone https://github.com/visualstudioblyat/yule.git
cd yule
cargo build
cargo test -j 1
```

Requires Rust 1.85+ (2024 edition).

For the Vulkan GPU path:
```bash
cargo build --features vulkan
cargo test -j 1 --features vulkan
```

Requires Vulkan SDK installed.

## Development

```bash
cargo fmt --all              # format
cargo clippy --workspace     # lint
cargo test -j 1 --workspace  # test (serial — some tests share state)
cargo bench -p yule-bench    # benchmarks
```

CI runs all three on every push. PRs must pass before merge.

## Project Structure

10-crate workspace. Each crate has a single responsibility:

| Crate | Owns |
|-------|------|
| `yule-core` | GGUF parsing, dequantization, tokenizer, SIMD |
| `yule-infer` | Forward pass, attention, KV cache, sampler |
| `yule-gpu` | Vulkan backend, compute shaders |
| `yule-verify` | Merkle trees, Ed25519, model manifests |
| `yule-attest` | Attestation records, audit log |
| `yule-sandbox` | OS-level process isolation |
| `yule-api` | HTTP server, auth, SSE streaming |
| `yule-registry` | HuggingFace client, model cache |
| `yule-cli` | CLI entry point |
| `yule-bench` | Criterion benchmarks |

## Code Standards

- No `unsafe` in application code. `unsafe` is only permitted in SIMD intrinsics (`yule-core/src/simd/`) and mmap (`yule-core/src/mmap.rs`).
- Every SIMD kernel must have a scalar reference implementation and differential tests.
- Every Vulkan shader must match the CPU path exactly — verified through end-to-end inference comparison.
- No `unwrap()` in library code. Use proper error types.

## Submitting Changes

1. Fork the repo
2. Create a branch (`git checkout -b fix/description`)
3. Make your changes
4. Run `cargo fmt && cargo clippy && cargo test -j 1`
5. Open a PR against `main`

Keep PRs focused. One logical change per PR.

## Security Issues

See [SECURITY.md](SECURITY.md) for responsible disclosure.
