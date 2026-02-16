fn main() {
    // SPIR-V shaders are pre-compiled and checked in under shaders/compiled/.
    // To recompile: run `shaders/compile.sh` (requires glslc from Vulkan SDK).
    #[cfg(feature = "vulkan")]
    for name in &[
        "add",
        "silu_mul",
        "rms_norm",
        "rope",
        "softmax",
        "embed_lookup",
        "attn_score",
        "attn_value",
        "qmv_q4_0",
        "qmv_q4_k",
        "qmv_q8_0",
    ] {
        println!("cargo:rerun-if-changed=shaders/{name}.comp");
    }
}
