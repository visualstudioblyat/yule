fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_asm_kernels)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if arch == "x86_64" {
        // Assembly kernels require clang (MSVC cannot compile .S files).
        // Try CLANG_PATH env var first, then probe common install locations.
        let clang = std::env::var("CLANG_PATH").ok().or_else(find_clang);

        let Some(compiler) = clang else {
            println!(
                "cargo::warning=clang not found, skipping assembly kernels (using Rust SIMD fallback)"
            );
            return;
        };

        let result = cc::Build::new()
            .compiler(&compiler)
            .file("src/kernels/q4k_gemv_avx2.S")
            .flag("-mavx2")
            .flag("-mfma")
            .flag("-mf16c")
            .try_compile("yule_asm_kernels");

        match result {
            Ok(()) => {
                println!("cargo::rustc-cfg=has_asm_kernels");
            }
            Err(e) => {
                println!("cargo::warning=assembly kernel compilation failed: {e}");
                println!("cargo::warning=falling back to Rust SIMD kernels");
            }
        }
    }
}

fn find_clang() -> Option<String> {
    // check common locations
    let candidates = [
        "clang",
        "clang-18",
        "clang-17",
        "clang-16",
        "/usr/bin/clang",
        "/opt/homebrew/opt/llvm/bin/clang",
    ];

    #[cfg(target_os = "windows")]
    let win_candidates = [
        "C:\\Program Files\\LLVM\\bin\\clang.exe",
        "C:\\Program Files (x86)\\LLVM\\bin\\clang.exe",
    ];

    for candidate in &candidates {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(candidate.to_string());
        }
    }

    #[cfg(target_os = "windows")]
    for candidate in &win_candidates {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    None
}
