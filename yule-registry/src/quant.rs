use crate::hf_api::HfFileEntry;

/// Preferred quantization order — best quality/size tradeoff first.
const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_S", "Q2_K",
    "IQ4_XS", "IQ4_NL",
];

/// Filter file listing to only GGUF files.
pub fn filter_gguf_files(files: &[HfFileEntry]) -> Vec<&HfFileEntry> {
    files
        .iter()
        .filter(|f| f.entry_type == "file" && f.path.ends_with(".gguf"))
        .collect()
}

/// Extract quantization label from a GGUF filename.
/// e.g. "Llama-3.2-1B-Instruct-Q4_K_M.gguf" → Some("Q4_K_M")
pub fn extract_quant_label(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf")?;
    // Walk known quant labels — match case-insensitively against the end of the stem
    for &quant in QUANT_PREFERENCE {
        if stem.to_ascii_uppercase().ends_with(quant) {
            return Some(quant.to_string());
        }
    }
    // Also check F16, F32, BF16
    for tag in &["F16", "F32", "BF16"] {
        if stem.to_ascii_uppercase().ends_with(tag) {
            return Some((*tag).to_string());
        }
    }
    None
}

/// Select the best GGUF file from a list, optionally matching a requested quant.
pub fn select_gguf_file<'a>(
    gguf_files: &[&'a HfFileEntry],
    requested_quant: Option<&str>,
) -> Option<&'a HfFileEntry> {
    if gguf_files.is_empty() {
        return None;
    }

    // If user requested a specific quant, match it case-insensitively
    if let Some(req) = requested_quant {
        let req_upper = req.to_ascii_uppercase();
        for file in gguf_files {
            if let Some(label) = extract_quant_label(&file.path) {
                if label.to_ascii_uppercase() == req_upper {
                    return Some(file);
                }
            }
        }
        // No exact match — fall through to preference order
    }

    // Walk preference order and pick the first match
    for &pref in QUANT_PREFERENCE {
        for file in gguf_files {
            if let Some(label) = extract_quant_label(&file.path) {
                if label == pref {
                    return Some(file);
                }
            }
        }
    }

    // Fallback: first GGUF file
    Some(gguf_files[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, size: u64) -> HfFileEntry {
        HfFileEntry {
            path: path.to_string(),
            size,
            entry_type: "file".to_string(),
        }
    }

    #[test]
    fn extract_quant_labels() {
        assert_eq!(
            extract_quant_label("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            extract_quant_label("model-Q8_0.gguf"),
            Some("Q8_0".to_string())
        );
        assert_eq!(
            extract_quant_label("model-F16.gguf"),
            Some("F16".to_string())
        );
        assert_eq!(extract_quant_label("README.md"), None);
    }

    #[test]
    fn select_with_preference() {
        let files = [
            entry("model-Q8_0.gguf", 1000),
            entry("model-Q4_K_M.gguf", 500),
            entry("model-Q6_K.gguf", 800),
        ];
        let gguf: Vec<&HfFileEntry> = files.iter().collect();

        let selected = select_gguf_file(&gguf, None).unwrap();
        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_with_explicit_quant() {
        let files = [
            entry("model-Q4_K_M.gguf", 500),
            entry("model-Q8_0.gguf", 1000),
        ];
        let gguf: Vec<&HfFileEntry> = files.iter().collect();

        let selected = select_gguf_file(&gguf, Some("q8_0")).unwrap();
        assert_eq!(selected.path, "model-Q8_0.gguf");
    }

    #[test]
    fn filter_gguf_only() {
        let files = vec![
            entry("README.md", 100),
            entry("model-Q4_K_M.gguf", 500),
            entry("config.json", 200),
            entry("model-Q8_0.gguf", 1000),
        ];
        let gguf = filter_gguf_files(&files);
        assert_eq!(gguf.len(), 2);
    }
}
