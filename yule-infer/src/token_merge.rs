//! Token merging for prefill optimization.
//! Merges adjacent tokens with high cosine similarity to reduce sequence length.
//! Reference: "Token Reduction Beyond Efficiency" (2025)

/// Configuration for token merging.
pub struct TokenMergeConfig {
    /// Merge if cosine similarity exceeds this value.
    pub similarity_threshold: f32,
    /// Don't merge more than this fraction of the sequence.
    pub max_merge_ratio: f32,
    /// Don't merge if sequence is shorter than this.
    pub min_sequence_length: usize,
}

impl Default for TokenMergeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            max_merge_ratio: 0.5,
            min_sequence_length: 16,
        }
    }
}

/// Result of token merging.
pub struct MergedTokens {
    /// Merged embedding vectors.
    pub merged_embeddings: Vec<Vec<f32>>,
    /// For each merged token, which original indices it contains.
    pub merge_map: Vec<Vec<usize>>,
    /// Original sequence length before merging.
    pub original_length: usize,
    /// Sequence length after merging.
    pub merged_length: usize,
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Merge adjacent tokens with high cosine similarity.
///
/// Scans pairs of adjacent embeddings and averages those whose cosine
/// similarity exceeds `config.similarity_threshold`, up to
/// `config.max_merge_ratio` of the sequence.
pub fn merge_tokens(embeddings: &[Vec<f32>], config: &TokenMergeConfig) -> MergedTokens {
    if embeddings.len() < config.min_sequence_length {
        // Don't merge short sequences
        return MergedTokens {
            merged_embeddings: embeddings.to_vec(),
            merge_map: (0..embeddings.len()).map(|i| vec![i]).collect(),
            original_length: embeddings.len(),
            merged_length: embeddings.len(),
        };
    }

    let max_merges = (embeddings.len() as f32 * config.max_merge_ratio) as usize;
    let mut merged = Vec::new();
    let mut merge_map = Vec::new();
    let mut i = 0;
    let mut merges_done = 0;

    while i < embeddings.len() {
        if i + 1 < embeddings.len() && merges_done < max_merges {
            let sim = cosine_similarity(&embeddings[i], &embeddings[i + 1]);
            if sim > config.similarity_threshold {
                // Merge: average the two embeddings
                let dim = embeddings[i].len();
                let mut avg = vec![0.0f32; dim];
                for d in 0..dim {
                    avg[d] = (embeddings[i][d] + embeddings[i + 1][d]) * 0.5;
                }
                merged.push(avg);
                merge_map.push(vec![i, i + 1]);
                i += 2;
                merges_done += 1;
                continue;
            }
        }
        merged.push(embeddings[i].clone());
        merge_map.push(vec![i]);
        i += 1;
    }

    MergedTokens {
        original_length: embeddings.len(),
        merged_length: merged.len(),
        merged_embeddings: merged,
        merge_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_merge_short_sequence() {
        let embeddings: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
        let config = TokenMergeConfig::default(); // min_sequence_length = 16
        let result = merge_tokens(&embeddings, &config);
        assert_eq!(result.merged_length, result.original_length);
        assert_eq!(result.merged_length, 8);
    }

    #[test]
    fn test_merge_identical_tokens() {
        // 16 tokens, first two are identical, rest are orthogonal unit vectors
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        embeddings.push(vec![1.0, 0.0, 0.0, 0.0]);
        embeddings.push(vec![1.0, 0.0, 0.0, 0.0]); // identical to first
        for i in 2..16 {
            let mut v = vec![0.0f32; 16];
            v[i] = 1.0; // orthogonal unit vectors -> cosine sim = 0
            embeddings.push(v);
        }
        let config = TokenMergeConfig::default();
        let result = merge_tokens(&embeddings, &config);
        // The two identical tokens should merge, reducing length by 1
        assert_eq!(result.original_length, 16);
        assert_eq!(result.merged_length, 15);
    }

    #[test]
    fn test_no_merge_dissimilar() {
        // 16 tokens, all orthogonal / very different
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        for i in 0..16 {
            let mut v = vec![0.0f32; 16];
            v[i] = 1.0;
            embeddings.push(v);
        }
        let config = TokenMergeConfig::default();
        let result = merge_tokens(&embeddings, &config);
        assert_eq!(result.merged_length, result.original_length);
    }

    #[test]
    fn test_merge_map_correctness() {
        // 16 tokens: pairs (0,1) and (2,3) are identical, rest differ
        let mut embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        for i in 4..16 {
            let mut v = vec![0.0f32; 16];
            v[i] = 1.0;
            embeddings.push(v);
        }
        let config = TokenMergeConfig::default();
        let result = merge_tokens(&embeddings, &config);
        // First merged token should map to originals [0, 1]
        assert_eq!(result.merge_map[0], vec![0, 1]);
        // Second merged token should map to originals [2, 3]
        assert_eq!(result.merge_map[1], vec![2, 3]);
        // Remaining should be singletons
        for entry in &result.merge_map[2..] {
            assert_eq!(entry.len(), 1);
        }
        assert_eq!(result.merged_length, 14);
    }
}
