use crate::error::{Result, YuleError};
use crate::gguf::{GgufFile, GgufValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: u32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub kind: TokenizerKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerKind {
    Bpe,
    SentencePiece,
    Tiktoken,
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> u32;
    fn bos_token(&self) -> Option<u32>;
    fn eos_token(&self) -> Option<u32>;
}

/// BPE tokenizer loaded from GGUF metadata.
/// Handles encode (text → tokens) and decode (tokens → text).
pub struct BpeTokenizer {
    // token_id → token bytes
    vocab: Vec<Vec<u8>>,
    // token string → token_id (for encoding)
    token_to_id: HashMap<Vec<u8>, u32>,
    // merge pairs: (first, second) → merged token_id, ordered by priority
    merges: Vec<(u32, u32)>,
    scores: Vec<f32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    // byte fallback tokens: byte value → token_id
    byte_fallback: HashMap<u8, u32>,
}

impl BpeTokenizer {
    /// Load tokenizer from parsed GGUF metadata.
    /// Reads tokenizer.ggml.tokens, tokenizer.ggml.scores, tokenizer.ggml.merges
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // extract token list
        let tokens_val = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| YuleError::Parse("missing tokenizer.ggml.tokens".into()))?;
        let token_strings = match tokens_val {
            GgufValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    GgufValue::String(s) => Ok(s.as_bytes().to_vec()),
                    _ => Err(YuleError::Parse("token is not a string".into())),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                return Err(YuleError::Parse(
                    "tokenizer.ggml.tokens is not an array".into(),
                ));
            }
        };

        let vocab_size = token_strings.len();

        // extract scores (used for merge priority in SentencePiece-style BPE)
        let scores = if let Some(GgufValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.scores")
        {
            arr.iter()
                .map(|v| match v {
                    GgufValue::Float32(f) => Ok(*f),
                    _ => Ok(0.0),
                })
                .collect::<Result<Vec<f32>>>()?
        } else {
            vec![0.0; vocab_size]
        };

        // build reverse lookup
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        for (id, tok) in token_strings.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        // extract merges if present (HF-style BPE)
        let merges = if let Some(GgufValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.merges")
        {
            arr.iter()
                .filter_map(|v| {
                    if let GgufValue::String(s) = v {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            let a = token_to_id.get(parts[0].as_bytes())?;
                            let b = token_to_id.get(parts[1].as_bytes())?;
                            Some((*a, *b))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // detect byte fallback tokens like <0x00> through <0xFF>
        let mut byte_fallback = HashMap::new();
        for (id, tok) in token_strings.iter().enumerate() {
            if tok.len() == 6 && tok.starts_with(b"<0x") && tok.ends_with(b">") {
                if let Ok(s) = std::str::from_utf8(&tok[3..5]) {
                    if let Ok(byte_val) = u8::from_str_radix(s, 16) {
                        byte_fallback.insert(byte_val, id as u32);
                    }
                }
            }
        }

        // special tokens
        let bos_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());

        Ok(Self {
            vocab: token_strings,
            token_to_id,
            merges,
            scores,
            bos_id,
            eos_id,
            byte_fallback,
        })
    }

    /// Encode using SentencePiece-style BPE (score-based merge priority).
    /// SentencePiece starts with individual UTF-8 codepoints (or byte fallbacks),
    /// then iteratively merges the highest-scoring adjacent pair.
    fn encode_bpe(&self, text: &[u8]) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // SentencePiece normalization: prepend space and replace spaces with ▁ (U+2581)
        let text_str = String::from_utf8_lossy(text);
        let normalized = format!(" {}", text_str);
        let normalized = normalized.replace(' ', "\u{2581}");
        let text = normalized.as_bytes();

        // Step 1: split into individual UTF-8 codepoints, map each to its token id.
        // If a codepoint isn't in the vocab, fall back to individual bytes.
        let mut tokens = Vec::new();
        let text_str = std::str::from_utf8(text).unwrap_or("");
        for ch in text_str.chars() {
            let ch_bytes = ch.to_string();
            if let Some(&id) = self.token_to_id.get(ch_bytes.as_bytes()) {
                tokens.push(id);
            } else {
                // byte fallback for each byte of this character
                for &b in ch_bytes.as_bytes() {
                    if let Some(&id) = self.byte_fallback.get(&b) {
                        tokens.push(id);
                    } else {
                        tokens.push(0); // unknown
                    }
                }
            }
        }

        // Step 2: iteratively merge the highest-scoring adjacent pair
        if !self.merges.is_empty() {
            self.apply_hf_merges(&mut tokens);
        } else if self.scores.len() == self.vocab.len() {
            self.apply_sp_merges(&mut tokens);
        }

        tokens
    }

    // HF-style BPE: apply merges in the order given
    fn apply_hf_merges(&self, tokens: &mut Vec<u32>) {
        // build merge priority lookup
        let mut merge_rank: HashMap<(u32, u32), usize> = HashMap::new();
        for (rank, &(a, b)) in self.merges.iter().enumerate() {
            merge_rank.insert((a, b), rank);
        }

        loop {
            if tokens.len() < 2 {
                break;
            }
            // find the merge with lowest rank (highest priority)
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&rank) = merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_rank == usize::MAX {
                break;
            }

            // merge: look up the concatenated token
            let a = &self.vocab[tokens[best_idx] as usize];
            let b = &self.vocab[tokens[best_idx + 1] as usize];
            let mut merged = a.clone();
            merged.extend_from_slice(b);
            if let Some(&merged_id) = self.token_to_id.get(&merged) {
                tokens[best_idx] = merged_id;
                tokens.remove(best_idx + 1);
            } else {
                break;
            }
        }
    }

    // SentencePiece-style: merge the pair whose merged token has highest score
    fn apply_sp_merges(&self, tokens: &mut Vec<u32>) {
        loop {
            if tokens.len() < 2 {
                break;
            }
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = 0;
            let mut best_id = 0u32;

            for i in 0..tokens.len() - 1 {
                let a = &self.vocab[tokens[i] as usize];
                let b = &self.vocab[tokens[i + 1] as usize];
                let mut merged = a.clone();
                merged.extend_from_slice(b);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }
            if best_score == f32::NEG_INFINITY {
                break;
            }
            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }
    }

    /// Decode tokens to string.
    /// Handles byte fallback tokens and partial UTF-8 sequences.
    fn decode_tokens(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            let id = id as usize;
            if id < self.vocab.len() {
                let tok = &self.vocab[id];
                // check if it's a byte fallback token
                if tok.len() == 6 && tok.starts_with(b"<0x") && tok.ends_with(b">") {
                    if let Ok(s) = std::str::from_utf8(&tok[3..5]) {
                        if let Ok(byte_val) = u8::from_str_radix(s, 16) {
                            bytes.push(byte_val);
                            continue;
                        }
                    }
                }
                // SentencePiece uses ▁ (U+2581) for space
                for &b in tok {
                    bytes.push(b);
                }
            }
        }

        // replace SentencePiece space marker with actual space
        let text = String::from_utf8_lossy(&bytes).to_string();
        text.replace('▁', " ")
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode_bpe(text.as_bytes()))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.decode_tokens(tokens))
    }

    fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }

    fn bos_token(&self) -> Option<u32> {
        self.bos_id
    }

    fn eos_token(&self) -> Option<u32> {
        self.eos_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_vocab() -> BpeTokenizer {
        // SentencePiece-style vocab: ▁ prefix + a, b, c, ▁ab, ▁abc
        // encode_bpe normalizes "abc" → "▁abc", so vocab must use ▁ prefix
        let spc = "\u{2581}"; // ▁
        let vocab: Vec<Vec<u8>> = vec![
            spc.as_bytes().to_vec(),          // 0: ▁
            b"a".to_vec(),                    // 1: a
            b"b".to_vec(),                    // 2: b
            b"c".to_vec(),                    // 3: c
            format!("{spc}a").into_bytes(),   // 4: ▁a
            format!("{spc}ab").into_bytes(),  // 5: ▁ab
            format!("{spc}abc").into_bytes(), // 6: ▁abc
        ];
        let mut token_to_id = HashMap::new();
        for (id, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }
        BpeTokenizer {
            vocab,
            token_to_id,
            // ▁+a → ▁a (rank 0), ▁a+b → ▁ab (rank 1), ▁ab+c → ▁abc (rank 2)
            merges: vec![(0, 1), (4, 2), (5, 3)],
            scores: vec![0.0; 7],
            bos_id: None,
            eos_id: None,
            byte_fallback: HashMap::new(),
        }
    }

    #[test]
    fn test_decode_simple() {
        let tok = make_simple_vocab();
        // decode [1, 2, 3] → "abc"
        let out = tok.decode(&[1, 2, 3]).unwrap();
        assert_eq!(out, "abc");
    }

    #[test]
    fn test_decode_merged() {
        let tok = make_simple_vocab();
        // decode [6] → "▁abc" → decoder replaces ▁ with space → " abc"
        let out = tok.decode(&[6]).unwrap();
        assert_eq!(out, " abc");
    }

    #[test]
    fn test_encode_merges() {
        let tok = make_simple_vocab();
        // "abc" → normalize → "▁abc" → chars [▁(0), a(1), b(2), c(3)]
        // merge ▁+a → ▁a(4), merge ▁a+b → ▁ab(5), merge ▁ab+c → ▁abc(6)
        let tokens = tok.encode("abc").unwrap();
        assert_eq!(tokens, vec![6]);
    }

    #[test]
    fn test_encode_partial() {
        let tok = make_simple_vocab();
        // "ab" → normalize → "▁ab" → chars [▁(0), a(1), b(2)]
        // merge ▁+a → ▁a(4), merge ▁a+b → ▁ab(5)
        let tokens = tok.encode("ab").unwrap();
        assert_eq!(tokens, vec![5]);
    }
}
