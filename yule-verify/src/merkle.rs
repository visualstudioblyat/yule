use yule_core::error::{Result, YuleError};

pub struct MerkleTree {
    leaf_size: usize,
}

impl MerkleTree {
    pub fn new() -> Self {
        Self {
            leaf_size: 1024 * 1024, // 1MB leaves — balance between granularity and overhead
        }
    }

    pub fn with_leaf_size(leaf_size: usize) -> Self {
        Self { leaf_size }
    }

    pub fn build(&self, data: &[u8]) -> MerkleRoot {
        let leaf_hashes: Vec<[u8; 32]> = data
            .chunks(self.leaf_size)
            .map(|chunk| blake3::hash(chunk).into())
            .collect();

        let root = self.compute_root(&leaf_hashes);

        MerkleRoot {
            hash: root,
            leaf_count: leaf_hashes.len(),
        }
    }

    pub fn verify(&self, data: &[u8], expected_root: &[u8; 32]) -> bool {
        let computed = self.build(data);
        computed.hash == *expected_root
    }

    pub fn verify_streaming<R: std::io::Read>(
        &self,
        mut reader: R,
        expected_root: &[u8; 32],
    ) -> Result<bool> {
        let mut leaf_hashes = Vec::new();
        let mut buf = vec![0u8; self.leaf_size];
        loop {
            let mut filled = 0;
            while filled < self.leaf_size {
                match reader.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(e) => return Err(YuleError::Io(e)),
                }
            }
            if filled == 0 {
                break;
            }
            leaf_hashes.push(blake3::hash(&buf[..filled]).into());
        }
        let root = self.compute_root(&leaf_hashes);
        Ok(root == *expected_root)
    }

    /// Build a `StreamingMerkleVerifier` that verifies chunks individually
    /// against the tree structure derived from `data`.
    pub fn streaming_verifier(&self, data: &[u8]) -> StreamingMerkleVerifier {
        let leaf_hashes: Vec<[u8; 32]> = data
            .chunks(self.leaf_size)
            .map(|chunk| blake3::hash(chunk).into())
            .collect();
        let root = self.compute_root(&leaf_hashes);
        StreamingMerkleVerifier::new(root, leaf_hashes)
    }

    fn compute_root(&self, leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut level = leaves.to_vec();
        while level.len() > 1 {
            let mut next_level = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                let hash = if pair.len() == 2 {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(&pair[0]);
                    hasher.update(&pair[1]);
                    hasher.finalize().into()
                } else {
                    pair[0] // odd node promoted
                };
                next_level.push(hash);
            }
            level = next_level;
        }
        level[0]
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MerkleRoot {
    pub hash: [u8; 32],
    pub leaf_count: usize,
}

/// Streaming Merkle verifier: accepts chunks one at a time and verifies each
/// against the expected leaf hash from the pre-built tree.
///
/// Useful for large models where you want to verify data as it streams in
/// without waiting for the entire file to be loaded.
pub struct StreamingMerkleVerifier {
    expected_root: [u8; 32],
    expected_leaves: Vec<[u8; 32]>,
    verified: Vec<bool>,
}

impl StreamingMerkleVerifier {
    /// Create a new streaming verifier from a known root hash and expected leaf hashes.
    pub fn new(expected_root: [u8; 32], expected_leaves: Vec<[u8; 32]>) -> Self {
        let count = expected_leaves.len();
        Self {
            expected_root,
            expected_leaves,
            verified: vec![false; count],
        }
    }

    /// Create a streaming verifier from raw data (builds the tree internally).
    /// The `leaf_size` controls chunk granularity.
    pub fn from_data(data: &[u8], leaf_size: usize) -> Self {
        let tree = MerkleTree::with_leaf_size(leaf_size);
        tree.streaming_verifier(data)
    }

    /// Feed a chunk at the given index. Returns `Ok(true)` if the chunk matches
    /// the expected leaf hash, `Ok(false)` if it doesn't.
    /// Returns an error if the chunk index is out of range.
    pub fn feed_chunk(&mut self, chunk_index: usize, data: &[u8]) -> Result<bool> {
        if chunk_index >= self.expected_leaves.len() {
            return Err(YuleError::Verification(format!(
                "chunk index {chunk_index} out of range (tree has {} leaves)",
                self.expected_leaves.len()
            )));
        }

        let hash: [u8; 32] = blake3::hash(data).into();
        let matches = hash == self.expected_leaves[chunk_index];
        if matches {
            self.verified[chunk_index] = true;
        }
        Ok(matches)
    }

    /// Check whether all chunks have been verified successfully.
    pub fn is_complete(&self) -> bool {
        self.verified.iter().all(|&v| v)
    }

    /// Finalize verification: returns `Ok(())` if every chunk has been verified,
    /// or an error listing which chunks are still missing.
    pub fn finalize(&self) -> Result<()> {
        let missing: Vec<usize> = self
            .verified
            .iter()
            .enumerate()
            .filter(|&(_, v)| !v)
            .map(|(i, _)| i)
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(YuleError::Verification(format!(
                "streaming verification incomplete: {} of {} chunks not verified (missing: {:?})",
                missing.len(),
                self.expected_leaves.len(),
                missing
            )))
        }
    }

    /// The expected root hash this verifier was built with.
    pub fn expected_root(&self) -> &[u8; 32] {
        &self.expected_root
    }

    /// Total number of chunks (leaves) in the tree.
    pub fn chunk_count(&self) -> usize {
        self.expected_leaves.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_verifier_all_chunks_in_order() {
        let leaf_size = 64;
        let data = vec![42u8; 200]; // 4 chunks: 64+64+64+8
        let tree = MerkleTree::with_leaf_size(leaf_size);
        let root = tree.build(&data);

        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);
        assert_eq!(verifier.expected_root(), &root.hash);
        assert!(!verifier.is_complete());

        for (i, chunk) in data.chunks(leaf_size).enumerate() {
            assert!(verifier.feed_chunk(i, chunk).unwrap());
        }

        assert!(verifier.is_complete());
        assert!(verifier.finalize().is_ok());
    }

    #[test]
    fn streaming_verifier_out_of_order() {
        let leaf_size = 64;
        let data = vec![7u8; 200];
        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);

        let chunks: Vec<&[u8]> = data.chunks(leaf_size).collect();
        // feed in reverse order
        for i in (0..chunks.len()).rev() {
            assert!(verifier.feed_chunk(i, chunks[i]).unwrap());
        }
        assert!(verifier.is_complete());
        assert!(verifier.finalize().is_ok());
    }

    #[test]
    fn streaming_verifier_wrong_chunk() {
        let leaf_size = 64;
        let data = vec![42u8; 128]; // 2 chunks
        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);

        // correct first chunk
        assert!(verifier.feed_chunk(0, &data[..leaf_size]).unwrap());

        // wrong second chunk
        let bad_data = vec![0u8; 64];
        assert!(!verifier.feed_chunk(1, &bad_data).unwrap());

        assert!(!verifier.is_complete());
        let err = verifier.finalize().unwrap_err();
        assert!(err.to_string().contains("not verified"));
    }

    #[test]
    fn streaming_verifier_index_out_of_range() {
        let leaf_size = 64;
        let data = vec![42u8; 64]; // 1 chunk
        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);

        let result = verifier.feed_chunk(5, &data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range"));
    }

    #[test]
    fn streaming_verifier_empty_data() {
        let verifier = StreamingMerkleVerifier::from_data(&[], 64);
        assert!(verifier.is_complete());
        assert!(verifier.finalize().is_ok());
        assert_eq!(verifier.chunk_count(), 0);
    }

    #[test]
    fn streaming_verifier_single_chunk() {
        let leaf_size = 1024;
        let data = vec![99u8; 100]; // smaller than leaf_size
        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);

        assert_eq!(verifier.chunk_count(), 1);
        assert!(verifier.feed_chunk(0, &data).unwrap());
        assert!(verifier.is_complete());
    }

    #[test]
    fn streaming_verifier_finalize_incomplete() {
        let leaf_size = 64;
        let data = vec![42u8; 192]; // 3 chunks
        let mut verifier = StreamingMerkleVerifier::from_data(&data, leaf_size);

        // only feed first chunk
        assert!(verifier.feed_chunk(0, &data[..leaf_size]).unwrap());

        let err = verifier.finalize().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("2 of 3 chunks not verified"));
    }
}
