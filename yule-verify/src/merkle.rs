use yule_core::error::Result;

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
        _reader: R,
        _expected_root: &[u8; 32],
    ) -> Result<bool> {
        // TODO: streaming verification — verify chunks as they load
        // without buffering entire model in memory
        todo!("streaming Merkle verification")
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
