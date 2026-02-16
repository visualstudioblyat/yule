pub mod keys;
pub mod manifest;
pub mod merkle;
pub mod signature;

use crate::manifest::ModelManifest;
use crate::merkle::MerkleTree;
use yule_core::error::{Result, YuleError};

pub struct IntegrityVerifier {
    merkle: MerkleTree,
}

impl IntegrityVerifier {
    pub fn new() -> Self {
        Self {
            merkle: MerkleTree::new(),
        }
    }

    /// Full model verification: merkle root + manifest signature.
    pub fn verify_model(
        &self,
        manifest: &ModelManifest,
        tensor_data: &[u8],
    ) -> Result<VerificationResult> {
        // 1. verify merkle root matches tensor data
        let root = self.merkle.build(tensor_data);
        let merkle_valid = root.hash == manifest.merkle_root;

        if !merkle_valid {
            return Err(YuleError::Verification(
                "merkle root mismatch — tensor data has been modified".into(),
            ));
        }

        // 2. verify publisher signature
        let signature_valid = match manifest.verify_signature() {
            Ok(valid) => Some(valid),
            Err(_) => Some(false),
        };

        Ok(VerificationResult {
            root_hash: root.hash,
            tensor_count: manifest.tensor_hashes.len(),
            verified: merkle_valid && signature_valid.unwrap_or(false),
            signature_valid,
        })
    }

    /// Verify only the merkle root (no manifest needed).
    pub fn verify_merkle_only(&self, tensor_data: &[u8], expected_root: &[u8; 32]) -> bool {
        self.merkle.verify(tensor_data, expected_root)
    }
}

impl Default for IntegrityVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct VerificationResult {
    pub root_hash: [u8; 32],
    pub tensor_count: usize,
    pub verified: bool,
    pub signature_valid: Option<bool>,
}
