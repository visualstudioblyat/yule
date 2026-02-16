use serde::{Deserialize, Serialize};
use yule_core::error::{Result, YuleError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub version: u32,
    pub model_name: String,
    pub architecture: String,
    pub publisher: Publisher,
    pub merkle_root: [u8; 32],
    pub tensor_hashes: Vec<TensorHash>,
    pub signature: ManifestSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Publisher {
    pub name: String,
    pub public_key_ed25519: Option<[u8; 32]>,
    pub public_key_ml_dsa: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorHash {
    pub name: String,
    pub blake3: [u8; 32],
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManifestSignature {
    Ed25519(Vec<u8>),
    MlDsa(Vec<u8>),
    Hybrid { ed25519: Vec<u8>, ml_dsa: Vec<u8> },
    Unsigned,
}

impl ModelManifest {
    /// Load manifest from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| YuleError::Verification(format!("invalid manifest: {e}")))
    }

    /// Load manifest from a file path.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(YuleError::Io)?;
        Self::from_json(&data)
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| YuleError::Verification(format!("serialize manifest: {e}")))
    }

    /// The signed content = manifest JSON with signature set to Unsigned.
    /// This is what gets signed/verified — the signature field itself is excluded.
    pub fn signable_bytes(&self) -> Result<Vec<u8>> {
        let mut copy = self.clone();
        copy.signature = ManifestSignature::Unsigned;
        serde_json::to_vec(&copy)
            .map_err(|e| YuleError::Verification(format!("signable bytes: {e}")))
    }

    /// Verify the Ed25519 signature against the publisher's key.
    pub fn verify_signature(&self) -> Result<bool> {
        let pubkey = match self.publisher.public_key_ed25519 {
            Some(ref k) => k,
            None => return Ok(false),
        };

        let sig_bytes = match &self.signature {
            ManifestSignature::Ed25519(s) => s,
            ManifestSignature::Hybrid { ed25519, .. } => ed25519,
            ManifestSignature::Unsigned => return Ok(false),
            ManifestSignature::MlDsa(_) => return Ok(false), // can't verify without ML-DSA impl
        };

        if sig_bytes.len() != 64 {
            return Err(YuleError::Verification(
                "Ed25519 signature must be 64 bytes".into(),
            ));
        }

        let message = self.signable_bytes()?;
        let verifier = crate::signature::SignatureVerifier::new();
        let sig: [u8; 64] = sig_bytes[..64].try_into().unwrap();
        verifier.verify_ed25519(pubkey, &message, &sig)
    }

    /// Verify merkle root matches actual tensor data.
    pub fn verify_merkle(&self, tensor_data: &[u8]) -> bool {
        let tree = crate::merkle::MerkleTree::new();
        tree.verify(tensor_data, &self.merkle_root)
    }

    /// Sign the manifest with an Ed25519 signing key.
    pub fn sign_ed25519(&mut self, signing_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        self.signature = ManifestSignature::Unsigned; // clear before computing signable bytes
        let message = serde_json::to_vec(self).unwrap();
        let sig = signing_key.sign(&message);
        self.signature = ManifestSignature::Ed25519(sig.to_bytes().to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            version: 1,
            model_name: "test-model".into(),
            architecture: "llama".into(),
            publisher: Publisher {
                name: "test-publisher".into(),
                public_key_ed25519: None,
                public_key_ml_dsa: None,
            },
            merkle_root: [0u8; 32],
            tensor_hashes: vec![],
            signature: ManifestSignature::Unsigned,
        }
    }

    #[test]
    fn roundtrip_json() {
        let m = test_manifest();
        let json = m.to_json().unwrap();
        let m2 = ModelManifest::from_json(&json).unwrap();
        assert_eq!(m2.model_name, "test-model");
        assert_eq!(m2.version, 1);
    }

    #[test]
    fn unsigned_manifest_verify_returns_false() {
        let m = test_manifest();
        assert!(!m.verify_signature().unwrap());
    }

    #[test]
    fn sign_and_verify() {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let signing_key = SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();

        let mut m = test_manifest();
        m.publisher.public_key_ed25519 = Some(verifying_key.to_bytes());
        m.sign_ed25519(&signing_key);

        assert!(m.verify_signature().unwrap());
    }

    #[test]
    fn tampered_manifest_fails_verify() {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let signing_key = SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();

        let mut m = test_manifest();
        m.publisher.public_key_ed25519 = Some(verifying_key.to_bytes());
        m.sign_ed25519(&signing_key);

        // tamper
        m.model_name = "tampered".into();
        assert!(!m.verify_signature().unwrap());
    }

    #[test]
    fn merkle_verification() {
        let data = vec![42u8; 2048];
        let tree = crate::merkle::MerkleTree::new();
        let root = tree.build(&data);

        let mut m = test_manifest();
        m.merkle_root = root.hash;
        assert!(m.verify_merkle(&data));
        assert!(!m.verify_merkle(&[0u8; 2048]));
    }
}
