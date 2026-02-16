pub mod log;
pub mod session;

use serde::{Deserialize, Serialize};

/// A signed record of one inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationRecord {
    pub session_id: String,
    pub timestamp: u64,
    pub model: ModelAttestation,
    pub sandbox: SandboxAttestation,
    pub inference: InferenceAttestation,
    /// Ed25519 signature over the canonical JSON of all fields above.
    pub signature: Vec<u8>,
    /// blake3 hash of the previous record (chain integrity).
    pub prev_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAttestation {
    pub name: String,
    pub merkle_root: [u8; 32],
    pub publisher: Option<String>,
    pub signature_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxAttestation {
    pub platform: String,
    pub active: bool,
    pub memory_limit_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceAttestation {
    pub tokens_generated: u64,
    pub prompt_hash: [u8; 32],
    pub output_hash: [u8; 32],
    pub temperature: f32,
    pub top_p: f32,
}

impl AttestationRecord {
    /// Compute blake3 hash of this record (for chain linking).
    pub fn hash(&self) -> [u8; 32] {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        blake3::hash(&bytes).into()
    }

    /// The content that gets signed = all fields except signature and prev_hash.
    pub fn signable_bytes(&self) -> Vec<u8> {
        // sign over the core attestation data, not the chain/signature metadata
        let content = serde_json::json!({
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "sandbox": self.sandbox,
            "inference": self.inference,
        });
        serde_json::to_vec(&content).unwrap_or_default()
    }

    /// Verify the Ed25519 signature on this record.
    pub fn verify_signature(&self, public_key: &[u8; 32]) -> bool {
        if self.signature.len() != 64 {
            return false;
        }
        let verifier = yule_verify::signature::SignatureVerifier::new();
        let sig: [u8; 64] = self.signature[..64].try_into().unwrap();
        verifier
            .verify_ed25519(public_key, &self.signable_bytes(), &sig)
            .unwrap_or(false)
    }
}
