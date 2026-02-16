use crate::{AttestationRecord, InferenceAttestation, ModelAttestation, SandboxAttestation};
use ed25519_dalek::SigningKey;
use yule_core::error::Result;

/// Builder for attestation records. Created at inference start, finalized at end.
pub struct AttestationSession {
    session_id: String,
    started_at: u64,
    model: Option<ModelAttestation>,
    sandbox: Option<SandboxAttestation>,
}

impl AttestationSession {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        Self {
            session_id: format!("{:016x}", now.as_nanos()),
            started_at: now.as_secs(),
            model: None,
            sandbox: None,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Record model info at load time.
    pub fn set_model(
        &mut self,
        name: String,
        merkle_root: [u8; 32],
        publisher: Option<String>,
        signature_verified: bool,
    ) {
        self.model = Some(ModelAttestation {
            name,
            merkle_root,
            publisher,
            signature_verified,
        });
    }

    /// Record sandbox state.
    pub fn set_sandbox(&mut self, active: bool, memory_limit_bytes: u64) {
        self.sandbox = Some(SandboxAttestation {
            platform: std::env::consts::OS.to_string(),
            active,
            memory_limit_bytes,
        });
    }

    /// Finalize: sign the record with the device key and chain to previous record.
    pub fn finalize(
        self,
        inference: InferenceAttestation,
        signing_key: &SigningKey,
        prev_hash: [u8; 32],
    ) -> Result<AttestationRecord> {
        use ed25519_dalek::Signer;

        let mut record = AttestationRecord {
            session_id: self.session_id,
            timestamp: self.started_at,
            model: self.model.unwrap_or(ModelAttestation {
                name: "unknown".into(),
                merkle_root: [0u8; 32],
                publisher: None,
                signature_verified: false,
            }),
            sandbox: self.sandbox.unwrap_or(SandboxAttestation {
                platform: std::env::consts::OS.to_string(),
                active: false,
                memory_limit_bytes: 0,
            }),
            inference,
            signature: vec![],
            prev_hash,
        };

        // sign the attestation content
        let message = record.signable_bytes();
        let sig = signing_key.sign(&message);
        record.signature = sig.to_bytes().to_vec();

        Ok(record)
    }
}

impl Default for AttestationSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn create_and_finalize_session() {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);

        let mut session = AttestationSession::new();
        session.set_model(
            "test-model".into(),
            [1u8; 32],
            Some("test-pub".into()),
            true,
        );
        session.set_sandbox(true, 32 * 1024 * 1024 * 1024);

        let inference = InferenceAttestation {
            tokens_generated: 42,
            prompt_hash: blake3::hash(b"hello world").into(),
            output_hash: blake3::hash(b"response text").into(),
            temperature: 0.7,
            top_p: 0.9,
        };

        let record = session.finalize(inference, &key, [0u8; 32]).unwrap();

        assert_eq!(record.model.name, "test-model");
        assert_eq!(record.inference.tokens_generated, 42);
        assert!(record.sandbox.active);
        assert_eq!(record.signature.len(), 64);

        // verify signature
        let pubkey = key.verifying_key().to_bytes();
        assert!(record.verify_signature(&pubkey));
    }

    #[test]
    fn tampered_record_fails_verification() {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);

        let mut session = AttestationSession::new();
        session.set_model("test".into(), [0u8; 32], None, false);

        let inference = InferenceAttestation {
            tokens_generated: 10,
            prompt_hash: [0u8; 32],
            output_hash: [0u8; 32],
            temperature: 1.0,
            top_p: 1.0,
        };

        let mut record = session.finalize(inference, &key, [0u8; 32]).unwrap();
        record.inference.tokens_generated = 999; // tamper

        let pubkey = key.verifying_key().to_bytes();
        assert!(!record.verify_signature(&pubkey));
    }
}
