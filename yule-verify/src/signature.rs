use ed25519_dalek::{Signature, VerifyingKey};
use yule_core::error::{Result, YuleError};

pub struct SignatureVerifier;

impl SignatureVerifier {
    pub fn new() -> Self {
        Self
    }

    pub fn verify_ed25519(
        &self,
        public_key: &[u8; 32],
        message: &[u8],
        signature: &[u8; 64],
    ) -> Result<bool> {
        let key = VerifyingKey::from_bytes(public_key)
            .map_err(|e| YuleError::Verification(format!("invalid public key: {e}")))?;

        let sig = Signature::from_bytes(signature);

        Ok(key.verify_strict(message, &sig).is_ok())
    }

    pub fn verify_ml_dsa(
        &self,
        _public_key: &[u8],
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool> {
        // TODO: post-quantum ML-DSA verification
        // waiting on mature Rust crate (pqcrypto or ml-dsa)
        todo!("ML-DSA signature verification")
    }
}

impl Default for SignatureVerifier {
    fn default() -> Self {
        Self::new()
    }
}
