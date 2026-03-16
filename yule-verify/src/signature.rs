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
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool> {
        let verifier = MlDsaVerifier;
        verifier.verify(public_key, message, signature)
    }
}

impl Default for SignatureVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ML-DSA (FIPS 204) post-quantum signature — parameter set constants
// ---------------------------------------------------------------------------

/// ML-DSA-44 (security category 2): smallest parameter set.
pub mod ml_dsa_44 {
    /// Public key size in bytes.
    pub const PUBLIC_KEY_SIZE: usize = 1312;
    /// Secret (signing) key size in bytes.
    pub const SECRET_KEY_SIZE: usize = 2560;
    /// Signature size in bytes.
    pub const SIGNATURE_SIZE: usize = 2420;
    /// NIST security category.
    pub const SECURITY_CATEGORY: u8 = 2;
}

/// ML-DSA-65 (security category 3): medium parameter set.
pub mod ml_dsa_65 {
    /// Public key size in bytes.
    pub const PUBLIC_KEY_SIZE: usize = 1952;
    /// Secret (signing) key size in bytes.
    pub const SECRET_KEY_SIZE: usize = 4032;
    /// Signature size in bytes.
    pub const SIGNATURE_SIZE: usize = 3309;
    /// NIST security category.
    pub const SECURITY_CATEGORY: u8 = 3;
}

/// ML-DSA-87 (security category 5): largest parameter set.
pub mod ml_dsa_87 {
    /// Public key size in bytes.
    pub const PUBLIC_KEY_SIZE: usize = 2592;
    /// Secret (signing) key size in bytes.
    pub const SECRET_KEY_SIZE: usize = 4896;
    /// Signature size in bytes.
    pub const SIGNATURE_SIZE: usize = 4627;
    /// NIST security category.
    pub const SECURITY_CATEGORY: u8 = 5;
}

const ML_DSA_UNAVAILABLE: &str =
    "ML-DSA verification not yet available: awaiting stable Rust implementation";

// ---------------------------------------------------------------------------
// ML-DSA verifier stub
// ---------------------------------------------------------------------------

/// Post-quantum signature verifier for ML-DSA (FIPS 204).
///
/// This is a placeholder — the API surface is ready so that swapping in a real
/// implementation only requires changing the method bodies.
pub struct MlDsaVerifier;

impl MlDsaVerifier {
    /// Whether a working ML-DSA implementation is linked.
    /// Returns `false` until a mature Rust crate is available.
    pub fn is_available() -> bool {
        false
    }

    /// Verify an ML-DSA signature over `message` using `public_key`.
    ///
    /// Currently returns an error because no stable Rust ML-DSA crate exists.
    pub fn verify(&self, _public_key: &[u8], _message: &[u8], _signature: &[u8]) -> Result<bool> {
        Err(YuleError::Verification(ML_DSA_UNAVAILABLE.into()))
    }
}

/// Stub key pair for ML-DSA signing (not yet functional).
#[derive(Debug)]
pub struct MlDsaKeyPair {
    pub public_key: Vec<u8>,
    pub secret_key: Vec<u8>,
}

impl MlDsaKeyPair {
    /// Generate a new ML-DSA key pair.
    ///
    /// Currently returns an error because no stable Rust ML-DSA crate exists.
    pub fn generate() -> Result<Self> {
        Err(YuleError::Verification(ML_DSA_UNAVAILABLE.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_dsa_not_available() {
        assert!(!MlDsaVerifier::is_available());
    }

    #[test]
    fn ml_dsa_verify_returns_unavailable_error() {
        let verifier = MlDsaVerifier;
        let result = verifier.verify(b"key", b"message", b"sig");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("ML-DSA verification not yet available"));
    }

    #[test]
    fn ml_dsa_keypair_generate_returns_unavailable_error() {
        let result = MlDsaKeyPair::generate();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("ML-DSA verification not yet available"));
    }

    #[test]
    fn ml_dsa_via_signature_verifier() {
        let sv = SignatureVerifier::new();
        let result = sv.verify_ml_dsa(b"key", b"message", b"sig");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ML-DSA"));
    }

    #[test]
    fn ml_dsa_parameter_constants() {
        // ML-DSA-44
        assert_eq!(ml_dsa_44::PUBLIC_KEY_SIZE, 1312);
        assert_eq!(ml_dsa_44::SECRET_KEY_SIZE, 2560);
        assert_eq!(ml_dsa_44::SIGNATURE_SIZE, 2420);
        assert_eq!(ml_dsa_44::SECURITY_CATEGORY, 2);

        // ML-DSA-65
        assert_eq!(ml_dsa_65::PUBLIC_KEY_SIZE, 1952);
        assert_eq!(ml_dsa_65::SECRET_KEY_SIZE, 4032);
        assert_eq!(ml_dsa_65::SIGNATURE_SIZE, 3309);
        assert_eq!(ml_dsa_65::SECURITY_CATEGORY, 3);

        // ML-DSA-87
        assert_eq!(ml_dsa_87::PUBLIC_KEY_SIZE, 2592);
        assert_eq!(ml_dsa_87::SECRET_KEY_SIZE, 4896);
        assert_eq!(ml_dsa_87::SIGNATURE_SIZE, 4627);
        assert_eq!(ml_dsa_87::SECURITY_CATEGORY, 5);
    }
}
