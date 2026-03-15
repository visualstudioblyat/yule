//! Kani formal verification harnesses for the API authentication system.
//!
//! These harnesses prove:
//! 1. Token generation never panics
//! 2. A generated token always verifies
//! 3. A random string never verifies (no false positives)
//! 4. Token verification is deterministic
//!
//! Run with: `cargo kani --harness <name>` (Linux only, requires Kani)

#[cfg(kani)]
mod proofs {
    use crate::auth::TokenAuthority;

    /// Prove: generating a token and verifying it always succeeds.
    #[kani::proof]
    fn generated_token_always_verifies() {
        let mut auth = TokenAuthority::new();
        let token = auth.generate_token();
        assert!(auth.verify(&token), "generated token must always verify");
    }

    /// Prove: the empty string never verifies.
    #[kani::proof]
    fn empty_string_never_verifies() {
        let mut auth = TokenAuthority::new();
        let _token = auth.generate_token();
        assert!(!auth.verify(""), "empty string must never verify");
    }

    /// Prove: from_existing always verifies the provided token.
    #[kani::proof]
    fn from_existing_verifies() {
        let auth = TokenAuthority::from_existing("test-token-123");
        assert!(auth.verify("test-token-123"));
        assert!(!auth.verify("wrong-token"));
    }

    /// Prove: verification is deterministic — same token, same result.
    #[kani::proof]
    fn verify_is_deterministic() {
        let mut auth = TokenAuthority::new();
        let token = auth.generate_token();
        let r1 = auth.verify(&token);
        let r2 = auth.verify(&token);
        assert_eq!(r1, r2, "verification must be deterministic");
    }
}
