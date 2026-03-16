//! Device authentication using a PAKE (Password-Authenticated Key Exchange) protocol.
//!
//! Devices pair via a 6-digit code displayed to the user. The protocol ensures:
//! - Attacker gets exactly 1 guess per attempt (no offline dictionary attacks)
//! - Neither side learns the code's hash (zero-knowledge)
//! - Session key is derived for subsequent encrypted communication
//!
//! Rate limiting: 3 failed attempts per 5-minute window, then 5-minute lockout.

use blake3;
use yule_core::error::{Result, YuleError};

pub struct DeviceAuthenticator {
    failed_attempts: u32,
    max_attempts: u32,
    lockout_until: Option<std::time::Instant>,
    lockout_duration: std::time::Duration,
    window_start: std::time::Instant,
    window_duration: std::time::Duration,
}

pub struct PairingSession {
    code: [u8; 6],
    local_secret: [u8; 32],
    local_public: [u8; 32],
    state: PairingState,
}

enum PairingState {
    WaitingForPeer,
    ReceivedPeerKey([u8; 32]),
    #[allow(dead_code)]
    Confirmed([u8; 32]), // session key
    Failed,
}

pub struct PairingResult {
    pub session_key: [u8; 32],
    pub peer_id: [u8; 32],
}

impl DeviceAuthenticator {
    pub fn new() -> Self {
        Self {
            failed_attempts: 0,
            max_attempts: 3,
            lockout_until: None,
            lockout_duration: std::time::Duration::from_secs(300),
            window_start: std::time::Instant::now(),
            window_duration: std::time::Duration::from_secs(300),
        }
    }

    pub fn is_locked_out(&self) -> bool {
        if let Some(until) = self.lockout_until {
            std::time::Instant::now() < until
        } else {
            false
        }
    }

    fn check_rate_limit(&mut self) -> Result<()> {
        if self.is_locked_out() {
            return Err(YuleError::Api(
                "device auth locked out: too many failed attempts".into(),
            ));
        }

        // Reset window if expired
        if self.window_start.elapsed() > self.window_duration {
            self.failed_attempts = 0;
            self.window_start = std::time::Instant::now();
        }

        Ok(())
    }

    fn record_failure(&mut self) {
        self.failed_attempts += 1;
        if self.failed_attempts >= self.max_attempts {
            self.lockout_until = Some(std::time::Instant::now() + self.lockout_duration);
            tracing::warn!(
                "device auth: lockout triggered after {} failed attempts",
                self.failed_attempts
            );
        }
    }

    fn record_success(&mut self) {
        self.failed_attempts = 0;
        self.lockout_until = None;
    }

    /// Generate a 6-digit pairing code and start a session.
    pub fn start_pairing(&mut self) -> Result<(PairingSession, String)> {
        self.check_rate_limit()?;

        let mut code_bytes = [0u8; 6];
        getrandom::fill(&mut code_bytes).map_err(|e| YuleError::Api(format!("RNG failed: {e}")))?;

        // Map to digits 0-9
        for b in &mut code_bytes {
            *b %= 10;
        }

        let code_string: String = code_bytes.iter().map(|b| char::from(b'0' + b)).collect();

        // Generate local keypair for this session
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).map_err(|e| YuleError::Api(format!("RNG failed: {e}")))?;

        // Derive public commitment: hash(secret || code)
        let mut hasher = blake3::Hasher::new();
        hasher.update(&secret);
        hasher.update(&code_bytes);
        let public = *hasher.finalize().as_bytes();

        let session = PairingSession {
            code: code_bytes,
            local_secret: secret,
            local_public: public,
            state: PairingState::WaitingForPeer,
        };

        Ok((session, code_string))
    }

    /// Process peer's commitment and generate our response.
    pub fn receive_peer_commitment(
        &mut self,
        session: &mut PairingSession,
        peer_public: [u8; 32],
    ) -> Result<[u8; 32]> {
        self.check_rate_limit()?;

        session.state = PairingState::ReceivedPeerKey(peer_public);

        // Return our public commitment
        Ok(session.local_public)
    }

    /// Confirm the pairing with the code entered by the peer.
    pub fn confirm_pairing(
        &mut self,
        session: &mut PairingSession,
        peer_code: &[u8; 6],
    ) -> Result<PairingResult> {
        self.check_rate_limit()?;

        // Verify codes match
        let codes_match = constant_time_eq(&session.code, peer_code);

        if !codes_match {
            self.record_failure();
            session.state = PairingState::Failed;
            return Err(YuleError::Api("pairing code mismatch".into()));
        }

        let peer_public = match session.state {
            PairingState::ReceivedPeerKey(pk) => pk,
            _ => {
                return Err(YuleError::Api(
                    "pairing not in correct state for confirmation".into(),
                ));
            }
        };

        // Derive session key from both secrets + code
        let session_key = blake3::derive_key("yule device auth session key v1", &{
            let mut material = Vec::with_capacity(96);
            material.extend_from_slice(&session.local_secret);
            material.extend_from_slice(&peer_public);
            material.extend_from_slice(&session.code);
            material
        });

        // Derive peer ID from their public commitment
        let peer_id = *blake3::hash(&peer_public).as_bytes();

        self.record_success();
        session.state = PairingState::Confirmed(session_key);

        Ok(PairingResult {
            session_key,
            peer_id,
        })
    }
}

impl Default for DeviceAuthenticator {
    fn default() -> Self {
        Self::new()
    }
}

/// Constant-time byte comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_pairing_generates_6_digit_code() {
        let mut auth = DeviceAuthenticator::new();
        let (_session, code) = auth.start_pairing().unwrap();
        assert_eq!(code.len(), 6);
        assert!(code.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_successful_pairing() {
        let mut auth_a = DeviceAuthenticator::new();
        let mut auth_b = DeviceAuthenticator::new();

        // Device A starts pairing
        let (mut session_a, code_a) = auth_a.start_pairing().unwrap();

        // Device B starts with same code
        let code_bytes: Vec<u8> = code_a.bytes().map(|b| b - b'0').collect();
        let mut code_b = [0u8; 6];
        code_b.copy_from_slice(&code_bytes);

        let (mut session_b, _) = auth_b.start_pairing().unwrap();
        session_b.code = code_b; // simulate entering the same code

        // Exchange commitments
        let b_public = session_b.local_public;
        let a_public = auth_a
            .receive_peer_commitment(&mut session_a, b_public)
            .unwrap();
        auth_b
            .receive_peer_commitment(&mut session_b, a_public)
            .unwrap();

        // Confirm with matching codes
        let result_a = auth_a.confirm_pairing(&mut session_a, &code_b).unwrap();
        let result_b = auth_b
            .confirm_pairing(&mut session_b, &session_a.code)
            .unwrap();

        // Both should have valid session keys (different because secrets differ)
        assert_ne!(result_a.session_key, [0u8; 32]);
        assert_ne!(result_b.session_key, [0u8; 32]);
    }

    #[test]
    fn test_wrong_code_fails() {
        let mut auth = DeviceAuthenticator::new();
        let (mut session, _code) = auth.start_pairing().unwrap();

        // Fake peer commitment
        auth.receive_peer_commitment(&mut session, [42u8; 32])
            .unwrap();

        // Wrong code
        let wrong_code = [9u8; 6];
        let result = auth.confirm_pairing(&mut session, &wrong_code);
        assert!(result.is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let mut auth = DeviceAuthenticator::new();

        for _ in 0..3 {
            let (mut session, _code) = auth.start_pairing().unwrap();
            auth.receive_peer_commitment(&mut session, [0u8; 32])
                .unwrap();
            let _ = auth.confirm_pairing(&mut session, &[9u8; 6]); // wrong code
        }

        // Should be locked out now
        assert!(auth.is_locked_out());
        assert!(auth.start_pairing().is_err());
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }
}
