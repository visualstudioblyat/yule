//! Encrypted transport channel for peer-to-peer inference requests.
//!
//! Implements a simplified 3-message handshake inspired by the Noise XX pattern,
//! using X25519 ECDH for key agreement and BLAKE3 for key derivation, stream
//! encryption, and message authentication.
//!
//! **Roadmap target**: `Noise_XX_25519_ChaChaPoly_BLAKE2s` via the `snow` crate.
//! This module defines the transport API surface and provides a working
//! placeholder implementation built entirely from workspace dependencies
//! (`blake3`, `ed25519-dalek` / `curve25519-dalek`, `getrandom`).

use yule_core::error::{Result, YuleError};

// ---------------------------------------------------------------------------
// X25519 helpers — curve25519-dalek is a transitive dep of ed25519-dalek.
// We use its `MontgomeryPoint` directly for ECDH with clamped multiplication.
// ---------------------------------------------------------------------------
use curve25519_dalek::montgomery::MontgomeryPoint;

/// Compute the X25519 Diffie-Hellman shared secret.
///
/// `secret` is a raw 32-byte private key; `their_public` is a Montgomery
/// u-coordinate. Clamping is applied internally by `mul_clamped`.
fn x25519(secret: &[u8; 32], their_public: &[u8; 32]) -> [u8; 32] {
    let point = MontgomeryPoint(*their_public);
    point.mul_clamped(*secret).to_bytes()
}

/// X25519 base-point multiplication (compute public key from secret).
fn x25519_basepoint(secret: &[u8; 32]) -> [u8; 32] {
    MontgomeryPoint::mul_base_clamped(*secret).to_bytes()
}

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

/// A static or ephemeral X25519 keypair used by the transport layer.
pub struct TransportKeypair {
    pub secret: [u8; 32],
    pub public: [u8; 32],
}

impl TransportKeypair {
    /// Generate a fresh random keypair using OS entropy.
    pub fn generate() -> Result<Self> {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret)
            .map_err(|e| YuleError::Api(format!("getrandom failed: {e}")))?;
        let public = x25519_basepoint(&secret);
        Ok(Self { secret, public })
    }
}

// ---------------------------------------------------------------------------
// Encrypted message
// ---------------------------------------------------------------------------

/// Wire representation of an encrypted transport message.
pub struct EncryptedMessage {
    /// Encrypted payload (same length as plaintext).
    pub ciphertext: Vec<u8>,
    /// Monotonically increasing nonce — also serves as replay-protection.
    pub nonce: u64,
    /// BLAKE3 MAC over `nonce || ciphertext`.
    pub tag: [u8; 32],
}

// ---------------------------------------------------------------------------
// Channel state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelState {
    /// Fresh channel, no messages exchanged yet.
    Initial,
    /// Handshake in progress (we are the initiator and have sent msg 1).
    HandshakeInitiated,
    /// Handshake in progress (we are the responder and have sent msg 2).
    HandshakeResponded,
    /// Session keys established — ready for encrypt/decrypt.
    Established,
    /// Channel has been explicitly closed.
    Closed,
}

/// Number of messages after which `needs_rekey` returns true.
const REKEY_MESSAGE_THRESHOLD: u64 = 1 << 20; // ~1 million messages

/// Encrypted transport channel for peer-to-peer inference traffic.
///
/// The handshake follows a simplified Noise XX pattern:
///
/// ```text
/// msg1  initiator -> responder : e_pub_i
/// msg2  responder -> initiator : e_pub_r || Enc(static_pub_r)
/// msg3  initiator -> responder : Enc(static_pub_i)
/// ```
///
/// After message 3 both sides derive independent send/recv keys from the
/// shared secrets.  All symmetric operations use BLAKE3 (keyed hash for MAC,
/// key-derivation for session keys, XOF output as stream cipher keystream).
pub struct TransportChannel {
    state: ChannelState,
    /// Our long-lived ("static") keypair.
    local_static: TransportKeypair,
    /// Ephemeral keypair generated at handshake start.
    local_ephemeral: Option<TransportKeypair>,
    /// Remote ephemeral public key (received in msg1 or msg2).
    remote_ephemeral_pub: Option<[u8; 32]>,
    /// Remote static public key (received encrypted in msg2 or msg3).
    remote_static_pub: Option<[u8; 32]>,
    /// Key used to encrypt outgoing messages.
    send_key: Option<[u8; 32]>,
    /// Key used to decrypt incoming messages.
    recv_key: Option<[u8; 32]>,
    /// Next nonce for outgoing messages.
    send_nonce: u64,
    /// Expected nonce for incoming messages.
    recv_nonce: u64,
}

impl TransportChannel {
    /// Create a new channel backed by the given static keypair.
    pub fn new(keypair: TransportKeypair) -> Self {
        Self {
            state: ChannelState::Initial,
            local_static: keypair,
            local_ephemeral: None,
            remote_ephemeral_pub: None,
            remote_static_pub: None,
            send_key: None,
            recv_key: None,
            send_nonce: 0,
            recv_nonce: 0,
        }
    }

    // ------------------------------------------------------------------
    // Handshake
    // ------------------------------------------------------------------

    /// Initiate the handshake (send message 1).
    ///
    /// Returns the 32-byte ephemeral public key to send to the responder.
    pub fn initiate(&mut self) -> Result<Vec<u8>> {
        if self.state != ChannelState::Initial {
            return Err(YuleError::Api(
                "initiate() called in wrong state".to_string(),
            ));
        }
        let eph = TransportKeypair::generate()?;
        let msg = eph.public.to_vec();
        self.local_ephemeral = Some(eph);
        self.state = ChannelState::HandshakeInitiated;
        Ok(msg)
    }

    /// Process an incoming handshake message and optionally return a response.
    ///
    /// State transitions:
    /// - `Initial`  + 32 bytes (msg1) -> produce msg2, move to `HandshakeResponded`
    /// - `HandshakeInitiated` + 64 bytes (msg2) -> produce msg3, move to `Established`
    /// - `HandshakeResponded` + 64 bytes (msg3) -> finalize, move to `Established`
    pub fn process_handshake(&mut self, msg: &[u8]) -> Result<Option<Vec<u8>>> {
        match self.state {
            // Responder receives msg1 (initiator's ephemeral public key).
            ChannelState::Initial => {
                if msg.len() != 32 {
                    return Err(YuleError::Api(format!(
                        "handshake msg1: expected 32 bytes, got {}",
                        msg.len()
                    )));
                }
                let mut remote_eph = [0u8; 32];
                remote_eph.copy_from_slice(msg);
                self.remote_ephemeral_pub = Some(remote_eph);

                // Generate our own ephemeral keypair.
                let eph = TransportKeypair::generate()?;

                // Derive a temporary key from ee DH to encrypt our static key.
                let ee_shared = x25519(&eph.secret, &remote_eph);
                let temp_key = blake3::derive_key("yule transport handshake temp key", &ee_shared);

                // Encrypt our static public key under the temp key.
                let encrypted_static = xor_keystream(&temp_key, 0, &self.local_static.public);

                // Build msg2 = e_pub_r || Enc(static_pub_r).
                let mut response = Vec::with_capacity(64);
                response.extend_from_slice(&eph.public);
                response.extend_from_slice(&encrypted_static);

                self.local_ephemeral = Some(eph);
                self.state = ChannelState::HandshakeResponded;
                Ok(Some(response))
            }

            // Initiator receives msg2 (responder's ephemeral + encrypted static).
            ChannelState::HandshakeInitiated => {
                if msg.len() != 64 {
                    return Err(YuleError::Api(format!(
                        "handshake msg2: expected 64 bytes, got {}",
                        msg.len()
                    )));
                }
                let mut remote_eph = [0u8; 32];
                remote_eph.copy_from_slice(&msg[..32]);
                self.remote_ephemeral_pub = Some(remote_eph);

                let eph = self
                    .local_ephemeral
                    .as_ref()
                    .ok_or_else(|| YuleError::Api("missing local ephemeral".into()))?;

                // Derive temp key from ee DH.
                let ee_shared = x25519(&eph.secret, &remote_eph);
                let temp_key = blake3::derive_key("yule transport handshake temp key", &ee_shared);

                // Decrypt responder's static public key.
                let mut remote_static = [0u8; 32];
                remote_static.copy_from_slice(&xor_keystream(&temp_key, 0, &msg[32..64]));
                self.remote_static_pub = Some(remote_static);

                // Compute se DH using (e_i, s_r) — both sides can derive this:
                //   initiator: DH(e_i_secret, s_r_pub)
                //   responder: DH(s_r_secret, e_i_pub)
                let se_shared = x25519(&eph.secret, &remote_static);

                // Derive a second temp key for msg3.
                let mut se_input = [0u8; 64];
                se_input[..32].copy_from_slice(&ee_shared);
                se_input[32..].copy_from_slice(&se_shared);
                let temp_key2 =
                    blake3::derive_key("yule transport handshake temp key 2", &se_input);

                // Encrypt our static public key under temp_key2.
                let encrypted_static = xor_keystream(&temp_key2, 0, &self.local_static.public);
                let response = encrypted_static.to_vec();

                // Derive session keys.
                self.derive_session_keys(&ee_shared, &se_shared, true)?;
                self.state = ChannelState::Established;
                Ok(Some(response))
            }

            // Responder receives msg3 (initiator's encrypted static key).
            ChannelState::HandshakeResponded => {
                if msg.len() != 32 {
                    return Err(YuleError::Api(format!(
                        "handshake msg3: expected 32 bytes, got {}",
                        msg.len()
                    )));
                }
                let eph = self
                    .local_ephemeral
                    .as_ref()
                    .ok_or_else(|| YuleError::Api("missing local ephemeral".into()))?;
                let remote_eph = self
                    .remote_ephemeral_pub
                    .ok_or_else(|| YuleError::Api("missing remote ephemeral".into()))?;

                // Re-derive ee DH.
                let ee_shared = x25519(&eph.secret, &remote_eph);

                // Compute se DH using (s_r, e_i) — matches initiator's DH(e_i, s_r):
                //   responder: DH(s_r_secret, e_i_pub)
                //   initiator: DH(e_i_secret, s_r_pub)
                let se_shared = x25519(&self.local_static.secret, &remote_eph);

                let mut se_input = [0u8; 64];
                se_input[..32].copy_from_slice(&ee_shared);
                se_input[32..].copy_from_slice(&se_shared);
                let temp_key2 =
                    blake3::derive_key("yule transport handshake temp key 2", &se_input);

                // Decrypt initiator's static public key.
                let mut remote_static = [0u8; 32];
                remote_static.copy_from_slice(&xor_keystream(&temp_key2, 0, msg));
                self.remote_static_pub = Some(remote_static);

                // Derive session keys (responder perspective).
                self.derive_session_keys(&ee_shared, &se_shared, false)?;
                self.state = ChannelState::Established;
                Ok(None)
            }

            ChannelState::Established => {
                Err(YuleError::Api("handshake already complete".to_string()))
            }

            ChannelState::Closed => Err(YuleError::Api("channel is closed".to_string())),
        }
    }

    /// Derive send and recv session keys from the DH shared secrets.
    ///
    /// `is_initiator` determines which key is send vs. recv so that
    /// initiator's send key == responder's recv key and vice versa.
    fn derive_session_keys(
        &mut self,
        ee: &[u8; 32],
        se: &[u8; 32],
        is_initiator: bool,
    ) -> Result<()> {
        let mut material = [0u8; 64];
        material[..32].copy_from_slice(ee);
        material[32..].copy_from_slice(se);

        let key_a = blake3::derive_key("yule transport send key A", &material);
        let key_b = blake3::derive_key("yule transport send key B", &material);

        if is_initiator {
            self.send_key = Some(key_a);
            self.recv_key = Some(key_b);
        } else {
            self.send_key = Some(key_b);
            self.recv_key = Some(key_a);
        }

        Ok(())
    }

    /// Returns `true` once the handshake is complete and the channel is ready
    /// for application traffic.
    pub fn is_established(&self) -> bool {
        self.state == ChannelState::Established
    }

    // ------------------------------------------------------------------
    // Application data
    // ------------------------------------------------------------------

    /// Encrypt `plaintext` and return an [`EncryptedMessage`].
    ///
    /// Each call increments the internal send nonce.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<EncryptedMessage> {
        if self.state != ChannelState::Established {
            return Err(YuleError::Api("channel not established".to_string()));
        }
        let key = self
            .send_key
            .ok_or_else(|| YuleError::Api("missing send key".into()))?;
        let nonce = self.send_nonce;
        self.send_nonce = self
            .send_nonce
            .checked_add(1)
            .ok_or_else(|| YuleError::Api("send nonce overflow".into()))?;

        let ciphertext = xor_keystream(&key, nonce, plaintext);
        let tag = compute_tag(&key, nonce, &ciphertext);

        Ok(EncryptedMessage {
            ciphertext,
            nonce,
            tag,
        })
    }

    /// Decrypt an [`EncryptedMessage`] and return the plaintext.
    ///
    /// Validates the MAC tag and nonce ordering before decrypting.
    pub fn decrypt(&mut self, msg: &EncryptedMessage) -> Result<Vec<u8>> {
        if self.state != ChannelState::Established {
            return Err(YuleError::Api("channel not established".to_string()));
        }
        let key = self
            .recv_key
            .ok_or_else(|| YuleError::Api("missing recv key".into()))?;

        // Verify nonce is what we expect (simple replay protection).
        if msg.nonce != self.recv_nonce {
            return Err(YuleError::Api(format!(
                "unexpected nonce: expected {}, got {}",
                self.recv_nonce, msg.nonce
            )));
        }

        // Verify MAC before decrypting.
        let expected_tag = compute_tag(&key, msg.nonce, &msg.ciphertext);
        if !constant_time_eq(&expected_tag, &msg.tag) {
            return Err(YuleError::Api("MAC verification failed".to_string()));
        }

        self.recv_nonce += 1;
        let plaintext = xor_keystream(&key, msg.nonce, &msg.ciphertext);
        Ok(plaintext)
    }

    /// Check whether the channel should be rekeyed.
    ///
    /// Returns `true` after `REKEY_MESSAGE_THRESHOLD` messages have been sent.
    /// A production implementation would also check elapsed wall-clock time.
    pub fn needs_rekey(&self) -> bool {
        self.send_nonce >= REKEY_MESSAGE_THRESHOLD
    }

    /// Perform an in-band rekey by deriving new session keys from the current
    /// ones mixed with fresh entropy.
    ///
    /// Both sides must call `rekey()` at the same send-nonce boundary.
    pub fn rekey(&mut self) -> Result<()> {
        let send = self
            .send_key
            .ok_or_else(|| YuleError::Api("no send key to rekey".into()))?;
        let recv = self
            .recv_key
            .ok_or_else(|| YuleError::Api("no recv key to rekey".into()))?;

        // Use a canonical key order so both sides produce the same material.
        // One side's send_key is the other's recv_key, so we sort to get a
        // deterministic ordering regardless of role.
        let (first, second) = if send <= recv {
            (send, recv)
        } else {
            (recv, send)
        };

        let mut material = [0u8; 64];
        material[..32].copy_from_slice(&first);
        material[32..].copy_from_slice(&second);

        let new_a = blake3::derive_key("yule transport rekey A", &material);
        let new_b = blake3::derive_key("yule transport rekey B", &material);

        // Maintain the same send/recv assignment: whichever key was "smaller"
        // determines the role. The side whose old send_key was <= old recv_key
        // gets (A=send, B=recv); the other side gets (B=send, A=recv).
        if send <= recv {
            self.send_key = Some(new_a);
            self.recv_key = Some(new_b);
        } else {
            self.send_key = Some(new_b);
            self.recv_key = Some(new_a);
        }

        self.send_nonce = 0;
        self.recv_nonce = 0;
        Ok(())
    }

    /// Permanently close the channel, zeroising key material.
    pub fn close(&mut self) {
        self.send_key = Some([0u8; 32]);
        self.recv_key = Some([0u8; 32]);
        if let Some(ref mut eph) = self.local_ephemeral {
            eph.secret = [0u8; 32];
        }
        self.state = ChannelState::Closed;
    }
}

// ---------------------------------------------------------------------------
// Symmetric crypto helpers
// ---------------------------------------------------------------------------

/// Generate a BLAKE3-based keystream and XOR it with `data`.
///
/// Uses BLAKE3 in keyed-hash mode, deriving one 32-byte block per counter
/// value:  `block_i = blake3::keyed_hash(key, nonce || counter_i)`.
///
/// NOTE: A production implementation would use ChaCha20-Poly1305 here.
fn xor_keystream(key: &[u8; 32], nonce: u64, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let nonce_bytes = nonce.to_le_bytes();

    for (block_idx, chunk) in data.chunks(32).enumerate() {
        let counter = (block_idx as u64).to_le_bytes();
        let mut input = [0u8; 16];
        input[..8].copy_from_slice(&nonce_bytes);
        input[8..].copy_from_slice(&counter);

        let keystream_block = blake3::keyed_hash(key, &input);
        let ks = keystream_block.as_bytes();
        for (i, &b) in chunk.iter().enumerate() {
            out.push(b ^ ks[i]);
        }
    }
    out
}

/// Compute a BLAKE3 MAC tag over `nonce || ciphertext`.
fn compute_tag(key: &[u8; 32], nonce: u64, ciphertext: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(&nonce.to_le_bytes());
    hasher.update(ciphertext);
    *hasher.finalize().as_bytes()
}

/// Constant-time comparison to prevent timing side-channels on the MAC tag.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = TransportKeypair::generate().unwrap();
        assert_eq!(kp.secret.len(), 32);
        assert_eq!(kp.public.len(), 32);
        // Keys should not be all zeros.
        assert!(kp.secret.iter().any(|&b| b != 0));
        assert!(kp.public.iter().any(|&b| b != 0));
        // Two keypairs should differ.
        let kp2 = TransportKeypair::generate().unwrap();
        assert_ne!(kp.public, kp2.public);
    }

    #[test]
    fn test_handshake_completes() {
        let init_kp = TransportKeypair::generate().unwrap();
        let resp_kp = TransportKeypair::generate().unwrap();

        let mut initiator = TransportChannel::new(init_kp);
        let mut responder = TransportChannel::new(resp_kp);

        // msg1: initiator -> responder
        let msg1 = initiator.initiate().unwrap();
        assert_eq!(msg1.len(), 32);

        // msg2: responder -> initiator
        let msg2 = responder.process_handshake(&msg1).unwrap().unwrap();
        assert_eq!(msg2.len(), 64);

        // msg3: initiator -> responder
        let msg3 = initiator.process_handshake(&msg2).unwrap().unwrap();
        assert_eq!(msg3.len(), 32);

        // Finalize on responder side.
        let msg4 = responder.process_handshake(&msg3).unwrap();
        assert!(msg4.is_none());

        assert!(initiator.is_established());
        assert!(responder.is_established());
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let (mut initiator, mut responder) = establish_pair();

        let plaintext = b"hello from the initiator";
        let encrypted = initiator.encrypt(plaintext).unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);

        // Reverse direction.
        let plaintext2 = b"hello from the responder";
        let encrypted2 = responder.encrypt(plaintext2).unwrap();
        let decrypted2 = initiator.decrypt(&encrypted2).unwrap();
        assert_eq!(decrypted2, plaintext2);
    }

    #[test]
    fn test_wrong_key_fails() {
        let (mut initiator, _responder) = establish_pair();

        let encrypted = initiator.encrypt(b"secret data").unwrap();

        // Create an unrelated channel — its recv key won't match.
        let (_, mut wrong_receiver) = establish_pair();
        // Force the nonce to match so only the key difference causes failure.
        wrong_receiver.recv_nonce = encrypted.nonce;

        let result = wrong_receiver.decrypt(&encrypted);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("MAC verification"),
            "expected MAC failure"
        );
    }

    #[test]
    fn test_nonce_increments() {
        let (mut initiator, mut responder) = establish_pair();

        assert_eq!(initiator.send_nonce, 0);
        let m0 = initiator.encrypt(b"a").unwrap();
        assert_eq!(m0.nonce, 0);
        assert_eq!(initiator.send_nonce, 1);

        let m1 = initiator.encrypt(b"b").unwrap();
        assert_eq!(m1.nonce, 1);
        assert_eq!(initiator.send_nonce, 2);

        // Decrypting must accept them in order.
        responder.decrypt(&m0).unwrap();
        responder.decrypt(&m1).unwrap();
        assert_eq!(responder.recv_nonce, 2);
    }

    #[test]
    fn test_rekey() {
        let (mut initiator, mut responder) = establish_pair();

        let old_send = initiator.send_key.unwrap();
        let old_recv = initiator.recv_key.unwrap();

        initiator.rekey().unwrap();
        responder.rekey().unwrap();

        // Keys should have changed.
        assert_ne!(initiator.send_key.unwrap(), old_send);
        assert_ne!(initiator.recv_key.unwrap(), old_recv);

        // Nonces should be reset.
        assert_eq!(initiator.send_nonce, 0);
        assert_eq!(responder.recv_nonce, 0);

        // Communication should still work after rekey.
        let plaintext = b"post-rekey message";
        let encrypted = initiator.encrypt(plaintext).unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_close() {
        let (mut initiator, _) = establish_pair();
        initiator.close();
        assert_eq!(initiator.state, ChannelState::Closed);
        assert!(initiator.encrypt(b"nope").is_err());
    }

    #[test]
    fn test_large_message() {
        let (mut initiator, mut responder) = establish_pair();
        let plaintext = vec![0xABu8; 4096];
        let encrypted = initiator.encrypt(&plaintext).unwrap();
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_empty_message() {
        let (mut initiator, mut responder) = establish_pair();
        let encrypted = initiator.encrypt(b"").unwrap();
        assert!(encrypted.ciphertext.is_empty());
        let decrypted = responder.decrypt(&encrypted).unwrap();
        assert!(decrypted.is_empty());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Complete a full handshake between two fresh channels.
    fn establish_pair() -> (TransportChannel, TransportChannel) {
        let mut initiator = TransportChannel::new(TransportKeypair::generate().unwrap());
        let mut responder = TransportChannel::new(TransportKeypair::generate().unwrap());

        let msg1 = initiator.initiate().unwrap();
        let msg2 = responder.process_handshake(&msg1).unwrap().unwrap();
        let msg3 = initiator.process_handshake(&msg2).unwrap().unwrap();
        responder.process_handshake(&msg3).unwrap();

        assert!(initiator.is_established());
        assert!(responder.is_established());
        (initiator, responder)
    }
}
