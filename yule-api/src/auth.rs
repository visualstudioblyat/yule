use axum::Extension;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use std::sync::Arc;

const TOKEN_PREFIX: &str = "yule_";

pub struct TokenAuthority {
    master: [u8; 32],
    hashes: Vec<[u8; 32]>,
}

impl Default for TokenAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenAuthority {
    pub fn new() -> Self {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("os entropy failed");
        Self {
            master: seed,
            hashes: Vec::new(),
        }
    }

    pub fn generate_token(&mut self) -> String {
        let input = [
            &self.master[..],
            &(self.hashes.len() as u64).to_le_bytes(),
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                .to_le_bytes(),
        ]
        .concat();

        let derived = blake3::hash(&input);
        let hex: String = derived.as_bytes()[..24]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let token = format!("{TOKEN_PREFIX}{hex}");

        self.hashes.push(*blake3::hash(token.as_bytes()).as_bytes());
        token
    }

    pub fn from_existing(token: &str) -> Self {
        let mut auth = Self {
            master: [0u8; 32],
            hashes: Vec::new(),
        };
        auth.hashes.push(*blake3::hash(token.as_bytes()).as_bytes());
        auth
    }

    pub fn verify(&self, provided: &str) -> bool {
        let hash = *blake3::hash(provided.as_bytes()).as_bytes();
        self.hashes.iter().any(|h| h == &hash)
    }
}

pub async fn require_auth(
    Extension(auth): Extension<Arc<TokenAuthority>>,
    req: Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    let token = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match token {
        Some(t) if auth.verify(t) => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_and_verify() {
        let mut auth = TokenAuthority::new();
        let token = auth.generate_token();
        assert!(token.starts_with("yule_"));
        assert_eq!(token.len(), 5 + 48); // prefix + 24 bytes as hex
        assert!(auth.verify(&token));
    }

    #[test]
    fn reject_garbage() {
        let mut auth = TokenAuthority::new();
        let _ = auth.generate_token();
        assert!(!auth.verify("not-a-real-token"));
        assert!(!auth.verify("yule_000000000000000000000000000000000000000000000000"));
        assert!(!auth.verify(""));
    }

    #[test]
    fn multiple_tokens_all_valid() {
        let mut auth = TokenAuthority::new();
        let t1 = auth.generate_token();
        let t2 = auth.generate_token();
        let t3 = auth.generate_token();
        assert!(auth.verify(&t1));
        assert!(auth.verify(&t2));
        assert!(auth.verify(&t3));
    }

    #[test]
    fn tokens_are_unique() {
        let mut auth = TokenAuthority::new();
        let t1 = auth.generate_token();
        let t2 = auth.generate_token();
        assert_ne!(t1, t2);
    }

    #[test]
    fn from_existing_verifies() {
        let auth = TokenAuthority::from_existing("my-secret-token");
        assert!(auth.verify("my-secret-token"));
        assert!(!auth.verify("wrong-token"));
    }

    #[test]
    fn stores_hashes_not_plaintext() {
        let mut auth = TokenAuthority::new();
        let token = auth.generate_token();
        // hashes vec stores 32-byte blake3 digests, not the raw token
        for h in &auth.hashes {
            assert_ne!(h, token.as_bytes().get(..32).unwrap_or(&[]));
        }
    }
}
