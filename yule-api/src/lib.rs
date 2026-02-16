pub mod auth;
pub mod inference;
pub mod routes;
pub mod types;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::middleware;
use axum::routing::{get, post};
use tokio::net::TcpListener;

use crate::auth::TokenAuthority;
use crate::inference::{InferenceHandle, ModelInfo};
use yule_core::error::Result;

pub struct AppState {
    pub inference_tx: std::sync::mpsc::Sender<inference::InferenceRequest>,
    pub auth: Arc<TokenAuthority>,
    pub start_time: Instant,
    pub model_info: ModelInfo,
    pub sandbox_active: bool,
    pub device_pubkey: [u8; 32],
    pub merkle_root_bytes: [u8; 32],
    pub signing_key: ed25519_dalek::SigningKey,
}

pub struct ApiServer {
    bind: String,
    model_path: PathBuf,
    token: Option<String>,
    sandbox_active: bool,
}

impl ApiServer {
    pub fn new(
        bind: String,
        model_path: PathBuf,
        token: Option<String>,
        sandbox_active: bool,
    ) -> Self {
        Self {
            bind,
            model_path,
            token,
            sandbox_active,
        }
    }

    pub async fn run(self) -> Result<()> {
        eprintln!("loading model: {}", self.model_path.display());

        let handle = InferenceHandle::spawn(self.model_path)?;

        eprintln!(
            "model loaded: {:?} ({} tensors, merkle: {})",
            handle.model_info.metadata.architecture,
            handle.model_info.tensor_count,
            &handle.model_info.merkle_root[..16],
        );

        // load device signing key for attestation
        let key_store = yule_verify::keys::KeyStore::open()
            .map_err(|e| yule_core::error::YuleError::Api(format!("key store: {e}")))?;
        let signing_key = key_store
            .device_key()
            .map_err(|e| yule_core::error::YuleError::Api(format!("device key: {e}")))?;
        let device_pubkey = signing_key.verifying_key().to_bytes();

        // parse merkle root hex → bytes
        let merkle_root_bytes = parse_merkle_hex(&handle.model_info.merkle_root);

        eprintln!(
            "attestation: device key loaded, pubkey {}",
            hex::short(&device_pubkey)
        );

        let mut auth = match &self.token {
            Some(t) => TokenAuthority::from_existing(t),
            None => TokenAuthority::new(),
        };

        let token = match &self.token {
            Some(t) => t.clone(),
            None => auth.generate_token(),
        };

        eprintln!();
        eprintln!("  token: {token}");
        eprintln!();

        let auth = Arc::new(auth);

        let state = Arc::new(AppState {
            inference_tx: handle.tx.clone(),
            auth: auth.clone(),
            start_time: Instant::now(),
            model_info: handle.model_info.clone(),
            sandbox_active: self.sandbox_active,
            device_pubkey,
            merkle_root_bytes,
            signing_key,
        });

        let app = build_router(state, auth);

        let addr: SocketAddr = self
            .bind
            .parse()
            .map_err(|e| yule_core::error::YuleError::Api(format!("invalid bind addr: {e}")))?;

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| yule_core::error::YuleError::Api(format!("bind failed: {e}")))?;

        eprintln!("listening on {addr}");
        eprintln!("  yule api:  http://{addr}/yule/health");
        eprintln!("  openai:    http://{addr}/v1/chat/completions");

        axum::serve(listener, app)
            .await
            .map_err(|e| yule_core::error::YuleError::Api(format!("server error: {e}")))?;

        handle.shutdown();
        Ok(())
    }
}

fn parse_merkle_hex(hex_str: &str) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        if i * 2 + 2 <= hex_str.len() {
            *byte = u8::from_str_radix(&hex_str[i * 2..i * 2 + 2], 16).unwrap_or(0);
        }
    }
    bytes
}

mod hex {
    pub fn short(bytes: &[u8; 32]) -> String {
        bytes[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            + "..."
    }
}

fn build_router(state: Arc<AppState>, auth: Arc<TokenAuthority>) -> Router {
    use routes::{native, openai_compat};

    Router::new()
        .route("/yule/health", get(native::health))
        .route("/yule/model", get(native::model_info))
        .route("/yule/chat", post(native::chat))
        .route("/yule/tokenize", post(native::tokenize))
        .route(
            "/v1/chat/completions",
            post(openai_compat::chat_completions),
        )
        .route("/v1/models", get(openai_compat::models))
        .layer(middleware::from_fn(auth::require_auth))
        .layer(axum::Extension(auth))
        .with_state(state)
}
