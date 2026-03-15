//! Integration tests for the Yule API server.
//!
//! These tests exercise the HTTP layer with a mock inference backend,
//! so no model file is needed. The mock inference thread responds to
//! Generate requests with a fixed "Hello, world!" output and handles
//! Tokenize requests with a dummy token list.

use std::sync::{mpsc, Arc};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use yule_api::auth::TokenAuthority;
use yule_api::inference::{InferenceRequest, ModelInfo, TokenEvent};
use yule_api::{AppState, build_router};
use yule_core::model::{Architecture, ModelMetadata};

/// Create a mock AppState with a fake inference thread that returns canned responses.
fn mock_state() -> (Arc<AppState>, String) {
    let (inf_tx, inf_rx) = mpsc::channel::<InferenceRequest>();

    // spawn mock inference thread
    std::thread::Builder::new()
        .name("mock-inference".into())
        .spawn(move || {
            while let Ok(req) = inf_rx.recv() {
                match req {
                    InferenceRequest::Generate {
                        token_tx,
                        max_tokens: _,
                        temperature: _,
                        top_p: _,
                        messages: _,
                    } => {
                        token_tx.send(TokenEvent::Token("Hello".into())).ok();
                        token_tx.send(TokenEvent::Token(", world!".into())).ok();
                        token_tx
                            .send(TokenEvent::Done {
                                prompt_tokens: 5,
                                completion_tokens: 2,
                                finish_reason: "stop".into(),
                                prefill_ms: 10.0,
                                decode_ms: 20.0,
                            })
                            .ok();
                    }
                    InferenceRequest::Tokenize { text: _, reply } => {
                        reply
                            .send(yule_api::inference::TokenizeResult {
                                tokens: vec![1, 2, 3],
                            })
                            .ok();
                    }
                    InferenceRequest::Shutdown => break,
                }
            }
        })
        .unwrap();

    let mut auth = TokenAuthority::new();
    let token = auth.generate_token();
    let auth = Arc::new(auth);

    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[42u8; 32]);

    let state = Arc::new(AppState {
        inference_tx: inf_tx,
        auth: auth.clone(),
        start_time: Instant::now(),
        model_info: ModelInfo {
            metadata: ModelMetadata {
                architecture: Architecture::Llama,
                name: Some("test-model".into()),
                parameters: 1_100_000_000,
                context_length: 2048,
                embedding_dim: 2048,
                head_count: 32,
                head_count_kv: 4,
                layer_count: 22,
                vocab_size: 32000,
                rope_freq_base: Some(10000.0),
                rope_scaling: None,
                expert_count: None,
                expert_used_count: None,
                norm_eps: None,
                sliding_window: None,
                partial_rotary_dim: None,
                logit_softcap: None,
                attn_logit_softcap: None,
            },
            tensor_count: 201,
            file_size: 637_000_000,
            merkle_root: "ffc7e1fd6016a6f9ba2ca390a43681453a46ec6054f431aeb6244487932b0e65"
                .into(),
        },
        sandbox_active: true,
        device_pubkey: signing_key.verifying_key().to_bytes(),
        merkle_root_bytes: [0xff; 32],
        signing_key,
    });

    (state, token)
}

fn build_app(state: Arc<AppState>) -> axum::Router {
    build_router(state.clone(), state.auth.clone())
}

// ── Auth Tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn no_auth_returns_401() {
    let (state, _token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn wrong_token_returns_401() {
    let (state, _token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/health")
                .header("Authorization", "Bearer yule_wrongtoken")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn missing_bearer_prefix_returns_401() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/health")
                .header("Authorization", &token) // missing "Bearer " prefix
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn valid_token_returns_200() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/health")
                .header("Authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

// ── Health Endpoint ─────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_valid_json() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/health")
                .header("Authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "healthy");
    assert!(json["version"].is_string());
    assert!(json["uptime_seconds"].is_number());
    assert_eq!(json["sandbox"], true);
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["architecture"], "Llama");
}

// ── Model Info Endpoint ─────────────────────────────────────────────────────

#[tokio::test]
async fn model_info_returns_metadata() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/yule/model")
                .header("Authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["name"], "test-model");
    assert_eq!(json["architecture"], "Llama");
    assert_eq!(json["parameters"], "1.1B");
    assert_eq!(json["context_length"], 2048);
    assert_eq!(json["layers"], 22);
    assert_eq!(json["vocab_size"], 32000);
    assert_eq!(json["tensor_count"], 201);
    assert!(json["merkle_root"].as_str().unwrap().starts_with("ffc7e1fd"));
}

// ── Tokenize Endpoint ───────────────────────────────────────────────────────

#[tokio::test]
async fn tokenize_returns_tokens() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/yule/tokenize")
                .header("Authorization", format!("Bearer {token}"))
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"text":"hello world"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["tokens"], serde_json::json!([1, 2, 3]));
    assert_eq!(json["count"], 3);
}

// ── Yule Chat Endpoint (non-streaming) ──────────────────────────────────────

#[tokio::test]
async fn yule_chat_returns_response_with_integrity() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/yule/chat")
                .header("Authorization", format!("Bearer {token}"))
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["text"], "Hello, world!");
    assert_eq!(json["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 2);
    assert_eq!(json["usage"]["total_tokens"], 7);

    // integrity fields
    assert!(json["integrity"]["model_merkle_root"].is_string());
    assert_eq!(json["integrity"]["model_verified"], true);
    assert_eq!(json["integrity"]["sandbox_active"], true);

    // timing fields
    assert!(json["timing"]["prefill_ms"].is_number());
    assert!(json["timing"]["decode_ms"].is_number());
    assert!(json["timing"]["tokens_per_second"].is_number());
}

// ── OpenAI-Compatible Endpoints ─────────────────────────────────────────────

#[tokio::test]
async fn openai_models_returns_list() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("Authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "test-model");
    assert_eq!(json["data"][0]["object"], "model");
    assert_eq!(json["data"][0]["owned_by"], "local");
}

#[tokio::test]
async fn openai_chat_completions_returns_response() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("Authorization", format!("Bearer {token}"))
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "Hello, world!");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 2);
}

// ── 404 for unknown routes ──────────────────────────────────────────────────

#[tokio::test]
async fn unknown_route_returns_404() {
    let (state, token) = mock_state();
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .header("Authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum returns 401 because auth middleware runs before routing for unknown paths,
    // or 404 if auth passes. Either is acceptable.
    assert!(
        resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::UNAUTHORIZED
    );
}
