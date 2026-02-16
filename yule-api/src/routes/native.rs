use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::AppState;
use crate::inference::{InferenceRequest, TokenEvent};
use crate::types::*;

use yule_attest::InferenceAttestation;
use yule_attest::log::AuditLog;
use yule_attest::session::AttestationSession;
use yule_core::chat_template::Role;

fn gen_id() -> String {
    format!(
        "yule-{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

fn parse_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

/// Create attestation record for a completed inference and log it.
/// Returns (session_id, device_pubkey_hex) on success.
fn attest_inference(
    state: &AppState,
    prompt_text: &str,
    output_text: &str,
    tokens_generated: u64,
    temperature: f32,
    top_p: f32,
) -> Option<(String, String)> {
    let mut session = AttestationSession::new();
    session.set_model(
        state
            .model_info
            .metadata
            .name
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        state.merkle_root_bytes,
        None,
        true,
    );
    session.set_sandbox(state.sandbox_active, 32 * 1024 * 1024 * 1024);

    let inference = InferenceAttestation {
        tokens_generated,
        prompt_hash: blake3::hash(prompt_text.as_bytes()).into(),
        output_hash: blake3::hash(output_text.as_bytes()).into(),
        temperature,
        top_p,
    };

    let audit_log = match AuditLog::default_path() {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("attestation: can't open audit log: {e}");
            return None;
        }
    };

    let prev_hash = audit_log.last_hash().unwrap_or([0u8; 32]);
    let session_id = session.session_id().to_string();

    let record = match session.finalize(inference, &state.signing_key, prev_hash) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("attestation: finalize failed: {e}");
            return None;
        }
    };

    if let Err(e) = audit_log.append(&record) {
        tracing::warn!("attestation: append failed: {e}");
    }

    let pubkey_hex: String = state
        .device_pubkey
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    Some((session_id, pubkey_hex))
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();
    let info = &state.model_info;

    Json(HealthResponse {
        status: "healthy".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        uptime_seconds: uptime,
        model: info.metadata.name.clone(),
        architecture: Some(format!("{:?}", info.metadata.architecture)),
        sandbox: state.sandbox_active,
    })
}

pub async fn model_info(State(state): State<Arc<AppState>>) -> Json<YuleModelInfo> {
    let info = &state.model_info;
    let meta = &info.metadata;

    let params = if meta.parameters >= 1_000_000_000 {
        format!("{:.1}B", meta.parameters as f64 / 1e9)
    } else if meta.parameters >= 1_000_000 {
        format!("{:.1}M", meta.parameters as f64 / 1e6)
    } else {
        meta.parameters.to_string()
    };

    Json(YuleModelInfo {
        name: meta.name.clone(),
        architecture: format!("{:?}", meta.architecture),
        parameters: params,
        context_length: meta.context_length,
        embedding_dim: meta.embedding_dim,
        layers: meta.layer_count,
        vocab_size: meta.vocab_size,
        tensor_count: info.tensor_count,
        merkle_root: info.merkle_root.clone(),
    })
}

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<YuleChatRequest>,
) -> impl IntoResponse {
    let messages: Vec<(Role, String)> = req
        .messages
        .iter()
        .map(|m| (parse_role(&m.role), m.content.clone()))
        .collect();

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let stream = req.stream.unwrap_or(false);

    // capture prompt for attestation hashing
    let prompt_text: String = req
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel();

    if state
        .inference_tx
        .send(InferenceRequest::Generate {
            messages,
            max_tokens,
            temperature,
            top_p,
            token_tx,
        })
        .is_err()
    {
        return Err(axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    if stream {
        Ok(stream_yule_response(state, temperature, top_p, prompt_text, token_rx).into_response())
    } else {
        Ok(
            collect_yule_response(state, temperature, top_p, prompt_text, token_rx)
                .await
                .into_response(),
        )
    }
}

async fn collect_yule_response(
    state: Arc<AppState>,
    temperature: f32,
    top_p: f32,
    prompt_text: String,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>,
) -> Json<YuleChatResponse> {
    let mut text = String::new();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut finish_reason = "length".to_string();
    let mut prefill_ms = 0.0;
    let mut decode_ms = 0.0;

    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token(t) => text.push_str(&t),
            TokenEvent::Done {
                prompt_tokens,
                completion_tokens,
                finish_reason: fr,
                prefill_ms: pm,
                decode_ms: dm,
            } => {
                usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                };
                finish_reason = fr;
                prefill_ms = pm;
                decode_ms = dm;
            }
            TokenEvent::Error(e) => {
                text = format!("error: {e}");
                finish_reason = "error".to_string();
            }
        }
    }

    let tps = if decode_ms > 0.0 {
        usage.completion_tokens as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    // sign attestation record for this inference
    let attestation = attest_inference(
        &state,
        &prompt_text,
        &text,
        usage.completion_tokens as u64,
        temperature,
        top_p,
    );

    let (attestation_id, device_pubkey) = match attestation {
        Some((id, pk)) => (Some(id), Some(pk)),
        None => (None, None),
    };

    Json(YuleChatResponse {
        id: gen_id(),
        text,
        finish_reason,
        usage,
        integrity: IntegrityInfo {
            model_merkle_root: state.model_info.merkle_root.clone(),
            model_verified: true,
            sandbox_active: state.sandbox_active,
            attestation_id,
            device_pubkey,
        },
        timing: TimingInfo {
            prefill_ms,
            decode_ms,
            tokens_per_second: tps,
        },
    })
}

fn stream_yule_response(
    state: Arc<AppState>,
    temperature: f32,
    top_p: f32,
    prompt_text: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>,
) -> Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>> {
    use std::sync::Mutex;
    let collected_text = Arc::new(Mutex::new(String::new()));

    let stream = UnboundedReceiverStream::new(rx).map(move |event| {
        let sse_event = match event {
            TokenEvent::Token(t) => {
                collected_text.lock().unwrap().push_str(&t);
                let data = YuleStreamEvent {
                    event_type: "token".into(),
                    text: Some(t),
                    usage: None,
                    integrity: None,
                    timing: None,
                    finish_reason: None,
                    error: None,
                };
                Event::default().data(serde_json::to_string(&data).unwrap())
            }
            TokenEvent::Done {
                prompt_tokens,
                completion_tokens,
                finish_reason,
                prefill_ms,
                decode_ms,
            } => {
                let tps = if decode_ms > 0.0 {
                    completion_tokens as f64 / (decode_ms / 1000.0)
                } else {
                    0.0
                };

                let output_text = collected_text.lock().unwrap().clone();
                let attestation = attest_inference(
                    &state,
                    &prompt_text,
                    &output_text,
                    completion_tokens as u64,
                    temperature,
                    top_p,
                );
                let (attestation_id, device_pubkey) = match attestation {
                    Some((id, pk)) => (Some(id), Some(pk)),
                    None => (None, None),
                };

                let data = YuleStreamEvent {
                    event_type: "done".into(),
                    text: None,
                    usage: Some(Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    }),
                    integrity: Some(IntegrityInfo {
                        model_merkle_root: state.model_info.merkle_root.clone(),
                        model_verified: true,
                        sandbox_active: state.sandbox_active,
                        attestation_id,
                        device_pubkey,
                    }),
                    timing: Some(TimingInfo {
                        prefill_ms,
                        decode_ms,
                        tokens_per_second: tps,
                    }),
                    finish_reason: Some(finish_reason),
                    error: None,
                };
                Event::default().data(serde_json::to_string(&data).unwrap())
            }
            TokenEvent::Error(e) => {
                let data = YuleStreamEvent {
                    event_type: "error".into(),
                    text: None,
                    usage: None,
                    integrity: None,
                    timing: None,
                    finish_reason: None,
                    error: Some(e),
                };
                Event::default().data(serde_json::to_string(&data).unwrap())
            }
        };
        Ok(sse_event)
    });

    Sse::new(stream)
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> impl IntoResponse {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();

    if state
        .inference_tx
        .send(InferenceRequest::Tokenize {
            text: req.text,
            reply: reply_tx,
        })
        .is_err()
    {
        return Err(axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    match reply_rx.await {
        Ok(result) => {
            let count = result.tokens.len();
            Ok(Json(TokenizeResponse {
                tokens: result.tokens,
                count,
            }))
        }
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}
