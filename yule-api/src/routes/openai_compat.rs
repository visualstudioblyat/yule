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

use yule_core::chat_template::Role;

fn gen_id() -> String {
    format!(
        "chatcmpl-{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn parse_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
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
        Ok(stream_openai_response(token_rx).into_response())
    } else {
        Ok(collect_openai_response(token_rx).await.into_response())
    }
}

async fn collect_openai_response(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>,
) -> Json<ChatCompletionResponse> {
    let mut text = String::new();
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut finish_reason = "length".to_string();

    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token(t) => text.push_str(&t),
            TokenEvent::Done {
                prompt_tokens,
                completion_tokens,
                finish_reason: fr,
                ..
            } => {
                usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                };
                finish_reason = fr;
            }
            TokenEvent::Error(e) => {
                text = format!("error: {e}");
                finish_reason = "error".to_string();
            }
        }
    }

    Json(ChatCompletionResponse {
        id: gen_id(),
        object: "chat.completion".into(),
        created: now_epoch(),
        model: "yule".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason,
        }],
        usage,
    })
}

fn stream_openai_response(
    rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>,
) -> Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = gen_id();
    let created = now_epoch();

    let stream = UnboundedReceiverStream::new(rx).map(move |event| {
        let sse_event = match event {
            TokenEvent::Token(t) => {
                let chunk = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: "yule".into(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content: Some(t),
                        },
                        finish_reason: None,
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap())
            }
            TokenEvent::Done { finish_reason, .. } => {
                let chunk = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: "yule".into(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(finish_reason),
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap())
            }
            TokenEvent::Error(e) => Event::default().data(format!("{{\"error\":\"{e}\"}}")),
        };
        Ok(sse_event)
    });

    Sse::new(stream)
}

pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let name = state
        .model_info
        .metadata
        .name
        .clone()
        .unwrap_or_else(|| "yule-model".into());

    Json(ModelListResponse {
        object: "list".into(),
        data: vec![OpenAiModelInfo {
            id: name,
            object: "model".into(),
            owned_by: "local".into(),
        }],
    })
}
