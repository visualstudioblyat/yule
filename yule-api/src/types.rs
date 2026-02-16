use serde::{Deserialize, Serialize};

// -- OpenAI compat types --

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<OpenAiModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

// -- SSE streaming (OpenAI format) --

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// -- Yule-native types --

#[derive(Debug, Deserialize)]
pub struct YuleChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct YuleChatResponse {
    pub id: String,
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
    pub integrity: IntegrityInfo,
    pub timing: TimingInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntegrityInfo {
    pub model_merkle_root: String,
    pub model_verified: bool,
    pub sandbox_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_pubkey: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TimingInfo {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub model: Option<String>,
    pub architecture: Option<String>,
    pub sandbox: bool,
}

#[derive(Debug, Serialize)]
pub struct YuleModelInfo {
    pub name: Option<String>,
    pub architecture: String,
    pub parameters: String,
    pub context_length: u32,
    pub embedding_dim: u32,
    pub layers: u32,
    pub vocab_size: u32,
    pub tensor_count: usize,
    pub merkle_root: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}

// -- Yule SSE stream event --

#[derive(Debug, Serialize)]
pub struct YuleStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrity: Option<IntegrityInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yule_chat_request_deserialize() {
        let json = r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":100,"temperature":0.5}"#;
        let req: YuleChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.stream, None);
    }

    #[test]
    fn yule_chat_request_defaults() {
        let json = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let req: YuleChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);
        assert_eq!(req.stream, None);
    }

    #[test]
    fn openai_request_deserialize() {
        let json = r#"{"model":"gpt-4","messages":[{"role":"system","content":"you are helpful"},{"role":"user","content":"hi"}],"stream":true}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn yule_chat_response_serialize() {
        let resp = YuleChatResponse {
            id: "test-123".into(),
            text: "hello world".into(),
            finish_reason: "stop".into(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            integrity: IntegrityInfo {
                model_merkle_root: "abcdef".into(),
                model_verified: true,
                sandbox_active: true,
                attestation_id: None,
                device_pubkey: None,
            },
            timing: TimingInfo {
                prefill_ms: 100.0,
                decode_ms: 200.0,
                tokens_per_second: 25.0,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"model_merkle_root\":\"abcdef\""));
        assert!(json.contains("\"sandbox_active\":true"));
        assert!(json.contains("\"tokens_per_second\":25.0"));
    }

    #[test]
    fn stream_event_skips_none_fields() {
        let event = YuleStreamEvent {
            event_type: "token".into(),
            text: Some("hi".into()),
            usage: None,
            integrity: None,
            timing: None,
            finish_reason: None,
            error: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"text\":\"hi\""));
        assert!(!json.contains("usage"));
        assert!(!json.contains("integrity"));
        assert!(!json.contains("timing"));
    }

    #[test]
    fn stream_delta_skips_none() {
        let delta = StreamDelta {
            role: None,
            content: Some("word".into()),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(!json.contains("role"));
        assert!(json.contains("\"content\":\"word\""));
    }

    #[test]
    fn health_response_serialize() {
        let resp = HealthResponse {
            status: "healthy".into(),
            version: "0.1.0".into(),
            uptime_seconds: 42,
            model: Some("llama".into()),
            architecture: Some("LlamaForCausalLM".into()),
            sandbox: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["status"], "healthy");
        assert_eq!(parsed["uptime_seconds"], 42);
        assert_eq!(parsed["sandbox"], false);
    }

    #[test]
    fn tokenize_request_deserialize() {
        let json = r#"{"text":"hello world"}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "hello world");
    }
}
