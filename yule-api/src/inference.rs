use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use yule_core::chat_template::{ChatTemplate, Role};
use yule_core::error::{Result, YuleError};
use yule_core::gguf::GgufParser;
use yule_core::model::ModelMetadata;
use yule_core::tokenizer::{BpeTokenizer, Tokenizer};
use yule_infer::SamplingParams;
use yule_infer::model_runner::{ModelRunner, TransformerRunner};
use yule_infer::sampler::Sampler;
use yule_infer::weight_loader::{TransformerWeights, WeightStore};
use yule_verify::merkle::MerkleTree;

pub struct InferenceHandle {
    pub tx: mpsc::Sender<InferenceRequest>,
    pub model_info: ModelInfo,
    join: thread::JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub metadata: ModelMetadata,
    pub tensor_count: usize,
    pub file_size: u64,
    pub merkle_root: String,
}

pub enum InferenceRequest {
    Generate {
        messages: Vec<(Role, String)>,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        token_tx: tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    },
    Tokenize {
        text: String,
        reply: tokio::sync::oneshot::Sender<TokenizeResult>,
    },
    Shutdown,
}

pub enum TokenEvent {
    Token(String),
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: String,
        prefill_ms: f64,
        decode_ms: f64,
    },
    Error(String),
}

pub struct TokenizeResult {
    pub tokens: Vec<u32>,
}

impl InferenceHandle {
    pub fn spawn(model_path: PathBuf) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<InferenceRequest>();
        let (init_tx, init_rx) = mpsc::channel::<Result<ModelInfo>>();

        let join = thread::Builder::new()
            .name("inference".into())
            .spawn(move || match inference_thread_init(&model_path) {
                Ok(state) => {
                    init_tx.send(Ok(state.info.clone())).ok();
                    inference_loop(state, rx);
                }
                Err(e) => {
                    init_tx.send(Err(e)).ok();
                }
            })
            .map_err(|e| YuleError::Api(format!("failed to spawn inference thread: {e}")))?;

        let model_info = init_rx
            .recv()
            .map_err(|_| YuleError::Api("inference thread died during init".into()))??;

        Ok(Self {
            tx,
            model_info,
            join,
        })
    }

    pub fn shutdown(self) {
        let _ = self.tx.send(InferenceRequest::Shutdown);
        let _ = self.join.join();
    }
}

struct InferenceState {
    info: ModelInfo,
    runner: TransformerRunner<'static>,
    tokenizer: BpeTokenizer,
    chat_template: Option<ChatTemplate>,
}

fn inference_thread_init(model_path: &std::path::Path) -> Result<InferenceState> {
    let parser = GgufParser::new();
    let gguf = parser.parse_file(model_path)?;
    let loaded = gguf.to_loaded_model()?;

    let mmap = yule_core::mmap::mmap_model(model_path)?;

    let merkle_root = if (gguf.data_offset as usize) <= mmap.len() {
        let tree = MerkleTree::new();
        let root = tree.build(&mmap[gguf.data_offset as usize..]);
        root.hash.iter().map(|b| format!("{b:02x}")).collect()
    } else {
        "none".into()
    };

    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;
    let chat_template = ChatTemplate::for_architecture(&loaded.metadata.architecture);

    let info = ModelInfo {
        metadata: loaded.metadata.clone(),
        tensor_count: loaded.tensors.len(),
        file_size: loaded.file_size,
        merkle_root,
    };

    // intentional leak — mmap lives for entire server lifetime on this thread
    let mmap_ref: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
    let mmap_static: &'static [u8] = mmap_ref.as_ref();

    let store = WeightStore::from_gguf(&gguf, mmap_static)?;
    let weights = TransformerWeights::new(store);
    let runner = TransformerRunner::new(weights)?;

    Ok(InferenceState {
        info,
        runner,
        tokenizer,
        chat_template,
    })
}

fn inference_loop(mut state: InferenceState, rx: mpsc::Receiver<InferenceRequest>) {
    while let Ok(req) = rx.recv() {
        match req {
            InferenceRequest::Generate {
                messages,
                max_tokens,
                temperature,
                top_p,
                token_tx,
            } => {
                handle_generate(
                    &mut state,
                    messages,
                    max_tokens,
                    temperature,
                    top_p,
                    &token_tx,
                );
            }
            InferenceRequest::Tokenize { text, reply } => match state.tokenizer.encode(&text) {
                Ok(tokens) => {
                    reply.send(TokenizeResult { tokens }).ok();
                }
                Err(_) => {
                    reply.send(TokenizeResult { tokens: vec![] }).ok();
                }
            },
            InferenceRequest::Shutdown => break,
        }
    }
}

fn handle_generate(
    state: &mut InferenceState,
    messages: Vec<(Role, String)>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    token_tx: &tokio::sync::mpsc::UnboundedSender<TokenEvent>,
) {
    state.runner.reset();

    // build prompt from messages
    let prompt = if let Some(ref tmpl) = state.chat_template {
        let msg_refs: Vec<(Role, &str)> = messages.iter().map(|(r, s)| (*r, s.as_str())).collect();
        tmpl.apply(&msg_refs)
    } else {
        // no template, just concatenate
        messages
            .iter()
            .map(|(_, s)| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    };

    // tokenize
    let mut tokens = Vec::new();
    if let Some(bos) = state.tokenizer.bos_token() {
        tokens.push(bos);
    }
    match state.tokenizer.encode(&prompt) {
        Ok(encoded) => tokens.extend(encoded),
        Err(e) => {
            token_tx
                .send(TokenEvent::Error(format!("tokenize failed: {e}")))
                .ok();
            return;
        }
    }

    let prompt_tokens = tokens.len() as u32;

    // prefill
    let prefill_start = Instant::now();
    let mut logits = match state.runner.prefill(&tokens) {
        Ok(l) => l,
        Err(e) => {
            token_tx
                .send(TokenEvent::Error(format!("prefill failed: {e}")))
                .ok();
            return;
        }
    };
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // decode
    let sampler = Sampler::new(SamplingParams {
        temperature,
        top_p,
        ..Default::default()
    });

    let eos = state.tokenizer.eos_token();
    let decode_start = Instant::now();
    let mut generated = 0u32;
    let mut finish_reason = "length".to_string();

    for _ in 0..max_tokens {
        let token = match sampler.sample(&logits) {
            Ok(t) => t,
            Err(e) => {
                token_tx
                    .send(TokenEvent::Error(format!("sample failed: {e}")))
                    .ok();
                return;
            }
        };

        if Some(token) == eos {
            finish_reason = "stop".to_string();
            break;
        }

        match state.tokenizer.decode(&[token]) {
            Ok(text) => {
                if token_tx.send(TokenEvent::Token(text)).is_err() {
                    return; // client disconnected
                }
            }
            Err(e) => {
                token_tx
                    .send(TokenEvent::Error(format!("decode failed: {e}")))
                    .ok();
                return;
            }
        }

        generated += 1;
        logits = match state.runner.decode_step(token) {
            Ok(l) => l,
            Err(e) => {
                token_tx
                    .send(TokenEvent::Error(format!("decode_step failed: {e}")))
                    .ok();
                return;
            }
        };
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    token_tx
        .send(TokenEvent::Done {
            prompt_tokens,
            completion_tokens: generated,
            finish_reason,
            prefill_ms,
            decode_ms,
        })
        .ok();
}
