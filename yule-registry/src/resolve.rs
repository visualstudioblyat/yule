use anyhow::Result;

pub struct ModelResolver;

impl ModelResolver {
    pub fn parse_model_ref(model_ref: &str) -> Result<ParsedModelRef> {
        // formats:
        //   "bartowski/Llama-3.2-1B-Instruct-GGUF" -> publisher + model name
        //   "bartowski/Llama-3.2-1B-Instruct-GGUF:q4_k_m" -> publisher + model + quant
        //   "/path/to/model.gguf" -> local file
        //   "C:\path\to\model.gguf" -> local file (Windows)

        // Detect local file paths
        if model_ref.starts_with('/')
            || model_ref.starts_with('.')
            || model_ref.contains('\\')
            || (model_ref.len() >= 2 && model_ref.as_bytes()[1] == b':')  // Windows drive letter
            || model_ref.ends_with(".gguf")
        {
            return Ok(ParsedModelRef::LocalFile(model_ref.into()));
        }

        let parts: Vec<&str> = model_ref.splitn(2, '/').collect();

        if parts.len() == 2 {
            let publisher = parts[0].to_string();
            let (name, quant) = if let Some((n, q)) = parts[1].split_once(':') {
                (n.to_string(), Some(q.to_string()))
            } else {
                (parts[1].to_string(), None)
            };

            return Ok(ParsedModelRef::Remote {
                publisher,
                name,
                quantization: quant,
            });
        }

        Ok(ParsedModelRef::Remote {
            publisher: String::new(),
            name: model_ref.to_string(),
            quantization: None,
        })
    }
}

#[derive(Debug)]
pub enum ParsedModelRef {
    LocalFile(String),
    Remote {
        publisher: String,
        name: String,
        quantization: Option<String>,
    },
}
