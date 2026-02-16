use crate::model::Architecture;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub struct ChatTemplate {
    bos: &'static str,
    system_pre: &'static str,
    system_suf: &'static str,
    user_pre: &'static str,
    user_suf: &'static str,
    assistant_pre: &'static str,
    assistant_suf: &'static str,
    generation_pre: &'static str,
}

impl ChatTemplate {
    pub fn for_architecture(arch: &Architecture) -> Option<Self> {
        match arch {
            Architecture::Llama => Some(Self {
                bos: "<|begin_of_text|>",
                system_pre: "<|start_header_id|>system<|end_header_id|>\n\n",
                system_suf: "<|eot_id|>",
                user_pre: "<|start_header_id|>user<|end_header_id|>\n\n",
                user_suf: "<|eot_id|>",
                assistant_pre: "<|start_header_id|>assistant<|end_header_id|>\n\n",
                assistant_suf: "<|eot_id|>",
                generation_pre: "<|start_header_id|>assistant<|end_header_id|>\n\n",
            }),
            Architecture::Mistral => Some(Self {
                bos: "<s>",
                system_pre: "[INST] ",
                system_suf: "\n",
                user_pre: "[INST] ",
                user_suf: " [/INST]",
                assistant_pre: "",
                assistant_suf: "</s>",
                generation_pre: "",
            }),
            Architecture::Phi => Some(Self {
                bos: "",
                system_pre: "<|system|>\n",
                system_suf: "<|end|>\n",
                user_pre: "<|user|>\n",
                user_suf: "<|end|>\n",
                assistant_pre: "<|assistant|>\n",
                assistant_suf: "<|end|>\n",
                generation_pre: "<|assistant|>\n",
            }),
            Architecture::Qwen => Some(Self {
                bos: "",
                system_pre: "<|im_start|>system\n",
                system_suf: "<|im_end|>\n",
                user_pre: "<|im_start|>user\n",
                user_suf: "<|im_end|>\n",
                assistant_pre: "<|im_start|>assistant\n",
                assistant_suf: "<|im_end|>\n",
                generation_pre: "<|im_start|>assistant\n",
            }),
            Architecture::Gemma => Some(Self {
                bos: "",
                system_pre: "<start_of_turn>user\n",
                system_suf: "<end_of_turn>\n",
                user_pre: "<start_of_turn>user\n",
                user_suf: "<end_of_turn>\n",
                assistant_pre: "<start_of_turn>model\n",
                assistant_suf: "<end_of_turn>\n",
                generation_pre: "<start_of_turn>model\n",
            }),
            _ => None,
        }
    }

    pub fn apply(&self, messages: &[(Role, &str)]) -> String {
        let mut result = String::from(self.bos);
        for (role, content) in messages {
            let (pre, suf) = match role {
                Role::System => (self.system_pre, self.system_suf),
                Role::User => (self.user_pre, self.user_suf),
                Role::Assistant => (self.assistant_pre, self.assistant_suf),
            };
            result.push_str(pre);
            result.push_str(content);
            result.push_str(suf);
        }
        result.push_str(self.generation_pre);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_template() {
        let tmpl = ChatTemplate::for_architecture(&Architecture::Llama).unwrap();
        let result = tmpl.apply(&[(Role::System, "You are helpful."), (Role::User, "Hello")]);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("Hello"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_qwen_template() {
        let tmpl = ChatTemplate::for_architecture(&Architecture::Qwen).unwrap();
        let result = tmpl.apply(&[(Role::User, "Hi")]);
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_gemma_template() {
        let tmpl = ChatTemplate::for_architecture(&Architecture::Gemma).unwrap();
        let result = tmpl.apply(&[(Role::User, "Test")]);
        assert!(result.contains("<start_of_turn>user\nTest<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }
}
