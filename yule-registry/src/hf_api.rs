use anyhow::{Context, Result};
use serde::Deserialize;

const HF_BASE: &str = "https://huggingface.co";
const USER_AGENT: &str = "yule/0.1.0";

pub struct HfApiClient {
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct HfFileEntry {
    pub path: String,
    pub size: u64,
    #[serde(rename = "type")]
    pub entry_type: String,
}

impl HfApiClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::USER_AGENT, USER_AGENT.parse().unwrap());
        if let Some(ref tok) = token {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {tok}")
                    .parse()
                    .context("invalid HF token")?,
            );
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self { client })
    }

    /// List files in a HuggingFace repo at the root level.
    pub async fn list_repo_files(&self, owner: &str, repo: &str) -> Result<Vec<HfFileEntry>> {
        let url = format!("{HF_BASE}/api/models/{owner}/{repo}/tree/main");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("failed to reach HuggingFace API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("HuggingFace API returned {status}: {body}");
        }

        let entries: Vec<HfFileEntry> = resp
            .json()
            .await
            .context("failed to parse HuggingFace file listing")?;

        Ok(entries)
    }

    /// Build the direct download URL for a file in a HuggingFace repo.
    pub fn download_url(owner: &str, repo: &str, filename: &str) -> String {
        format!("{HF_BASE}/{owner}/{repo}/resolve/main/{filename}")
    }

    /// Start a download, optionally resuming from a byte offset.
    /// Returns the response and the total file size (from Content-Range or Content-Length).
    pub async fn start_download(
        &self,
        url: &str,
        resume_from: Option<u64>,
    ) -> Result<(reqwest::Response, Option<u64>)> {
        let mut req = self.client.get(url);

        if let Some(offset) = resume_from {
            req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }

        let resp = req.send().await.context("download request failed")?;

        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            anyhow::bail!("download failed with status {}", resp.status());
        }

        // Parse total size from Content-Range (resume) or Content-Length (fresh)
        let total = if let Some(range) = resp.headers().get(reqwest::header::CONTENT_RANGE) {
            // Content-Range: bytes 12345-99999/100000
            range
                .to_str()
                .ok()
                .and_then(|s| s.rsplit('/').next())
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            resp.content_length()
        };

        Ok((resp, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_url_format() {
        let url = HfApiClient::download_url(
            "bartowski",
            "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        );
        assert_eq!(
            url,
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        );
    }
}
