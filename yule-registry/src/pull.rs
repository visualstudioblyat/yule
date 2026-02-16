use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

use crate::hf_api::HfApiClient;

pub struct ModelPuller {
    client: HfApiClient,
}

impl ModelPuller {
    pub fn new(client: HfApiClient) -> Self {
        Self { client }
    }

    /// Download a file from `url` to `dest`, with resume support and progress reporting.
    /// Returns the final path (`.part` suffix stripped).
    pub async fn download(&self, url: &str, dest: &Path) -> Result<PathBuf> {
        let part_path = dest.with_extension(format!(
            "{}.part",
            dest.extension().unwrap_or_default().to_str().unwrap_or("")
        ));

        // Check for existing partial download
        let resume_from = if part_path.exists() {
            let meta = tokio::fs::metadata(&part_path)
                .await
                .context("failed to stat .part file")?;
            let size = meta.len();
            if size > 0 {
                eprintln!("resuming download from {}", format_bytes(size));
                Some(size)
            } else {
                None
            }
        } else {
            None
        };

        // Ensure parent directory exists
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("failed to create download directory")?;
        }

        let (resp, total_size) = self
            .client
            .start_download(url, resume_from)
            .await
            .context("failed to start download")?;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .await
            .context("failed to open .part file")?;

        let mut downloaded = resume_from.unwrap_or(0);
        let mut last_report = std::time::Instant::now();
        let mut last_bytes = downloaded;
        let start = std::time::Instant::now();

        let mut stream = resp.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("error reading download stream")?;
            file.write_all(&chunk)
                .await
                .context("failed to write chunk")?;
            downloaded += chunk.len() as u64;

            // Progress report every 500ms
            let now = std::time::Instant::now();
            if now.duration_since(last_report).as_millis() >= 500 {
                let speed = (downloaded - last_bytes) as f64
                    / now.duration_since(last_report).as_secs_f64();
                last_bytes = downloaded;
                last_report = now;

                if let Some(total) = total_size {
                    let pct = (downloaded as f64 / total as f64 * 100.0).min(100.0);
                    let eta = if speed > 0.0 {
                        let remaining = total.saturating_sub(downloaded) as f64 / speed;
                        format_duration(remaining)
                    } else {
                        "??".to_string()
                    };
                    eprint!(
                        "\r  {} / {} ({:.1}%) [{}/s] ETA {}    ",
                        format_bytes(downloaded),
                        format_bytes(total),
                        pct,
                        format_bytes(speed as u64),
                        eta,
                    );
                } else {
                    eprint!(
                        "\r  {} [{}/s]    ",
                        format_bytes(downloaded),
                        format_bytes(speed as u64),
                    );
                }
            }
        }

        file.flush().await.context("failed to flush file")?;
        drop(file);

        let elapsed = start.elapsed();
        let avg_speed = (downloaded - resume_from.unwrap_or(0)) as f64 / elapsed.as_secs_f64();
        eprintln!(
            "\r  {} downloaded in {} ({}/s)          ",
            format_bytes(downloaded),
            format_duration(elapsed.as_secs_f64()),
            format_bytes(avg_speed as u64),
        );

        // Rename .part → final
        tokio::fs::rename(&part_path, dest)
            .await
            .context("failed to rename .part file")?;

        Ok(dest.to_path_buf())
    }
}

fn format_bytes(n: u64) -> String {
    if n >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", n as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if n >= 1024 * 1024 {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn format_duration(secs: f64) -> String {
    if secs >= 3600.0 {
        format!(
            "{}h{}m",
            (secs / 3600.0) as u64,
            ((secs % 3600.0) / 60.0) as u64
        )
    } else if secs >= 60.0 {
        format!("{}m{}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{:.0}s", secs)
    }
}
