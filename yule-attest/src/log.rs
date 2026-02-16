use crate::AttestationRecord;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use yule_core::error::{Result, YuleError};

/// Append-only, hash-chained audit log stored as JSON-lines.
/// Each record includes blake3(previous_record) for tamper detection.
pub struct AuditLog {
    path: PathBuf,
}

impl AuditLog {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Default log location: ~/.yule/audit.jsonl
    pub fn default_path() -> Result<Self> {
        let home = if cfg!(windows) {
            std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .map_err(|_| YuleError::Verification("cannot determine home directory".into()))?
        } else {
            std::env::var("HOME")
                .map_err(|_| YuleError::Verification("cannot determine home directory".into()))?
        };
        let path = PathBuf::from(home).join(".yule").join("audit.jsonl");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(YuleError::Io)?;
        }
        Ok(Self { path })
    }

    /// Get the hash of the last record in the log (for chaining).
    /// Returns [0u8; 32] if the log is empty or doesn't exist.
    pub fn last_hash(&self) -> Result<[u8; 32]> {
        if !self.path.exists() {
            return Ok([0u8; 32]);
        }
        let records = self.read_all()?;
        match records.last() {
            Some(r) => Ok(r.hash()),
            None => Ok([0u8; 32]),
        }
    }

    /// Append a signed record to the log.
    pub fn append(&self, record: &AttestationRecord) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(YuleError::Io)?;

        let line = serde_json::to_string(record)
            .map_err(|e| YuleError::Verification(format!("serialize record: {e}")))?;
        writeln!(file, "{}", line).map_err(YuleError::Io)?;
        Ok(())
    }

    /// Verify the entire hash chain. Returns (valid, total_records).
    pub fn verify_chain(&self) -> Result<(bool, usize)> {
        let records = self.read_all()?;
        if records.is_empty() {
            return Ok((true, 0));
        }

        // first record should chain from [0u8; 32]
        if records[0].prev_hash != [0u8; 32] {
            return Ok((false, records.len()));
        }

        // each subsequent record's prev_hash must equal hash of previous record
        for i in 1..records.len() {
            let expected = records[i - 1].hash();
            if records[i].prev_hash != expected {
                tracing::warn!(record = i, "hash chain broken at record {i}");
                return Ok((false, records.len()));
            }
        }

        Ok((true, records.len()))
    }

    /// Query the last N records.
    pub fn query_last(&self, count: usize) -> Result<Vec<AttestationRecord>> {
        let records = self.read_all()?;
        let start = records.len().saturating_sub(count);
        Ok(records[start..].to_vec())
    }

    /// Read all records from the log file.
    fn read_all(&self) -> Result<Vec<AttestationRecord>> {
        if !self.path.exists() {
            return Ok(vec![]);
        }
        let file = std::fs::File::open(&self.path).map_err(YuleError::Io)?;
        let reader = std::io::BufReader::new(file);
        let mut records = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(YuleError::Io)?;
            if line.trim().is_empty() {
                continue;
            }
            let record: AttestationRecord = serde_json::from_str(&line).map_err(|e| {
                YuleError::Verification(format!("invalid record at line {}: {e}", i + 1))
            })?;
            records.push(record);
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceAttestation;
    use crate::session::AttestationSession;
    use ed25519_dalek::SigningKey;

    fn make_record(key: &SigningKey, prev_hash: [u8; 32]) -> AttestationRecord {
        let mut session = AttestationSession::new();
        session.set_model("test".into(), [1u8; 32], None, false);
        session.set_sandbox(true, 1024);
        let inference = InferenceAttestation {
            tokens_generated: 10,
            prompt_hash: [2u8; 32],
            output_hash: [3u8; 32],
            temperature: 0.7,
            top_p: 0.9,
        };
        session.finalize(inference, key, prev_hash).unwrap()
    }

    #[test]
    fn append_and_query() {
        let path = std::env::temp_dir().join("yule-test-audit-1.jsonl");
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::new(path.clone());

        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);

        let r1 = make_record(&key, [0u8; 32]);
        log.append(&r1).unwrap();

        let r2 = make_record(&key, r1.hash());
        log.append(&r2).unwrap();

        let records = log.query_last(10).unwrap();
        assert_eq!(records.len(), 2);

        let last = log.query_last(1).unwrap();
        assert_eq!(last.len(), 1);
        assert_eq!(last[0].session_id, r2.session_id);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn verify_valid_chain() {
        let path = std::env::temp_dir().join("yule-test-audit-2.jsonl");
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::new(path.clone());

        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);

        let r1 = make_record(&key, [0u8; 32]);
        log.append(&r1).unwrap();

        let r2 = make_record(&key, r1.hash());
        log.append(&r2).unwrap();

        let r3 = make_record(&key, r2.hash());
        log.append(&r3).unwrap();

        let (valid, count) = log.verify_chain().unwrap();
        assert!(valid);
        assert_eq!(count, 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn detect_tampered_chain() {
        let path = std::env::temp_dir().join("yule-test-audit-3.jsonl");
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::new(path.clone());

        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);

        let r1 = make_record(&key, [0u8; 32]);
        log.append(&r1).unwrap();

        // intentionally break the chain — wrong prev_hash
        let r2 = make_record(&key, [99u8; 32]);
        log.append(&r2).unwrap();

        let (valid, count) = log.verify_chain().unwrap();
        assert!(!valid);
        assert_eq!(count, 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_log_is_valid() {
        let path = std::env::temp_dir().join("yule-test-audit-4.jsonl");
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::new(path.clone());

        let (valid, count) = log.verify_chain().unwrap();
        assert!(valid);
        assert_eq!(count, 0);
    }
}
