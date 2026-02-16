use ed25519_dalek::{SigningKey, VerifyingKey};
use std::path::{Path, PathBuf};
use yule_core::error::{Result, YuleError};

/// Manages device signing keys and publisher trust store.
pub struct KeyStore {
    base_dir: PathBuf,
}

impl KeyStore {
    /// Initialize key store at ~/.yule/keys/
    pub fn open() -> Result<Self> {
        let base = dirs_path()?;
        std::fs::create_dir_all(&base).map_err(YuleError::Io)?;
        Ok(Self { base_dir: base })
    }

    /// Open key store at a specific directory (for testing).
    pub fn open_at(path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&path).map_err(YuleError::Io)?;
        Ok(Self { base_dir: path })
    }

    /// Load or generate the device signing key.
    /// Generated on first run, persisted for all future sessions.
    pub fn device_key(&self) -> Result<SigningKey> {
        let key_path = self.base_dir.join("device.key");
        if key_path.exists() {
            self.load_signing_key(&key_path)
        } else {
            let key = self.generate_and_save(&key_path)?;
            tracing::info!("generated new device signing key");
            Ok(key)
        }
    }

    /// Get the device verifying (public) key.
    pub fn device_public_key(&self) -> Result<VerifyingKey> {
        Ok(self.device_key()?.verifying_key())
    }

    /// Trust a publisher's public key (saves to keys/{name}.pub).
    pub fn trust_publisher(&self, name: &str, public_key: &[u8; 32]) -> Result<()> {
        let path = self.base_dir.join(format!("{name}.pub"));
        std::fs::write(&path, public_key).map_err(YuleError::Io)?;
        tracing::info!(publisher = name, "trusted publisher key");
        Ok(())
    }

    /// Load a trusted publisher's public key.
    pub fn publisher_key(&self, name: &str) -> Result<Option<VerifyingKey>> {
        let path = self.base_dir.join(format!("{name}.pub"));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).map_err(YuleError::Io)?;
        if bytes.len() != 32 {
            return Err(YuleError::Verification(format!(
                "publisher key '{name}' is {} bytes, expected 32",
                bytes.len()
            )));
        }
        let key_bytes: [u8; 32] = bytes.try_into().unwrap();
        let key = VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| YuleError::Verification(format!("invalid publisher key: {e}")))?;
        Ok(Some(key))
    }

    /// List all trusted publishers.
    pub fn list_publishers(&self) -> Result<Vec<String>> {
        let mut publishers = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir).map_err(YuleError::Io)?;
        for entry in entries {
            let entry = entry.map_err(YuleError::Io)?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pub") && name != "device.pub" {
                publishers.push(name.trim_end_matches(".pub").to_string());
            }
        }
        Ok(publishers)
    }

    fn generate_and_save(&self, path: &Path) -> Result<SigningKey> {
        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret)
            .map_err(|e| YuleError::Verification(format!("CSPRNG failed: {e}")))?;
        let key = SigningKey::from_bytes(&secret);
        // save raw 32-byte secret key
        std::fs::write(path, key.to_bytes()).map_err(YuleError::Io)?;
        // also save public key alongside for reference
        let pub_path = path.with_extension("pub");
        std::fs::write(&pub_path, key.verifying_key().to_bytes()).map_err(YuleError::Io)?;
        Ok(key)
    }

    fn load_signing_key(&self, path: &Path) -> Result<SigningKey> {
        let bytes = std::fs::read(path).map_err(YuleError::Io)?;
        if bytes.len() != 32 {
            return Err(YuleError::Verification(format!(
                "device key is {} bytes, expected 32",
                bytes.len()
            )));
        }
        let key_bytes: [u8; 32] = bytes.try_into().unwrap();
        Ok(SigningKey::from_bytes(&key_bytes))
    }
}

/// Default key store directory: ~/.yule/keys/
fn dirs_path() -> Result<PathBuf> {
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .map_err(|_| YuleError::Verification("cannot determine home directory".into()))?
    } else {
        std::env::var("HOME")
            .map_err(|_| YuleError::Verification("cannot determine home directory".into()))?
    };
    Ok(PathBuf::from(home).join(".yule").join("keys"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_and_load_device_key() {
        let dir = std::env::temp_dir().join("yule-test-keys-1");
        let _ = std::fs::remove_dir_all(&dir);
        let store = KeyStore::open_at(dir.clone()).unwrap();

        // first call generates
        let key1 = store.device_key().unwrap();
        // second call loads the same key
        let key2 = store.device_key().unwrap();
        assert_eq!(key1.to_bytes(), key2.to_bytes());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trust_and_load_publisher() {
        let dir = std::env::temp_dir().join("yule-test-keys-2");
        let _ = std::fs::remove_dir_all(&dir);
        let store = KeyStore::open_at(dir.clone()).unwrap();

        let mut secret = [0u8; 32];
        getrandom::fill(&mut secret).unwrap();
        let key = SigningKey::from_bytes(&secret);
        let pubkey = key.verifying_key().to_bytes();

        store.trust_publisher("test-pub", &pubkey).unwrap();

        let loaded = store.publisher_key("test-pub").unwrap().unwrap();
        assert_eq!(loaded.to_bytes(), pubkey);

        let missing = store.publisher_key("nonexistent").unwrap();
        assert!(missing.is_none());

        let publishers = store.list_publishers().unwrap();
        assert!(publishers.contains(&"test-pub".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
