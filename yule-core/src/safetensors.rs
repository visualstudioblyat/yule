use crate::error::{Result, YuleError};
use crate::model::LoadedModel;
use std::path::Path;

const _MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024; // 100MB header cap

pub struct SafetensorsParser;

impl SafetensorsParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, _path: &Path) -> Result<LoadedModel> {
        // TODO: implement safetensors parsing
        // 1. read 8-byte LE header length
        // 2. validate header length <= MAX_HEADER_SIZE
        // 3. parse JSON header
        // 4. validate tensor offsets don't overlap and are within bounds
        // 5. build LoadedModel
        todo!("safetensors parsing — next milestone")
    }

    fn _validate_header_size(size: u64) -> Result<()> {
        if size > _MAX_HEADER_SIZE {
            return Err(YuleError::AllocationTooLarge {
                requested: size,
                max: _MAX_HEADER_SIZE,
            });
        }
        Ok(())
    }
}

impl Default for SafetensorsParser {
    fn default() -> Self {
        Self::new()
    }
}
