use crate::dtype::DType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size_bytes: u64,
}

impl TensorInfo {
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn validate(&self, file_size: u64) -> crate::error::Result<()> {
        if self.shape.is_empty() {
            return Err(crate::error::YuleError::InvalidTensorShape {
                name: self.name.clone(),
                reason: "empty shape".into(),
            });
        }

        // overflow check on element count
        let mut elements: u64 = 1;
        for &dim in &self.shape {
            elements = elements.checked_mul(dim).ok_or_else(|| {
                crate::error::YuleError::InvalidTensorShape {
                    name: self.name.clone(),
                    reason: format!("shape overflow: dimension {dim} causes overflow"),
                }
            })?;
        }

        // bounds check
        let end = self.offset.checked_add(self.size_bytes).ok_or_else(|| {
            crate::error::YuleError::TensorOutOfBounds {
                name: self.name.clone(),
                offset: self.offset,
                file_size,
            }
        })?;

        if end > file_size {
            return Err(crate::error::YuleError::TensorOutOfBounds {
                name: self.name.clone(),
                offset: self.offset,
                file_size,
            });
        }

        Ok(())
    }
}

pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(info: &'a TensorInfo, data: &'a [u8]) -> Self {
        Self { info, data }
    }
}
