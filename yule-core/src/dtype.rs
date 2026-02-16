use serde::{Deserialize, Serialize};

/// GGML quantization data types.
///
/// GGUF type IDs are defined by the ggml_type enum in ggml.h.
/// Block types use either QK=32 (basic) or QK_K=256 (K-quant super-blocks).
///
/// Note: Q3_K, Q4_K, Q5_K each have S/M/L variants at the model level,
/// but they all share the same block format and GGUF type ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum DType {
    // --- Floating point ---
    F32 = 0,
    F16 = 1,
    BF16 = 30,
    F64 = 28,

    // --- Basic blocks (QK = 32) ---
    Q4_0 = 2, // symmetric 4-bit, 18 B/block, 4.5 BPW
    Q4_1 = 3, // asymmetric 4-bit, 20 B/block, 5.0 BPW
    Q5_0 = 6, // symmetric 5-bit, 22 B/block, 5.5 BPW
    Q5_1 = 7, // asymmetric 5-bit, 24 B/block, 6.0 BPW
    Q8_0 = 8, // symmetric 8-bit, 34 B/block, 8.5 BPW
    Q8_1 = 9, // 8-bit with precomputed sum, 36 B/block, 9.0 BPW

    // --- K-quant super-blocks (QK_K = 256) ---
    Q2_K = 10, // 2-bit asymmetric, 84 B, 2.625 BPW
    Q3_K = 11, // 3-bit symmetric, 110 B, 3.4375 BPW
    Q4_K = 12, // 4-bit asymmetric, 144 B, 4.5 BPW
    Q5_K = 13, // 5-bit asymmetric, 176 B, 5.5 BPW
    Q6_K = 14, // 6-bit symmetric, 210 B, 6.5625 BPW
    Q8_K = 15, // 8-bit super-block, 292 B (f32 d + 256 qs + 16*f32 bsums)

    // --- I-Quant (importance-matrix quantization) ---
    IQ2_XXS = 16, // 2.06 BPW, E8 lattice
    IQ2_XS = 17,  // 2.31 BPW
    IQ3_XXS = 18, // 3.06 BPW
    IQ1_S = 19,   // 1.56 BPW
    IQ4_NL = 20,  // 4.5 BPW, 16-entry non-linear codebook
    IQ3_S = 21,   // 3.44 BPW
    IQ2_S = 22,   // 2.5 BPW
    IQ4_XS = 23,  // 4.25 BPW

    // --- Integer types ---
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,

    // --- IQ1_M ---
    IQ1_M = 29, // 1.75 BPW

    // --- Ternary (BitNet-style) ---
    TQ1_0 = 34, // 1.69 BPW, 5 trits per byte
    TQ2_0 = 35, // 2.06 BPW, 2-bit ternary {-1, 0, +1}
}

impl DType {
    /// Size in bytes of one quantization block.
    pub fn size_of_block(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 => 1,
            DType::I64 | DType::F64 => 8,
            // basic blocks (QK = 32)
            DType::Q4_0 => 18, // 2 (f16 d) + 16 (nibbles)
            DType::Q4_1 => 20, // 2 (f16 d) + 2 (f16 m) + 16 (nibbles)
            DType::Q5_0 => 22, // 2 (f16 d) + 4 (qh) + 16 (qs)
            DType::Q5_1 => 24, // 2 (f16 d) + 2 (f16 m) + 4 (qh) + 16 (qs)
            DType::Q8_0 => 34, // 2 (f16 d) + 32 (int8 qs)
            DType::Q8_1 => 36, // 2 (f16 d) + 2 (f16 s) + 32 (int8 qs)
            // k-quant super-blocks (QK_K = 256)
            DType::Q2_K => 84,  // 16 (scales) + 64 (qs) + 2 (d) + 2 (dmin)
            DType::Q3_K => 110, // 32 (hmask) + 64 (qs) + 12 (scales) + 2 (d)
            DType::Q4_K => 144, // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
            DType::Q5_K => 176, // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs)
            DType::Q6_K => 210, // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
            DType::Q8_K => 292, // 4 (f32 d) + 256 (int8 qs) + 16*2 (f16 bsums)
            // I-Quants
            DType::IQ1_S => 50,
            DType::IQ1_M => 56,
            DType::IQ2_XXS => 66,
            DType::IQ2_XS => 74,
            DType::IQ2_S => 82,
            DType::IQ3_XXS => 98,
            DType::IQ3_S => 110,
            DType::IQ4_XS => 136,
            DType::IQ4_NL => 18, // same as Q4_0: 2 (d) + 16 (qs), block of 32
            // Ternary
            DType::TQ1_0 => 54,
            DType::TQ2_0 => 66,
        }
    }

    /// Number of weights per block.
    pub fn block_size(&self) -> usize {
        match self {
            DType::F32
            | DType::F16
            | DType::BF16
            | DType::F64
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64 => 1,
            DType::Q8_0
            | DType::Q8_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Q4_0
            | DType::Q4_1
            | DType::IQ4_NL => 32,
            _ => 256, // all K-quants, I-quants, and ternary use QK_K = 256
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            DType::F32
                | DType::F16
                | DType::BF16
                | DType::F64
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
        )
    }

    /// Effective bits per weight (for memory estimation).
    pub fn bits_per_weight(&self) -> f32 {
        (self.size_of_block() as f32 * 8.0) / self.block_size() as f32
    }

    /// Convert from GGUF type ID to DType.
    pub fn from_gguf_type_id(id: u32) -> Option<DType> {
        match id {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            3 => Some(DType::Q4_1),
            6 => Some(DType::Q5_0),
            7 => Some(DType::Q5_1),
            8 => Some(DType::Q8_0),
            9 => Some(DType::Q8_1),
            10 => Some(DType::Q2_K),
            11 => Some(DType::Q3_K),
            12 => Some(DType::Q4_K),
            13 => Some(DType::Q5_K),
            14 => Some(DType::Q6_K),
            15 => Some(DType::Q8_K),
            16 => Some(DType::IQ2_XXS),
            17 => Some(DType::IQ2_XS),
            18 => Some(DType::IQ3_XXS),
            19 => Some(DType::IQ1_S),
            20 => Some(DType::IQ4_NL),
            21 => Some(DType::IQ3_S),
            22 => Some(DType::IQ2_S),
            23 => Some(DType::IQ4_XS),
            24 => Some(DType::I8),
            25 => Some(DType::I16),
            26 => Some(DType::I32),
            27 => Some(DType::I64),
            28 => Some(DType::F64),
            29 => Some(DType::IQ1_M),
            30 => Some(DType::BF16),
            34 => Some(DType::TQ1_0),
            35 => Some(DType::TQ2_0),
            _ => None,
        }
    }
}
