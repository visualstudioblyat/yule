use crate::dequant;
use crate::dtype::DType;

pub fn vec_dot_dispatch(dtype: DType, block: &[u8], act: &[f32]) -> Option<f32> {
    match dtype {
        DType::Q4_0 => Some(dequant::vec_dot_q4_0(block, act)),
        DType::Q8_0 => Some(dequant::vec_dot_q8_0(block, act)),
        DType::Q2_K => Some(dequant::vec_dot_q2_k(block, act)),
        DType::Q3_K => Some(dequant::vec_dot_q3_k(block, act)),
        DType::Q4_K => Some(dequant::vec_dot_q4_k(block, act)),
        DType::Q5_K => Some(dequant::vec_dot_q5_k(block, act)),
        DType::Q6_K => Some(dequant::vec_dot_q6_k(block, act)),
        DType::IQ4_NL => Some(dequant::vec_dot_iq4_nl(block, act)),
        _ => None,
    }
}
