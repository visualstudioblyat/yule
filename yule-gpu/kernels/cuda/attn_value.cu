// Attention value aggregation: out[out_offset+d] = sum_pos(weights[pos] * V[pos*kv_stride+kv_offset+d])
// One workgroup per output dimension, parallel reduction over sequence positions.
extern "C" __global__ void attn_value_kernel(
    const float* attn_weights,
    const float* value_cache,
    float* out,
    unsigned int head_dim,
    unsigned int seq_len,
    unsigned int kv_offset,
    unsigned int kv_stride,
    unsigned int out_offset)
{
    __shared__ float partial_sums[256];
    unsigned int dim_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (dim_idx >= head_dim) return;

    float sum = 0.0f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        sum += attn_weights[pos] * value_cache[pos * kv_stride + kv_offset + dim_idx];
    }
    partial_sums[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        out[out_offset + dim_idx] = partial_sums[0];
    }
}
