// Attention score: scores[pos] = dot(Q[head_offset..], K_cache[pos*kv_stride+kv_offset..]) / sqrt(head_dim)
// One workgroup per sequence position, parallel reduction over head_dim.
extern "C" __global__ void attn_score_kernel(
    const float* query,
    const float* key_cache,
    float* scores,
    unsigned int head_dim,
    unsigned int seq_len,
    unsigned int head_offset,
    unsigned int kv_offset,
    unsigned int kv_stride)
{
    __shared__ float partial_dots[256];
    unsigned int tid = threadIdx.x;
    unsigned int pos = blockIdx.x;
    if (pos >= seq_len) return;

    float dot = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        dot += query[head_offset + i] * key_cache[pos * kv_stride + kv_offset + i];
    }
    partial_dots[tid] = dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_dots[tid] += partial_dots[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        scores[pos] = partial_dots[0] / sqrtf((float)head_dim);
    }
}
