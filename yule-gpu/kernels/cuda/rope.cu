// Rotary Position Embedding: applies rotation to Q and K heads in-place.
// Each thread handles one (cos,sin) pair for one head.
extern "C" __global__ void rope_kernel(
    float* q,
    float* k,
    unsigned int pos,
    unsigned int head_dim,
    float freq_base,
    unsigned int n_heads_q,
    unsigned int n_heads_k)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_dim = head_dim / 2;
    unsigned int total_q = n_heads_q * half_dim;
    unsigned int total_k = n_heads_k * half_dim;

    if (idx >= total_q + total_k) return;

    if (idx < total_q) {
        unsigned int pair_idx = idx % half_dim;
        unsigned int head = idx / half_dim;
        unsigned int base_off = head * head_dim;

        float freq = 1.0f / powf(freq_base, (float)(2 * pair_idx) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = q[base_off + 2 * pair_idx];
        float q1 = q[base_off + 2 * pair_idx + 1];
        q[base_off + 2 * pair_idx]     = q0 * cos_a - q1 * sin_a;
        q[base_off + 2 * pair_idx + 1] = q0 * sin_a + q1 * cos_a;
    } else {
        unsigned int k_idx = idx - total_q;
        unsigned int pair_idx = k_idx % half_dim;
        unsigned int head = k_idx / half_dim;
        unsigned int base_off = head * head_dim;

        float freq = 1.0f / powf(freq_base, (float)(2 * pair_idx) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float k0 = k[base_off + 2 * pair_idx];
        float k1 = k[base_off + 2 * pair_idx + 1];
        k[base_off + 2 * pair_idx]     = k0 * cos_a - k1 * sin_a;
        k[base_off + 2 * pair_idx + 1] = k0 * sin_a + k1 * cos_a;
    }
}
