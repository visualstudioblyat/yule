// Embedding table lookup: out[i] = embed_table[token_ids[token_idx] * embed_dim + i]
extern "C" __global__ void embed_lookup_kernel(
    const float* embed_table,
    const unsigned int* token_ids,
    float* out,
    unsigned int embed_dim,
    unsigned int token_idx)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= embed_dim) return;
    unsigned int tok = token_ids[token_idx];
    out[idx] = embed_table[tok * embed_dim + idx];
}
