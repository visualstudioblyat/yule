// Q6_K quantized matrix-vector multiply: 210 bytes/super-block = 256 weights.
// Layout: [ql: 128B] [qh: 64B] [scales: 16B int8] [d: f16 (2B)]
// 6-bit symmetric quantization with bias of 32.
// One workgroup per output row, parallel reduction over super-blocks.

__device__ float unpack_f16_q6k(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 1;
    unsigned int exp_val = (bits >> 10) & 0x1F;
    unsigned int mant = bits & 0x3FF;
    if (exp_val == 0) {
        float val = ldexpf((float)mant, -24);
        return sign ? -val : val;
    }
    if (exp_val == 31) return sign ? -INFINITY : INFINITY;
    float val = ldexpf((float)(mant | 0x400), (int)exp_val - 25);
    return sign ? -val : val;
}

extern "C" __global__ void qmv_q6_k_kernel(
    const unsigned char* weights,
    const float* input,
    float* out,
    unsigned int n_rows,
    unsigned int n_cols,
    unsigned int blocks_per_row)
{
    __shared__ float partial_sums[256];
    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;
    unsigned int tid = threadIdx.x;

    unsigned int row_byte_offset = row * blocks_per_row * 210;
    float sum = 0.0f;

    for (unsigned int block = tid; block < blocks_per_row; block += blockDim.x) {
        unsigned int boff = row_byte_offset + block * 210;

        // d is at offset 208 within the block
        float d = unpack_f16_q6k(*(const unsigned short*)(weights + boff + 208));

        unsigned int col_base = block * 256;

        // Process two halves of 128 weights each
        for (unsigned int h = 0; h < 2; h++) {
            unsigned int ql_off = boff + h * 64;           // 64 bytes of ql per half
            unsigned int qh_off = boff + 128 + h * 32;     // 32 bytes of qh per half
            unsigned int sc_off = boff + 192;               // scales base
            unsigned int sc_base = h * 8;
            unsigned int out_base = col_base + h * 128;

            for (unsigned int l = 0; l < 32; l++) {
                unsigned int is_idx = l / 16;  // sub-block index within half

                unsigned char ql_lo = weights[ql_off + l];
                unsigned char ql_hi = weights[ql_off + l + 32];
                unsigned char qh_val = weights[qh_off + l];

                // 4 groups of weights from this position
                int q1 = (int)((ql_lo & 0xFu) | (((qh_val >> 0) & 0x3u) << 4)) - 32;
                int q2 = (int)((ql_hi & 0xFu) | (((qh_val >> 2) & 0x3u) << 4)) - 32;
                int q3 = (int)((ql_lo >> 4u)   | (((qh_val >> 4) & 0x3u) << 4)) - 32;
                int q4 = (int)((ql_hi >> 4u)   | (((qh_val >> 6) & 0x3u) << 4)) - 32;

                // Read int8 scales (signed)
                float s1 = (float)(*(const signed char*)(weights + sc_off + sc_base + is_idx + 0));
                float s2 = (float)(*(const signed char*)(weights + sc_off + sc_base + is_idx + 2));
                float s3 = (float)(*(const signed char*)(weights + sc_off + sc_base + is_idx + 4));
                float s4 = (float)(*(const signed char*)(weights + sc_off + sc_base + is_idx + 6));

                unsigned int c1 = out_base + l;
                unsigned int c2 = out_base + l + 32;
                unsigned int c3 = out_base + l + 64;
                unsigned int c4 = out_base + l + 96;

                if (c1 < n_cols) sum += d * s1 * (float)q1 * input[c1];
                if (c2 < n_cols) sum += d * s2 * (float)q2 * input[c2];
                if (c3 < n_cols) sum += d * s3 * (float)q3 * input[c3];
                if (c4 < n_cols) sum += d * s4 * (float)q4 * input[c4];
            }
        }
    }

    partial_sums[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = partial_sums[0];
}
