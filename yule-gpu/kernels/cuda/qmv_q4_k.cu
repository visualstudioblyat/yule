// Q4_K quantized matrix-vector multiply: 144 bytes/super-block = 256 weights.
// Layout: f16 d (2B) + f16 dmin (2B) + scales[12] (12B) + quants[128] (128B)
// 4-bit asymmetric K-quant with 6-bit sub-block scales/mins.
// One workgroup per output row, parallel reduction over super-blocks.

__device__ float unpack_f16_q4k(unsigned short bits) {
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

extern "C" __global__ void qmv_q4_k_kernel(
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

    unsigned int row_byte_offset = row * blocks_per_row * 144;
    float sum = 0.0f;

    for (unsigned int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        unsigned int boff = row_byte_offset + blk * 144;

        float d    = unpack_f16_q4k(*(const unsigned short*)(weights + boff));
        float dmin = unpack_f16_q4k(*(const unsigned short*)(weights + boff + 2));

        // Read 12 raw scale bytes
        unsigned char q[12];
        for (unsigned int i = 0; i < 12; i++) {
            q[i] = weights[boff + 4 + i];
        }

        // Extract 6-bit scales and mins (get_scale_min_k4 from ggml)
        unsigned int sc8[8], mn8[8];
        for (unsigned int j = 0; j < 4; j++) {
            sc8[j] = q[j] & 63u;
            mn8[j] = q[j + 4] & 63u;
        }
        for (unsigned int j = 4; j < 8; j++) {
            sc8[j] = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6u) << 4u);
            mn8[j] = (q[j + 4] >> 4u)   | ((q[j] >> 6u) << 4u);
        }

        unsigned int quant_off = boff + 16;
        unsigned int col_base = blk * 256;

        // 4 groups of 64 weights, each group = 32 bytes of quants
        for (unsigned int grp = 0; grp < 4; grp++) {
            unsigned int qs_off = quant_off + grp * 32;

            // Sub-block 2*grp: low nibbles (first 32 weights)
            unsigned int sb0 = 2 * grp;
            float scale0 = d * (float)sc8[sb0];
            float min0   = dmin * (float)mn8[sb0];
            for (unsigned int l = 0; l < 32; l++) {
                unsigned int c = col_base + 64 * grp + l;
                if (c < n_cols) {
                    float qval = (float)(weights[qs_off + l] & 0xFu);
                    sum += (qval * scale0 - min0) * input[c];
                }
            }

            // Sub-block 2*grp+1: high nibbles (next 32 weights)
            unsigned int sb1 = 2 * grp + 1;
            float scale1 = d * (float)sc8[sb1];
            float min1   = dmin * (float)mn8[sb1];
            for (unsigned int l = 0; l < 32; l++) {
                unsigned int c = col_base + 64 * grp + 32 + l;
                if (c < n_cols) {
                    float qval = (float)((weights[qs_off + l] >> 4u) & 0xFu);
                    sum += (qval * scale1 - min1) * input[c];
                }
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
