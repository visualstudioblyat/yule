// Q4_0 quantized matrix-vector multiply: 18 bytes/block = f16 scale + 16 bytes (32 nibbles).
// Symmetric 4-bit quantization, zero-point = 8.
// One workgroup per output row, parallel reduction over blocks.

__device__ float unpack_f16(unsigned short bits) {
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

extern "C" __global__ void qmv_q4_0_kernel(
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

    unsigned int row_offset = row * blocks_per_row * 18;
    float sum = 0.0f;

    for (unsigned int block = tid; block < blocks_per_row; block += blockDim.x) {
        unsigned int boff = row_offset + block * 18;
        unsigned short f16_bits = *(const unsigned short*)(weights + boff);
        float scale = unpack_f16(f16_bits);

        unsigned int col_base = block * 32;
        for (unsigned int j = 0; j < 16; j++) {
            unsigned char byte_val = weights[boff + 2 + j];
            float q0 = (float)((int)(byte_val & 0xF) - 8);
            float q1 = (float)((int)((byte_val >> 4) & 0xF) - 8);
            unsigned int c0 = col_base + j * 2;
            if (c0 < n_cols) sum += scale * q0 * input[c0];
            if (c0 + 1 < n_cols) sum += scale * q1 * input[c0 + 1];
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
