// RMS normalization: out[i] = input[i] * inv_rms * weight[i]
// where inv_rms = rsqrt(mean(input^2) + eps)
// Launched with a single workgroup of 256 threads.
extern "C" __global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* out,
    unsigned int size,
    float eps)
{
    __shared__ float partial_sums[256];
    unsigned int tid = threadIdx.x;

    // Compute sum of squares
    float sum = 0.0f;
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float v = input[i];
        sum += v * v;
    }
    partial_sums[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(partial_sums[0] / (float)size + eps);
    __syncthreads();

    // Apply normalization
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        out[i] = input[i] * inv_rms * weight[i];
    }
}
