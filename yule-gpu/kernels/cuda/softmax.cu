// Numerically stable softmax: out[i] = exp(input[i] - max) / sum(exp(input - max))
// Launched with a single workgroup of 256 threads.
extern "C" __global__ void softmax_kernel(
    const float* input,
    float* out,
    unsigned int size)
{
    __shared__ float shared_data[256];
    unsigned int tid = threadIdx.x;

    // Max reduction
    float local_max = -1.0e30f;
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    shared_data[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();

    // Exp + sum reduction
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        float e = expf(input[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    shared_data[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] += shared_data[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared_data[0];
    __syncthreads();

    // Normalize
    for (unsigned int i = tid; i < size; i += blockDim.x) {
        out[i] *= inv_sum;
    }
}
