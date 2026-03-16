// Element-wise addition: out[i] = a[i] + b[i]
extern "C" __global__ void add_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = a[idx] + b[idx];
}
