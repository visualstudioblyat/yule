// Fused SiLU activation and element-wise multiply: out[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void silu_mul_kernel(
    const float* gate,
    const float* up,
    float* out,
    unsigned int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = gate[idx];
    float silu_x = x / (1.0f + expf(-x));
    out[idx] = silu_x * up[idx];
}
