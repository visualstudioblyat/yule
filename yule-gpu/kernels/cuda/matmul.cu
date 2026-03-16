// Dense float32 matrix-vector multiply: out[row] = dot(A[row], B)
// A is (m x k) row-major, B is (k,) vector, out is (m,) vector.
// One workgroup per row, parallel reduction over k.
extern "C" __global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned int m,
    unsigned int n,
    unsigned int k)
{
    __shared__ float partial_sums[256];
    unsigned int row = blockIdx.x;
    if (row >= m) return;
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int i = tid; i < k; i += blockDim.x) {
        sum += a[row * k + i] * b[i];
    }
    partial_sums[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = partial_sums[0];
    }
}
