// kernels/07_branch_divergence.cu
extern "C" __global__
void branch_divergence(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (threadIdx.x % 2 == 0)
            data[idx] += 1.0f;
        else
            data[idx] -= 1.0f;
    }
}
