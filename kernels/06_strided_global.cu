// kernels/06_strided_global.cu
extern "C" __global__
void strided_global(const float* in, float* out, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidx = idx * stride;

    if (gidx < N)
        out[gidx] = in[gidx] * 2.0f;
}
