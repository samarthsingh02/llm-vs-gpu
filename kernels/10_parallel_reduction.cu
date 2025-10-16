// kernels/10_parallel_reduction.cu
extern "C" __global__
void parallel_reduction(float* data, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = (idx < N) ? data[idx] + ((idx + blockDim.x < N) ? data[idx+blockDim.x] : 0) : 0;
    __syncthreads();

    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid==0)
        out[blockIdx.x] = sdata[0];
}
