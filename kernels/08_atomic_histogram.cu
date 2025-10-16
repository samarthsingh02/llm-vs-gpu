// kernels/08_atomic_histogram.cu
extern "C" __global__
void atomic_histogram(int* hist, const int* data, int N, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = data[idx] % bins;
        atomicAdd(&hist[bin], 1); // contention on small bins
    }
}
