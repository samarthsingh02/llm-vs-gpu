// kernels/05_bank_conflict.cu
#define SIZE 1024

extern "C" __global__
void bank_conflict(int *data) {
    __shared__ int tile[SIZE];
    int idx = threadIdx.x;

    // Access shared memory with stride = 1 (bank conflict)
    tile[idx] = idx;
    __syncthreads();

    tile[(idx*2) % SIZE] += 1; // induce conflict
    __syncthreads();
}
