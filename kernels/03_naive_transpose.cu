// kernels/03_naive_transpose.cu
extern "C" __global__
void naive_transpose(const float* in, float* out, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        out[row + col*N] = in[row*N + col]; // note: column-major write → uncoalesced
    }
}
