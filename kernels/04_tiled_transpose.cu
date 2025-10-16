// kernels/04_tiled_transpose.cu
#define TILE_DIM 32
#define BLOCK_ROWS 8

extern "C" __global__
void tiled_transpose(const float* in, float* out, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < N && y+i < N)
            tile[threadIdx.y+i][threadIdx.x] = in[(y+i)*N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < N && y+i < N)
            out[(y+i)*N + x] = tile[threadIdx.x][threadIdx.y+i];
    }
}
