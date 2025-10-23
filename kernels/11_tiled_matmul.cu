// kernels/11_tiled_matmul.cu
#define TILE_WIDTH 32

extern "C" __global__
void tiled_matmul(const float* A, const float* B, float* C, int N) {
    // Shared memory tiles for A and B
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the global row and col of this thread's C element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // Loop over all tiles in the A and B matrices
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        // Load tile from global memory into shared memory (sA)
        int a_row = row;
        int a_col = m * TILE_WIDTH + tx;
        if (a_row < N && a_col < N) {
            sA[ty][tx] = A[a_row * N + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile from global memory into shared memory (sB)
        int b_row = m * TILE_WIDTH + ty;
        int b_col = col;
        if (b_row < N && b_col < N) {
            sB[ty][tx] = B[b_row * N + b_col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // Wait for all threads in the block to finish loading
        __syncthreads();

        // Compute partial dot product from shared memory
        // This is the fast, compute-heavy part
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        // Wait for all threads to finish computation before loading the next tile
        __syncthreads();
    }

    // Write the final result back to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}