#include <cstdio>
#include <cuda_runtime.h>

// Declare kernel functions (extern "C" to match kernels)
extern "C" __global__ void saxpy(float *x, float *y, float a, int N);
extern "C" __global__ void naive_matmul(float *A, float *B, float *C, int N);
extern "C" __global__ void naive_transpose(float *A, float *B, int N);
extern "C" __global__ void tiled_transpose(float *A, float *B, int N);
extern "C" __global__ void bank_conflict(int *data, int N);
extern "C" __global__ void strided_global(float *data, int N);
extern "C" __global__ void branch_divergence(float *data, int N);
extern "C" __global__ void atomic_histogram(int *data, int *hist, int N);
extern "C" __global__ void high_reg_pressure(float *data, int N);
extern "C" __global__ void parallel_reduction(float *data, float *result, int N);

int main() {
    int N = 1024; // example size
    float *dA, *dB, *dC;
    int *dInt, *dHist;

    // TODO: allocate memory, init, and call kernels
    // Example for saxpy:
    // saxpy<<<N/256, 256>>>(dA, dB, 2.0f, N);
    // cudaDeviceSynchronize();

    printf("Harness compiled successfully! Add kernel calls here.\n");
    return 0;
}
