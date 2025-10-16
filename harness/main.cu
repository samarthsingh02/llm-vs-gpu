#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <functional> // Required for std::function
#include <cuda_runtime.h>

// Declare kernel functions
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n);
extern "C" __global__ void naive_matmul(const float* A, const float* B, float* C, int N);
extern "C" __global__ void naive_transpose(const float* in, float* out, int N);
extern "C" __global__ void tiled_transpose(const float* in, float* out, int N);
extern "C" __global__ void bank_conflict(int *data);
extern "C" __global__ void strided_global(const float* in, float* out, int N, int stride);
extern "C" __global__ void branch_divergence(float* data, int N);
extern "C" __global__ void atomic_histogram(int* hist, const int* data, int N, int bins);
extern "C" __global__ void high_reg_pressure(float* data, int N);
extern "C" __global__ void parallel_reduction(float *data, float *result, int N);

// Helper to check for CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
void time_kernel(const std::string& kernel_name, std::function<void()> kernel_launch, int iterations = 100) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel_launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Benchmarking: " << kernel_name << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        kernel_launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // This printf helps Nsight Compute to distinguish kernel calls
    printf("Finished benchmarking %s. Avg time: %f ms\n", kernel_name.c_str(), milliseconds / iterations);
}

int main() {
    int N = 4096; // Matrix/vector dimension
    int data_size_float = N * N * sizeof(float);
    int data_size_int = N * N * sizeof(int);

    // Allocate unified memory for simplicity
    float *dA, *dB, *dC;
    int *dIntA, *dIntB;
    CUDA_CHECK(cudaMallocManaged(&dA, data_size_float));
    CUDA_CHECK(cudaMallocManaged(&dB, data_size_float));
    CUDA_CHECK(cudaMallocManaged(&dC, data_size_float));
    CUDA_CHECK(cudaMallocManaged(&dIntA, data_size_int));
    CUDA_CHECK(cudaMallocManaged(&dIntB, data_size_int));

    printf("Running kernels...\n\n");

    // 1. SAXPY
    time_kernel("saxpy", [&]() {
        saxpy<<< (N*N + 255) / 256, 256 >>>(2.0f, dA, dB, N * N);
    });

    // 2. Naive Matrix Multiply
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks( (N + 15) / 16, (N + 15) / 16);
    time_kernel("naive_matmul", [&]() {
        naive_matmul<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    });

    // 3. Naive Transpose
    time_kernel("naive_transpose", [&]() {
        naive_transpose<<<numBlocks, threadsPerBlock>>>(dA, dB, N);
    });

    // 4. Tiled Transpose
    time_kernel("tiled_transpose", [&]() {
        tiled_transpose<<<numBlocks, threadsPerBlock>>>(dA, dB, N);
    });

    // 5. Shared Memory Bank Conflict
    time_kernel("bank_conflict", [&]() {
       bank_conflict<<<1, 1024>>>(dIntA);
    });

    // 6. Strided Global Access
    time_kernel("strided_global", [&]() {
        strided_global<<<(N*N + 255) / 256, 256>>>(dA, dB, N*N, 16);
    });

    // 7. Branch Divergence
    time_kernel("branch_divergence", [&]() {
        branch_divergence<<<(N*N + 255)/256, 256>>>(dA, N*N);
    });

    // 8. Atomic Contention (Histogram)
    int bins = 64;
    CUDA_CHECK(cudaMemset(dIntB, 0, bins * sizeof(int)));
    time_kernel("atomic_histogram", [&]() {
        atomic_histogram<<<(N*N + 255)/256, 256>>>(dIntB, dIntA, N*N, bins);
    });

    // 9. High Register Pressure
    time_kernel("high_reg_pressure", [&]() {
        high_reg_pressure<<<(N*N + 255)/256, 256>>>(dA, N*N);
    });

    // 10. Parallel Reduction
    int reduction_blocks = 512;
    float* d_reduction_out;
    CUDA_CHECK(cudaMallocManaged(&d_reduction_out, reduction_blocks * sizeof(float)));
    time_kernel("parallel_reduction", [&]() {
        parallel_reduction<<<reduction_blocks, 512, 512*sizeof(float)>>>(dA, d_reduction_out, N*N);
    });

    printf("\nAll kernels executed.\n");

    // Free memory
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dIntA));
    CUDA_CHECK(cudaFree(dIntB));
    CUDA_CHECK(cudaFree(d_reduction_out));

    return 0;
}