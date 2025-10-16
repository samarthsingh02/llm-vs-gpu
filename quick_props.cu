// quick_props.cu (Windows compatible)
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("Name: %s\n", p.name);
    printf("SMs: %d\n", p.multiProcessorCount);
    printf("WarpSize: %d\n", p.warpSize);
    printf("Regs/SM: %d\n", p.regsPerMultiprocessor);
    printf("SharedMem/SM KB: %zu\n", p.sharedMemPerMultiprocessor / 1024);
    printf("Total global mem MB: %zu\n", p.totalGlobalMem / (1024*1024));
    printf("Memory Bus Width bits: %d\n", p.memoryBusWidth);

    // Comment out memoryClockRate if compilation fails
    // printf("Memory Clock KHz: %d\n", p.memoryClockRate);

    return 0;
}
