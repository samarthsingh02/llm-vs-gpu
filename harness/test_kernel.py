import cupy as cp

# Define the CUDA kernel
kernel_code = r'''
extern "C" __global__
void hello_kernel() {
    printf("Hello GPU from Python!\\n");
}
'''

# Compile kernel
hello_kernel = cp.RawKernel(kernel_code, 'hello_kernel')

# Launch kernel: 1 block, 1 thread, no arguments
hello_kernel((1,), (1,), args=())

# Wait for GPU to finish
cp.cuda.Device().synchronize()
