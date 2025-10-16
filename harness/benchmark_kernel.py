import cupy as cp
import time
import csv
import os

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# CUDA kernel code
kernel_code = r'''
extern "C" __global__ void hello_kernel() {
    printf("Hello GPU from Python!\\n");
}
'''

# Compile kernel
hello_kernel = cp.RawKernel(kernel_code, 'hello_kernel')

# Benchmark settings
num_runs = 10
results = []

for i in range(num_runs):
    start = time.time()
    # Launch kernel: 1 block, 1 thread, no arguments
    hello_kernel((1,), (1,), args=())
    cp.cuda.Device().synchronize()  # wait for GPU
    end = time.time()
    elapsed = end - start
    results.append(elapsed)
    print(f"Run {i+1}: {elapsed:.6f} seconds")

# Save results to CSV
csv_file = "data/benchmark_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "TimeSeconds"])
    for idx, t in enumerate(results, 1):
        writer.writerow([idx, t])

print(f"\nResults saved to {csv_file}")
