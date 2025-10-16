cmake_minimum_required(VERSION 3.18)
project(kernel_harness LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find the CUDA toolkit to handle linking correctly
find_package(CUDAToolkit REQUIRED)

# Explicitly list all source files
add_executable(harness
    harness/main.cu
    kernels/01_saxpy.cu
    kernels/02_naive_matmul.cu
    kernels/03_naive_transpose.cu
    kernels/04_tiled_transpose.cu
    kernels/05_bank_conflict.cu
    kernels/06_strided_global.cu
    kernels/07_branch_divergence.cu
    kernels/08_atomic_histogram.cu
    kernels/09_high_reg_pressure.cu
    kernels/10_parallel_reduction.cu
)

# Link the CUDA runtime library explicitly
target_link_libraries(harness PRIVATE CUDA::cudart)

# Set the native architecture
set_target_properties(harness PROPERTIES CUDA_ARCHITECTURES "native")

# Optional: Suppress common warnings for MSVC
if(MSVC)
    target_compile_options(harness PRIVATE /W3 /EHsc)
endif()