cmake_minimum_required(VERSION 3.18)
project(kernel_harness LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # important for multiple .cu files

# Collect source files
file(GLOB KERNEL_SOURCES kernels/*.cu)
set(HARNESS_SOURCES harness/main.cu)

# Add executable
add_executable(harness ${HARNESS_SOURCES} ${KERNEL_SOURCES})

# Set native CUDA architecture
set_target_properties(harness PROPERTIES CUDA_ARCHITECTURES "native")

# Optional: suppress warnings for MSVC
if(MSVC)
    target_compile_options(harness PRIVATE /W3 /EHsc)
endif()
