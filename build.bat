@echo off
rem Assuming CUDA toolkit is in the default location for version 13.0
set CUDA_BIN_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"

rem Check if nvcc exists
if not exist %CUDA_BIN_PATH%\nvcc.exe (
    echo Error: nvcc.exe not found in %CUDA_BIN_PATH%
    echo Please update CUDA_BIN_PATH in build.bat to your CUDA installation path.
    goto :eof
)

echo Building harness...
%CUDA_BIN_PATH%\nvcc.exe -o harness.exe harness/main.cu ^
    kernels/01_saxpy.cu ^
    kernels/02_naive_matmul.cu ^
    kernels/03_naive_transpose.cu ^
    kernels/04_tiled_transpose.cu ^
    kernels/05_bank_conflict.cu ^
    kernels/06_strided_global.cu ^
    kernels/07_branch_divergence.cu ^
    kernels/08_atomic_histogram.cu ^
    kernels/09_high_reg_pressure.cu ^
    kernels/10_parallel_reduction.cu ^
    kernels/11_tiled_matmul.cu ^
    -arch=native -std=c++14

if %errorlevel% neq 0 (
    echo Build failed.
) else (
    echo Build successful: harness.exe created.
)

:eof