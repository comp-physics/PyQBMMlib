cmake_minimum_required (VERSION 3.16)

project (qbmmlib CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

# CUDA dependencies
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 \
                        -lineinfo ")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# OPEN_MP settings 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

add_executable(run_chyqmom9 run_chyqmom9.cpp chyqmom9.cu hyqmom.cu)
add_executable(run_chyqmom4 run_chyqmom4.cpp chyqmom4.cu hyqmom.cu)

add_executable(compare_chyqmom4 compare_chyqmom4.cpp chyqmom4.cu qmom_openmp.cpp hyqmom.cu)
add_executable(compare_chyqmom9 compare_chyqmom9.cpp chyqmom9.cu qmom_openmp.cpp hyqmom.cu)

add_executable(run_chyqmom4_1kernel run_chyqmom4_1kernel.cpp chyqmom4_single_kernel)
add_executable(compare_1kernel compare_1kernel.cpp chyqmom4.cu chyqmom4_single_kernel.cu hyqmom.cu)
