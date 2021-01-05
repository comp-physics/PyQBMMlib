cmake_minimum_required (VERSION 3.16)

project (qbmmlib CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

# CUDA dependencies
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# OPEN_MP settings 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(OMP_NUM_THREADS 1)

add_executable(test_qmom main.cpp qmom_cuda.cu qmom_openmp.cpp)
