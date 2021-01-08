cmake_minimum_required (VERSION 3.16)

project (qbmmlib CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

# CUDA dependencies
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 \
                        -gencode arch=compute_61,code=compute_61 \
                        -gencode arch=compute_75,code=sm_75")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# OPEN_MP settings 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

add_executable(test_qmom1 main.cpp qmom_cuda.cu qmom_openmp.cpp)
add_executable(test_qmom2 main.cpp qmom_cuda2.cu qmom_openmp.cpp)
add_executable(test_qmom3 main.cpp qmom_cuda3.cu qmom_openmp.cpp)
add_executable(test_cuda basic_example.cu)
add_executable(test_hyqmom2 test_hyqmom2.cpp test_hyqmom2.cu)
