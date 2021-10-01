cmake_minimum_required (VERSION 3.16)

project (mem_coalese_experiment CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 \
                        -lineinfo ")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(mem_coal main.cpp dummy_hyqmom.cu init.cpp)
add_executable(compare plot_res.cpp dummy_hyqmom.cu init.cpp)
