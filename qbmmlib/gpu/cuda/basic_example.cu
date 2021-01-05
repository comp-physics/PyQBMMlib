#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_kernel(int N) {
    // 1D block thread indexing 
    printf("hello! ");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread_idx: %d got integer: %d \n", idx, N);
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int num_blocks = atoi(argv[3]);
    printf("passing number %d to GPU kernel ...\n", N);
    test_kernel<<<num_blocks, num_threads>>>(N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
    return 0;

}