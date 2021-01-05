#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_kernel(int N) {
    // 1D block thread indexing 
    printf("hello! ");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread_idx: %d got integer: %d \n", idx, N);
}

int main() {
    int N = 5;
    printf("passing number %d to GPU kernel ...\n", N);
    test_kernel<<<1, 10>>>(N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
    return 0;

}