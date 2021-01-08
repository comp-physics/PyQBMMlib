
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>

__global__ void hyqmom2_idx(float mon[], float x[], float w[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // printf("hello from thread %d \n", idx);
        float C2 = ((mon[idx] * mon[idx]) - (mon[idx] * mon[idx])) 
                    / (mon[idx] * mon[idx]);
        float C3 = mon[idx]/2;
        float C4 = mon[idx]/2;
        float C5 = (mon[idx]/mon[idx]) - sqrt(C2);
        float C6 = (mon[idx]/mon[idx]) + sqrt(C2);
        mon[idx] = C6;
        // x[idx] = C6;
        idx += blockDim.x;
    };
}


__global__ void hyqmom2_kernel(float mon[], float x[], float w[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // printf("hello from thread %d \n", idx);
        float C2 = ((mon[3*idx] * mon[3*idx+2]) - (mon[3*idx+1] * mon[3*idx+1])) 
                    / (mon[3*idx] * mon[3*idx]);
        w[2*idx] = mon[3*idx]/2;
        w[2*idx+1] = mon[3*idx]/2;
        x[2*idx] = (mon[3*idx+1]/mon[3*idx]) - sqrt(C2);
        x[2*idx+1] = (mon[3*idx+1]/mon[3*idx]) + sqrt(C2);
        idx += blockDim.x;
    };
}

float hyqmom2_cuda(float input_moments[], int num_moments) {
    
    /* CUDA */
    float* x_out_cuda, *w_out_cuda;
    float* moment_d, *moment_h;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    moment_h = new float[num_moments*3];
    cudaMalloc(&x_out_cuda, sizeof(float)*num_moments*2);
    cudaMalloc(&moment_d, sizeof(float)*num_moments*3);
    cudaMalloc(&w_out_cuda, sizeof(float)*num_moments*2);
    cudaMemcpy(moment_d, input_moments, sizeof(float)*num_moments*3, cudaMemcpyHostToDevice);

    printf("[CUDA] starting calculation. Timer on ... \n");
    cudaEventRecord(start); //start the timer


    hyqmom2_kernel<<<1, 1024>>>(moment_d, x_out_cuda, w_out_cuda, num_moments);
    hyqmom2_idx<<<1, 1024>>> (moment_d, x_out_cuda, w_out_cuda, num_moments);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }

    cudaEventRecord(stop); //stop the timer
    printf("[CUDA] Finished calculation. Timer off... \n");

    cudaMemcpy(moment_h, moment_d, sizeof(float)*num_moments*3, cudaMemcpyDeviceToHost);

    float cuda_time; 
    cudaEventElapsedTime(&cuda_time, start, stop);

    return cuda_time * 1e-3;
}