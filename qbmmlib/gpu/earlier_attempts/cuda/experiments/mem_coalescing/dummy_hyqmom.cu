#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cassert>

//input and oytput are row majored

static __global__ void hyqmom2_naive(float mon[], float x[], float w[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        // printf("hello from thread %d \n", idx);
        float C2 = ((mon[3*idx] * mon[3*idx+2]) - (mon[3*idx+1] * mon[3*idx+1])) 
                    / (mon[3*idx] * mon[3*idx]);
        w[2*idx] = mon[3*idx]/2;
        w[2*idx+1] = mon[3*idx]/2;
        x[2*idx] = (mon[3*idx+1]/mon[3*idx]) - sqrt(C2);
        x[2*idx+1] = (mon[3*idx+1]/mon[3*idx]) + sqrt(C2);
        idx += blockDim.x*gridDim.x;
    };
}

// input and output are column majored

static __global__ void hyqmom2_coalesced(float mon[], float x[], float w[], int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mon_local[3];
    while (idx < N) {
        for (int i = 0; i < 3; i++) {
            mon_local[i] = mon[i * N + idx];
            // printf("thread [%d]: mon_local[%d] = %f \n", idx, i, mon_local[i]);
        }
        float C2 = ((mon_local[0] * mon_local[2]) - (mon_local[1] * mon_local[1])) 
            / (mon_local[0] * mon_local[0]);
        for (int i=0; i<2; i++) {
            w[i*N+idx] = mon_local[0]/2;
        }
        x[idx] = (mon_local[1]/mon_local[0]) - sqrt(C2);
        x[N + idx] = (mon_local[1]/mon_local[0]) + sqrt(C2);
        idx += blockDim.x*gridDim.x;
    }
}


float run_naive(const float moment[], const int size, float x_out[], float w_out[]) {
    float *x_out_cuda, *w_out_cuda, *moment_d;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc(&x_out_cuda, sizeof(float)*size*2);
    cudaMalloc(&moment_d, sizeof(float)*size*3);
    cudaMalloc(&w_out_cuda, sizeof(float)*size*2);

    cudaMemcpy(moment_d, moment, sizeof(float)*size*3, cudaMemcpyHostToDevice);

    int gridSize = ceil(size/1024);
    cudaEventRecord(start);
    hyqmom2_naive<<<gridSize, 1024>>>(moment_d, x_out_cuda, w_out_cuda, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(x_out, x_out_cuda, sizeof(float)*size*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_out, w_out_cuda, sizeof(float)*size*2, cudaMemcpyDeviceToHost);

    float calc_duration; 
    cudaEventElapsedTime(&calc_duration, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x_out_cuda);
    cudaFree(w_out_cuda);
    cudaFree(moment_d);

    return calc_duration;
}

float run_coal(const float moment[], const int size, float x_out[], float w_out[]) {
    float *x_out_cuda, *w_out_cuda, *moment_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&x_out_cuda, sizeof(float)*size*2);
    cudaMalloc(&moment_d, sizeof(float)*size*3);
    cudaMalloc(&w_out_cuda, sizeof(float)*size*2);

    cudaMemcpy(moment_d, moment, sizeof(float)*size*3, cudaMemcpyHostToDevice);

    int gridSize = ceil(size/1024);
    cudaEventRecord(start);
    hyqmom2_coalesced<<<gridSize, 1024>>>(moment_d, x_out_cuda, w_out_cuda, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(x_out, x_out_cuda, sizeof(float)*size*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_out, w_out_cuda, sizeof(float)*size*2, cudaMemcpyDeviceToHost);

    float calc_duration; 
    cudaEventElapsedTime(&calc_duration, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x_out_cuda);
    cudaFree(w_out_cuda);
    cudaFree(moment_d);

    return calc_duration;
}
