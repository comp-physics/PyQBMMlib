#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cassert>
#include <chrono>

#include "hyqmom.hpp"

// a helper function for calculating nth moment 
__device__ float sum_pow(float rho[], float yf[], float n, const int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += rho[i] * pow(yf[i], n); 
    }
    return sum;
}

// set a segment of memory to a specific value
static __global__ void float_value_set(float *addr, float value, int size) {
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        addr[idx] = value;
    }
}

__global__ void chyqmom4_cmoments(const float moments[], float c_moments[], const int size, const int stride) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {
        float mom[6];
        mom[0] = moments[idx];
        // printf("[tIdx %d] mom[0] = %f\n", idx, mom[0]);
        // normalize mom by mom[0];
        // mom[i] = mom[i]/mom[0] for i !=0
        for (int n=1; n<6; n++) {
            mom[n] = moments[n * stride + idx] / mom[0];
            // printf("[tIdx %d] mom[%d] = %f\n", idx, n, mom[n]);
        }

        c_moments[0*stride + idx] = mom[3] - mom[1] * mom[1];
        c_moments[1*stride + idx] = mom[4] - mom[1] * mom[2];
        c_moments[2*stride + idx] = mom[5] - mom[2] * mom[2];
    };
};

static __global__ void chyqmom4_mu_yf(
    const float c_moments[], 
    const float xp[], 
    const float rho[],
    float yf[], 
    float mu[], 
    const int size, 
    const int stride) 
{
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x*2;
    for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {

        float c_11 = c_moments[stride + idx];
        float c_20 = c_moments[2*stride + idx];
        float coef = c_11/c_20;

        float yf_local[2] = {
            coef * xp[0*stride + idx],
            coef * xp[1*stride + idx],
        };

        float rho_local[2] = {
            rho[idx],          
            rho[1*stride + idx], 
        };
        float mu_avg = c_20 - sum_pow(rho_local, yf_local, 2, 3);
        yf[0*stride + idx] = yf_local[0];
        yf[1*stride + idx] = yf_local[1];
        mu[idx] = mu_avg;
    };
};

static __global__ void chyqmom4_wout(
    const float moments[], 
    const float r1[], 
    const float r2[], 
    float w[],
    const int size,
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float mom = moments[idx];
        for (int row = 0; row < 2; row ++) {
            for (int col = 0; col < 2; col ++) {
                w[(2*row + col) * stride + idx] = 
                    r1[row * stride + idx] * r2[col* stride + idx] * mom;
                // printf("[tIdx %d] w[%d] = %f \n", tIdx, (3*row + col) * stride + idx, w[(3*row + col) * stride + idx]);
            }
        }
    }
}

static __global__ void chyqmom4_xout(
    float moments[], 
    float xp[],
    float x[],
    const int size, 
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float x_local[2];
        float bx = moments[stride + idx] / moments[idx];
        for (int n = 0; n < 2; n++) {
            x_local[n] = xp[n * stride + idx];
        }
        for (int row = 0; row < 2; row ++) {
            float val = x_local[row] + bx;
            for (int col = 0; col < 2; col ++) {
                x[(2*row + col) * stride + idx] = val;
            }
        }
    }
}

static __global__ void chyqmom4_yout(
    float moments[], 
    float xp3[],
    float yf[],
    float y[],
    const int size,
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float x_local[2];
        float yf_local[2];
        
        for (int n = 0; n < 2; n++) {
            x_local[n] = xp3[n * stride + idx];
            yf_local[n]= yf[n * stride + idx];
        }
        float by = moments[2*stride + idx] / moments[idx];

        for (int row = 0; row < 2; row ++) {
            for (int col = 0; col < 2; col ++) {
                y[(2*row + col) * stride + idx] = yf_local[row] + x_local[col] + by;
            }
        }
    }
}

float chyqmom4(float moments[], const int size, float w[], float x[], float y[], const int batch_size) {

    // timer for measuring kernel execution time
    // measurement done in miliseconds
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // printf("[CUDA] Setting up memmory... \n");
    float *moments_d, *c_moments;
    float *m, *w1, *w2, *x1, *x2;
    float *yf, *mu;
    float *w_out_d, *x_out_d, *y_out_d;

    //// allocate device memory 
    // input
    gpuErrchk(cudaMalloc(&moments_d, sizeof(float)*size*6));
    // Central moments
    gpuErrchk(cudaMalloc(&c_moments, sizeof(float)*size*3));
    // intermediate M, w, x as input and output of HyQMOM2
    gpuErrchk(cudaMalloc(&m, sizeof(float)*size*3));
    gpuErrchk(cudaMalloc(&w1, sizeof(float)*size*2));
    gpuErrchk(cudaMalloc(&w2, sizeof(float)*size*2));
    gpuErrchk(cudaMalloc(&x1, sizeof(float)*size*2));
    gpuErrchk(cudaMalloc(&x2, sizeof(float)*size*2));
    // intermediate values 
    gpuErrchk(cudaMalloc(&yf, sizeof(float)*size*2));
    gpuErrchk(cudaMalloc(&mu, sizeof(float)*size));
    // final weight, abscissas: 
    gpuErrchk(cudaMalloc(&w_out_d, sizeof(float)*size*4));
    gpuErrchk(cudaMalloc(&x_out_d, sizeof(float)*size*4));
    gpuErrchk(cudaMalloc(&y_out_d, sizeof(float)*size*4));  

    // Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(moments, size*6*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(w, size*4*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(x, size*4*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(y, size*4*sizeof(float), cudaHostRegisterPortable));


    // Set up streams
    // Allocate 3 concurrent streams to each batch
    const int num_streams = batch_size*3;
    cudaStream_t stream[num_streams];
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    // thread block is set to be 1D,
    int gridSize, blockSize;
    blockSize = 1024;
    gridSize = (size + blockSize - 1) / blockSize; 
    // printf("[CHYQMOM4] Grid Size: %d Block Size: %d\n", gridSize, blockSize);

    int size_per_batch = ceil((float)size / batch_size);
    printf("[CHYQMOM9] streams: %d size: %d, size_per_batch: %d\n",num_streams, size, size_per_batch);

    gpuErrchk(cudaEventRecord(start));
    for (int i=0; i<num_streams; i+=3) {
        // beginning location in memory 
        int loc = (i/3) * size_per_batch;
        if (loc + size_per_batch > size) {
            size_per_batch = size - loc;
        }

        // transfer data from host to device 
        gpuErrchk(cudaMemcpy2DAsync(&moments_d[loc], size*sizeof(float), 
                                    &moments[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 6, 
                                    cudaMemcpyHostToDevice, stream[i]));
        // Central moments
        chyqmom4_cmoments<<<gridSize, blockSize, 0, stream[i]>>>(&moments_d[loc], &c_moments[loc], size_per_batch, size);
        // setup first hyqmom2
        float_value_set<<<gridSize, blockSize, 0, stream[i+1]>>>(&m[loc], 1, size_per_batch);
        float_value_set<<<gridSize, blockSize, 0, stream[i+2]>>>(&m[size + loc], 0, size_per_batch);
        gpuErrchk(cudaMemcpyAsync(&m[2* size + loc], &c_moments[loc], size_per_batch*sizeof(float), 
                                    cudaMemcpyDeviceToDevice, stream[i]));

        hyqmom2<<<gridSize, blockSize, 0, stream[i]>>>(&m[loc], &x1[loc], &w1[loc], size_per_batch, size);
        // Compute mu and yf
        chyqmom4_mu_yf<<<gridSize, blockSize, 0, stream[i]>>>(&c_moments[loc], &x1[loc], &w1[loc], &yf[loc], &mu[loc], size_per_batch, size);
        // Set up second hyqmom3
        float_value_set<<<gridSize, blockSize, 0, stream[i+1]>>>(&m[loc], 1, size_per_batch);
        float_value_set<<<gridSize, blockSize, 0, stream[i+2]>>>(&m[size + loc], 0, size_per_batch);
        gpuErrchk(cudaMemcpyAsync(&m[2* size + loc], &mu[loc], size_per_batch*sizeof(float), 
                                    cudaMemcpyDeviceToDevice, stream[i]));

        hyqmom2<<<gridSize, blockSize, 0, stream[i]>>>(&m[loc], &x2[loc], &w2[loc], size_per_batch, size);

        // compute weight and copy data to host 
        chyqmom4_wout<<<gridSize, blockSize, 0, stream[i]>>>(&moments_d[loc], &w1[loc], &w2[loc], &w_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&y[loc], size*sizeof(float), 
                                    &y_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 4, 
                                    cudaMemcpyDeviceToHost, stream[i+2]));

        // compute x and copy data to host 
        chyqmom4_xout<<<gridSize, blockSize, 0, stream[i+1]>>>(&moments_d[loc], &x1[loc], &x_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&x[loc], size*sizeof(float), 
                                    &x_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 4, 
                                    cudaMemcpyDeviceToHost, stream[i+1]));
        // compute y and copy data to host 
        chyqmom4_yout<<<gridSize, blockSize, 0, stream[i+2]>>>(&moments_d[loc], &x2[loc], &yf[loc], &y_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&w[loc], size*sizeof(float), 
                                    &w_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 4, 
                                    cudaMemcpyDeviceToHost, stream[i]));                         
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    // Unregisters host memory 
    gpuErrchk(cudaHostUnregister(moments));
    gpuErrchk(cudaHostUnregister(w));
    gpuErrchk(cudaHostUnregister(x));
    gpuErrchk(cudaHostUnregister(y));
    
    float calc_duration;
    cudaEventElapsedTime(&calc_duration, start, stop);
    // clean up
    cudaFree(moments_d);
    cudaFree(w_out_d);
    cudaFree(x_out_d);
    cudaFree(y_out_d);
    cudaFree(c_moments);
    cudaFree(mu);
    cudaFree(m);
    cudaFree(yf);
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(w1);
    cudaFree(w2);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return calc_duration;
}