
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

static __global__ void chyqmom9_cmoments(
    const float moments[], 
    float c_moments[],
    const int size, 
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        // copy moments to local registers
        float mom[10], cmom[7];
        mom[0] = moments[idx];
        // printf("[tIdx %d] mom[0] = %f\n", idx, mom[0]);
        // normalize mom by mom[0];
        // mom[i] = mom[i]/mom[0] for i !=0
        for (int n=1; n<10; n++) {
            mom[n] = moments[n * stride + idx] / mom[0];
            // printf("[tIdx %d] mom[%d] = %f\n", idx, n, mom[n]);
        }
        //compute central moments
        cmom[0] = mom[3] - mom[1] * mom[1];
        cmom[1] = mom[4] - mom[1] * mom[2];
        cmom[2] = mom[5] - mom[2] * mom[2];
        cmom[3] = mom[6] - 3*mom[1]*mom[3] + 2*mom[1]*mom[1]*mom[1];
        cmom[4] = mom[7] - 3*mom[2]*mom[5] + 2*mom[2]*mom[2]*mom[2];
        cmom[5] = mom[8] - 4*mom[1]*mom[6] + 6*mom[1]*mom[1]*mom[3] -
        3*mom[1]*mom[1]*mom[1]*mom[1];
        cmom[6] = mom[9] - 4*mom[2]*mom[7] + 6*mom[2]*mom[2]*mom[5] -
        3*mom[2]*mom[2]*mom[2]*mom[2];

        c_moments[idx] = cmom[0];
        c_moments[1*stride + idx] =cmom[1];
        c_moments[2*stride + idx] =cmom[2];
        c_moments[3*stride + idx] =cmom[3];
        c_moments[4*stride + idx] =cmom[4];
        c_moments[5*stride + idx] =cmom[5];
        c_moments[6*stride + idx] =cmom[6];
    }
}

static __global__ void chyqmom9_mu_yf(
    const float c_moments[], 
    const float xp[], 
    const float rho[],
    float yf[], 
    float mu[], 
    const int size, 
    const int stride) 
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float c_local[5] = {
            c_moments[idx],             // c02
            c_moments[1*stride + idx],  // c11
            c_moments[2*stride + idx],  // c20
            c_moments[4*stride + idx],  // c03
            c_moments[6*stride + idx]   // c04
        };
        float mu_avg = c_local[2] - c_local[1]*c_local[1]/c_local[0];
        float rho_local[3] = {
            rho[idx],          
            rho[1*stride + idx], 
            rho[2*stride + idx]  
        };
        float yf_local[3] = {
            c_local[1] * xp[idx] / c_local[2],
            c_local[1] * xp[stride + idx] / c_local[2],
            c_local[1] * xp[2*stride + idx] / c_local[2]
        };
        yf[idx] = yf_local[0];
        yf[stride + idx] = yf_local[1];
        yf[2*stride + idx] = yf_local[2];

        // if mu > csmall
        float q = (c_local[3] - sum_pow(rho_local, yf_local, 3.0, 3)) / 
                    pow(mu_avg, (3.0 / 2.0));
        float eta = (c_local[4] - sum_pow(rho_local, yf_local, 4.0, 3) - 
                    6 * sum_pow(rho_local, yf_local, 2.0, 3) * mu_avg) / 
                    pow(mu_avg, 2.0);

        float mu3 = q * pow(mu_avg, 3/2);
        float mu4 = eta * mu_avg * mu_avg;

        mu[idx] = mu_avg;
        mu[stride + idx] = mu3;

        mu[2*stride + idx] = mu4;
    }
}

static __global__ void chyqmom9_wout(
    float moments[], 
    float rho_1[], 
    float rho_2[], 
    float w[],
    const int size,
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float r1[3], r2[3];
        float mom = moments[idx];
        for (int n=0; n<3; n++) {
            r1[n] = rho_1[n * stride + idx];
            r2[n] = rho_2[n * stride + idx];
        }
        
        for (int row = 0; row < 3; row ++) {
            for (int col = 0; col < 3; col ++) {
                w[(3*row + col) * stride + idx] = r1[row] * r2[col] * mom;
                // printf("[tIdx %d] w[%d] = %f \n", tIdx, (3*row + col) * stride + idx, w[(3*row + col) * stride + idx]);
            }
        }
    }
}

static __global__ void chyqmom9_xout(
    float moments[], 
    float xp[],
    float x[],
    const int size, 
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float x_local[3];
        float bx = moments[stride + idx] / moments[idx];
        for (int n = 0; n < 3; n++) {
            x_local[n] = xp[n * stride + idx];
        }
        for (int row = 0; row < 3; row ++) {
            float val = x_local[row] + bx;
            for (int col = 0; col < 3; col ++) {
                x[(3*row + col) * stride + idx] = val;
            }
        }
    }
}

static __global__ void chyqmom9_yout(
    float moments[], 
    float xp3[],
    float yf[],
    float y[],
    const int size,
    const int stride)
{
    const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
        float x_local[3];
        float yf_local[3];
        
        for (int n = 0; n < 3; n++) {
            x_local[n] = xp3[n * stride + idx];
            yf_local[n]= yf[n * stride + idx];
        }
        float by = moments[2*stride + idx] / moments[idx];

        for (int row = 0; row < 3; row ++) {
            for (int col = 0; col < 3; col ++) {
                y[(3*row + col) * stride + idx] = yf_local[row] + x_local[col] + by;
            }
        }
    }
}

float chyqmom9(float moments[], const int size, float w[], float x[], float y[], const int batch_size) {

    using namespace std::chrono;

    float *moments_d, *w_out_d, *x_out_d, *y_out_d;
    float *c_moments, *mu, *yf;
    float *m1, *x1, *w1, *x2, *w2;

    // memory allocation
    gpuErrchk(cudaMalloc(&moments_d, sizeof(float)*size*10));
    gpuErrchk(cudaMalloc(&w_out_d, sizeof(float)*size*9));
    gpuErrchk(cudaMalloc(&x_out_d, sizeof(float)*size*9));
    gpuErrchk(cudaMalloc(&y_out_d, sizeof(float)*size*9));

    gpuErrchk(cudaMalloc(&c_moments, sizeof(float)*size*7));
    gpuErrchk(cudaMalloc(&mu, sizeof(float)*size*3));
    gpuErrchk(cudaMalloc(&yf, sizeof(float)*size*3));

    gpuErrchk(cudaMalloc(&m1, sizeof(float)*size*5));
    gpuErrchk(cudaMalloc(&x1, sizeof(float)*size*3));
    gpuErrchk(cudaMalloc(&w1, sizeof(float)*size*3));
    gpuErrchk(cudaMalloc(&x2, sizeof(float)*size*3));
    gpuErrchk(cudaMalloc(&w2, sizeof(float)*size*3));

    // Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(moments, size*10*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(w, size*9*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(x, size*9*sizeof(float), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(y, size*9*sizeof(float), cudaHostRegisterPortable));

    // Set up streams
    // Allocate 3 concurrent streams to each batch
    const int num_streams = batch_size*3;
    cudaStream_t stream[num_streams];
    for (int i=0; i<num_streams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    // Calculate optimal block and grid sizes
    int gridSize, blockSize;
    blockSize = 1024;
    gridSize = (size + blockSize - 1) / blockSize; 
    printf("[CHYQMOM9] Grid Size: %d Block Size: %d\n", gridSize, blockSize);
    // setup timer 
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    // cudaProfilerStart();
    
    int size_per_batch = ceil(size / batch_size);
    printf("[CHYQMOM9] streams: %d size: %d, size_per_batch: %d\n",num_streams, size, size_per_batch);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    gpuErrchk(cudaEventRecord(start));
    for (int i=0; i<num_streams; i+=3) {
        // beginning location in memory 
        int loc = (i/3) * size_per_batch;
        // transfer data from host to device 


        gpuErrchk(cudaMemcpy2DAsync(&moments_d[loc], size*sizeof(float), 
                                    &moments[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 10, 
                                    cudaMemcpyHostToDevice, stream[i]));
        
        // Central moments
        chyqmom9_cmoments<<<gridSize, blockSize, 0, stream[i]>>>(&moments_d[loc], &c_moments[loc], size_per_batch, size);
        // setup first hyqmom3
        float_value_set<<<gridSize, blockSize, 0, stream[i+1]>>>(&m1[loc], 1, size_per_batch);
        float_value_set<<<gridSize, blockSize, 0, stream[i+2]>>>(&m1[size + loc], 0, size_per_batch);
        gpuErrchk(cudaMemcpyAsync(&m1[2* size + loc], &c_moments[loc], size_per_batch*sizeof(float), 
                                    cudaMemcpyDeviceToDevice, stream[i]));
        gpuErrchk(cudaMemcpy2DAsync(&m1[3*size + loc], size*sizeof(float), 
                                    &c_moments[4*size + loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 2, 
                                    cudaMemcpyDeviceToDevice, stream[i]));


        hyqmom3<<<gridSize, blockSize, 0, stream[i]>>>(&m1[loc], &x1[loc], &w1[loc], size_per_batch, size);
        // Compute mu and yf
        chyqmom9_mu_yf<<<gridSize, blockSize, 0, stream[i]>>>(&c_moments[loc], &x1[loc], &w1[loc], &yf[loc], &mu[loc], size_per_batch, size);
        // Set up second hyqmom3
        float_value_set<<<gridSize, blockSize, 0, stream[i+1]>>>(&m1[loc], 1, size_per_batch);
        float_value_set<<<gridSize, blockSize, 0, stream[i+2]>>>(&m1[size + loc], 0, size_per_batch);
        gpuErrchk(cudaMemcpy2DAsync(&m1[2*size + loc], size*sizeof(float), 
                                    &mu[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 3, 
                                    cudaMemcpyDeviceToDevice, stream[i]));

        hyqmom3<<<gridSize, blockSize, 0, stream[i]>>>(&m1[loc], &x2[loc], &w2[loc], size_per_batch, size);
        // cudaStreamSynchronize(stream[i]);
        // cudaStreamSynchronize(stream[i+1]);
        // cudaStreamSynchronize(stream[i+2]);

        // compute weight and copy data to host 
        chyqmom9_wout<<<gridSize, blockSize, 0, stream[i]>>>(&moments_d[loc], &w1[loc], &w2[loc], &w_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&w[loc], size*sizeof(float), 
                                    &w_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 9, 
                                    cudaMemcpyDeviceToHost, stream[i]));

        // compute x and copy data to host 
        chyqmom9_xout<<<gridSize, blockSize, 0, stream[i+1]>>>(&moments_d[loc], &x1[loc], &x_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&x[loc], size*sizeof(float), 
                                    &x_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 9, 
                                    cudaMemcpyDeviceToHost, stream[i+1]));
        // compute y and copy data to host 
        chyqmom9_yout<<<gridSize, blockSize, 0, stream[i+2]>>>(&moments_d[loc], &x2[loc], &yf[loc], &y_out_d[loc], size_per_batch, size);
        gpuErrchk(cudaMemcpy2DAsync(&y[loc], size*sizeof(float), 
                                    &y_out_d[loc], size*sizeof(float),
                                    size_per_batch * sizeof(float), 9, 
                                    cudaMemcpyDeviceToHost, stream[i+2]));
    }
    cudaDeviceSynchronize();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();


    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    float calc_duration;
    cudaEventElapsedTime(&calc_duration, start, stop);
    printf("[CUDA] %f ms \n", calc_duration);
    printf("[CHRONO] %f ms \n", time_span.count()*1e3);
    // clean up
    cudaFree(moments_d);
    cudaFree(w_out_d);
    cudaFree(x_out_d);
    cudaFree(y_out_d);
    cudaFree(c_moments);
    cudaFree(mu);
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