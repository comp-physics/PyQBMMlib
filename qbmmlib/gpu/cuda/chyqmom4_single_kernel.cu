#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cudaErr.hpp"
#include "main.hpp"

/**********************************
 * gpu kernels 
 */

__global__ void chyqmom4(float *M, float *w, float *x, float *y, int N) {
    /* 1D block and grid shapes */
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x *gridDim.x;

    for (int idx = thread_idx; idx < N; idx += num_threads) {

        register float c_moments[3];

        /* central moments */
        float denom = (M[6*idx+0] * M[6*idx+0]);
        c_moments[0] = ((M[6*idx+3] * M[6*idx+0]) - (M[6*idx+1] * M[6*idx+1])) 
                        / denom; // C_20
        c_moments[1] = ((M[6*idx+4] * M[6*idx+0]) - (M[6*idx+1] * M[6*idx+2])) 
                        / denom; // C_11
        c_moments[2] = ((M[6*idx+5] * M[6*idx+0]) - (M[6*idx+2] * M[6*idx+2])) 
                        / denom; // C_02

        /* first hyqmom2 */
        // input is {1, 0, C_20}
        float w1[2] = {1.0/2.0, 1.0/2.0};
        float x1[2] = {0-sqrt(c_moments[0]), 0+sqrt(c_moments[0])};

        /* intermediate */
        float coef = c_moments[1]/c_moments[0];
        float nu[2] = {coef*x1[0], coef*x1[1]};
        float mu = c_moments[2] - coef*c_moments[1];

        /* second hyqmom2*/
        // input is {1, 0, mu}
        float w2[2] = {w1[0], w1[1]};
        float x2[2] = {0-sqrt(mu), 0+sqrt(mu)};

        /*final results */
        // float w_res[4], x_res[4], y_res[4];

        w[4*idx+0] = M[6*idx+0] * w1[0] * w2[0];
        w[4*idx+1] = M[6*idx+0] * w1[0] * w2[1];
        w[4*idx+2] = M[6*idx+0] * w1[1] * w2[0]; 
        w[4*idx+3] = M[6*idx+0] * w1[1] * w2[1];
        // memcpy(&w[4*idx], w_res, sizeof(float)*4);

        float first_term = M[6*idx+1]/M[6*idx+0]; // for x_final
        x[4*idx+0] = first_term + x1[0];
        x[4*idx+1] = first_term + x1[0];
        x[4*idx+2] = first_term + x1[1];
        x[4*idx+3] = first_term + x1[1];
        // memcpy(&x[4*idx], x_res, sizeof(float)*4);        

        first_term = M[6*idx+2]/M[6*idx+0]; // for y_final
        y[4*idx+0] = first_term + nu[0] + x2[0],
        y[4*idx+1] = first_term + nu[0] + x2[1],
        y[4*idx+2] = first_term + nu[1] + x2[0],
        y[4*idx+3] = first_term + nu[1] + x2[1];

        // memcpy(&y[4*idx], y_res, sizeof(float)*4);
    }
}

float chyqmom4_single_kernel(float moments[], int num_moments,
                float xout[], float yout[], float wout[]) 
{

    // timer for measuring kernel execution time
    // measurement done in miliseconds
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *moments_gpu;

    float *w_final_gpu, *x_final_gpu, *y_final_gpu;

    float *moment_col_major = new float[num_moments*6];
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < num_moments; col ++) {
            moment_col_major[col * 6 + row] = moments[row * num_moments + col];
        }
    }

    //// allocate device memory 
    // input
    gpuErrchk(cudaMalloc(&moments_gpu, sizeof(float)*num_moments*6));
    // final weight, abscissas: 
    gpuErrchk(cudaMalloc(&w_final_gpu, sizeof(float)*num_moments*4));
    gpuErrchk(cudaMalloc(&x_final_gpu, sizeof(float)*num_moments*4));
    cudaMalloc(&y_final_gpu, sizeof(float)*num_moments*4);  
    //copy input from host to device 

    // thread block is set to be 1D,
    int minGridSize, num_blocks, num_threads;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &num_threads, chyqmom4, 0, num_moments));
    num_blocks = (num_moments + num_threads - 1) / num_threads; 

    // int num_threads = 1024;
    // int num_blocks = 1;

    //// Calculating 
    cudaEventRecord(start); //start the timer
    gpuErrchk(cudaMemcpy(moments_gpu, moment_col_major, 
        sizeof(float)*num_moments*6, cudaMemcpyHostToDevice)
    );
    chyqmom4<<<num_blocks, num_threads>>>(moments_gpu, w_final_gpu, x_final_gpu, y_final_gpu, num_moments);


    // copy result from device to host 
    gpuErrchk(cudaMemcpy(wout, w_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(xout, x_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(yout, y_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    // --TODO-- verify the result somehow? 
    cudaEventRecord(stop); //stop the timer
    cudaEventSynchronize(stop);


    float calc_duration; 
    cudaEventElapsedTime(&calc_duration, start, stop);

    cudaFree(moments_gpu);
    cudaFree(w_final_gpu);
    cudaFree(x_final_gpu);
    cudaFree(y_final_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    delete[] moment_col_major;

    return calc_duration;
} 