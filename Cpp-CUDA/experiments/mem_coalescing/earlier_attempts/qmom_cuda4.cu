
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
    const int stride = 6;

    for (int idx = thread_idx; idx < N; idx += num_threads) {
        register float* moments = &(M[idx * stride]);
        register float c_moments[3];

        /* central moments */
        float denom = (moments[0] * moments[0]);
        c_moments[0] = ((moments[3] * moments[0]) - (moments[1] * moments[1])) 
                        / denom; // C_20
        c_moments[1] = ((moments[4] * moments[0]) - (moments[1] * moments[2])) 
                        / denom; // C_11
        c_moments[2] = ((moments[5] * moments[0]) - (moments[2] * moments[2])) 
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
        float w_res[4], x_res[4], y_res[4];

        w_res[0] = moments[0] * w1[0] * w2[0];
        w_res[1] = moments[0] * w1[0] * w2[1];
        w_res[2] = moments[0] * w1[1] * w2[0]; 
        w_res[3] = moments[0] * w1[1] * w2[1];
        memcpy(&w[4*idx], w_res, sizeof(float)*4);

        float x_res[4];
        float first_term = moments[1]/moments[0]; // for x_final
        x_res[0] = first_term + x1[0];
        x_res[1] = first_term + x1[0];
        x_res[2] = first_term + x1[1];
        x_res[3] = first_term + x1[1];
        memcpy(&x[4*idx], x_res, sizeof(float)*4);        

        float y_res[4];
        first_term = moments[2]/moments[0]; // for y_final
        y_res[0] = first_term + nu[0] + x2[0],
        y_res[1] = first_term + nu[0] + x2[1],
        y_res[2] = first_term + nu[1] + x2[0],
        y_res[3] = first_term + nu[1] + x2[1];

        memcpy(&y[4*idx], y_res, sizeof(float)*4);
    }
}

float qmom_cuda(float moments[], int num_moments,
                float xout[], float yout[], float wout[]) 
{

    // timer for measuring kernel execution time
    // measurement done in miliseconds
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("[CUDA] Setting up memmory... \n");
    float *moments_gpu;

    float *w_final_gpu, *x_final_gpu, *y_final_gpu;

    //// allocate device memory 
    // input
    gpuErrchk(cudaMalloc(&moments_gpu, sizeof(float)*num_moments*6));
    // final weight, abscissas: 
    gpuErrchk(cudaMalloc(&w_final_gpu, sizeof(float)*num_moments*4));
    gpuErrchk(cudaMalloc(&x_final_gpu, sizeof(float)*num_moments*4));
    cudaMalloc(&y_final_gpu, sizeof(float)*num_moments*4);  
    //copy input from host to device 
    gpuErrchk(cudaMemcpy(moments_gpu, moments, 
                    sizeof(float)*num_moments*6, cudaMemcpyHostToDevice)
    );


    // thread block is set to be 1D,
    int minGridSize, num_blocks, num_threads;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &num_threads, chyqmom4, 0, num_moments));
    num_blocks = (num_moments + num_threads - 1) / num_threads; 
    printf("[CUDA] calculated grid size: %d Block_size %d \n", num_blocks, num_threads);

    // int num_threads = 1024;
    // int num_blocks = 1;

    //// Calculating 
    printf("[CUDA] starting calculation. Timer on ... \n");
    cudaEventRecord(start); //start the timer
    chyqmom4<<<num_blocks, num_threads>>>(moments_gpu, w_final_gpu, x_final_gpu, y_final_gpu, num_moments);
    cudaEventRecord(stop); //stop the timer
    printf("[CUDA] Finished calculation. Timer off... \n");

    // copy result from device to host 
    gpuErrchk(cudaMemcpy(wout, w_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(xout, x_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(yout, y_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost));
    // --TODO-- verify the result somehow? 

    cudaEventSynchronize(stop);
    float calc_duration; 
    cudaEventElapsedTime(&calc_duration, start, stop);

    cudaFree(moments_gpu);
    cudaFree(w_final_gpu);
    cudaFree(x_final_gpu);
    cudaFree(y_final_gpu);
    return calc_duration;
}