
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cudaErr.hpp"
#include "main.hpp"

/**********************************
 * gpu kernels 
 */

__global__ void c20_kernel(float* M, float* c20, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        c20[idx] = ((M[6*idx+3] * M[6*idx]) - (M[6*idx+1] * M[6*idx+1])) 
                    / (M[6*idx] * M[6*idx]);
        idx += blockDim.x;
    };
};

__global__ void c11_kernel(float* M, float* c11, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        c11[idx] = ((M[6*idx+4] * M[6*idx]) - (M[6*idx+1] * M[6*idx+2])) 
                    / (M[6*idx] * M[6*idx]);
        idx += blockDim.x;
    };
};

__global__ void c02_kernel(float* M, float* c02, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        c02[idx] = ((M[6*idx+5] * M[6*idx]) - (M[6*idx+2] * M[6*idx+2])) 
                    / (M[6*idx] * M[6*idx]);
        idx += blockDim.x;
    };
};

__global__ void init_M(float* value, float* M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        M[3*idx] = 1;
        M[3*idx+1] = 0;
        M[3*idx+2] = value[idx];
        idx += blockDim.x;
    };
};

__global__ void nu_kernel(float* c11, float* c20, float* xi, float* nu, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        float c = c11[idx]/c20[idx];
        nu[2*idx] = c*xi[2*idx];
        nu[2*idx+1] = c*xi[2*idx+1];
        idx += blockDim.x;
    };
};

__global__ void mu_kernel(float* c02, float* nu, float* w, float* mu, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        mu[idx] = c02[idx] - (w[2*idx]*nu[2*idx]*nu[2*idx] 
                    + w[2*idx+1]*nu[2*idx+1]*nu[2*idx+1]);
        idx += blockDim.x;
    };
};

__global__ void mu_kernel2(float* c02, float* c11, float* c20, float* mu, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        mu[idx] = c02[idx] - c11[idx]*c11[idx]/c20[idx];
        idx += blockDim.x;
    };
};

__global__ void hyqmom2_kernel(float* M, float* w, float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        float C2 = ((M[3*idx] * M[3*idx+2]) - (M[3*idx+1] * M[3*idx+1])) 
                    / (M[3*idx] * M[3*idx]);
        w[2*idx] = M[3*idx]/2;
        w[2*idx+1] = M[3*idx]/2;
        x[2*idx] = (M[3*idx+1]/M[3*idx]) - sqrt(C2);
        x[2*idx+1] = (M[3*idx+1]/M[3*idx]) + sqrt(C2);
        // printf("[hyqmom2] w: %f %f, x: %f %f \n", w[2*idx], w[2*idx+1], x[2*idx], x[2*idx+1]);
        idx += blockDim.x;
    };
};

__global__ void weight_kernel(float* M, float* w1, float* w2, float* w_final, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x*2;

    while (idx < N) {

        float4 *w_final_4 = reinterpret_cast<float4*>(&(w_final[idx*4]));
        float4 *w1_4 = reinterpret_cast<float4*>(&(w1[idx*2]));
        float4 *w2_4 = reinterpret_cast<float4*>(&(w2[idx*2]));

        float4 temp_final1;
        float4 temp_final2;
        float temp_M6_0 = M[idx*6];
        float temp_M6_1 = M[(idx+1)*6];
        float4 temp_w1 = w1_4[0];
        float4 temp_w2 = w2_4[0];

        temp_final1.x = temp_M6_0 * temp_w1.x * temp_w2.x;
        temp_final1.y = temp_M6_0 * temp_w1.x * temp_w2.y;
        temp_final1.z = temp_M6_0 * temp_w1.y * temp_w2.x;
        temp_final1.w = temp_M6_0 * temp_w1.y * temp_w2.y;

        temp_final2.x = temp_M6_1 * temp_w1.z * temp_w2.z;
        temp_final2.y = temp_M6_1 * temp_w1.z * temp_w2.w;
        temp_final2.z = temp_M6_1 * temp_w1.w * temp_w2.z;
        temp_final2.w = temp_M6_1 * temp_w1.w * temp_w2.w;

        w_final_4[0] = temp_final1;
        w_final_4[1] = temp_final2;
        
        // printf("[thread %d] temp_x1: %f %f %f %f \n" , idx, temp_x1.x, temp_x1.y, temp_x1.z, temp_x1.w);
        // printf("[thread %d] quotient: %f \n",idx, quotient2);
        // printf("[thread %d] temp_final2: %f %f %f %f \n" , idx, temp_final2.x, temp_final2.y, temp_final2.z, temp_final2.w);
        idx += 2*blockDim.x;
    };
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // while (idx < N) {
    //     w_final[4*idx] = M[6*idx] * w1[2*idx] * w2[2*idx];
    //     w_final[4*idx+1] = M[6*idx] * w1[2*idx] * w2[2*idx+1];
    //     w_final[4*idx+2] = M[6*idx] * w1[2*idx+1] * w2[2*idx];
    //     w_final[4*idx+3] = M[6*idx] * w1[2*idx+1] * w2[2*idx+1];
    //     idx += blockDim.x;
    // };
};

__global__ void x_kernel(float* M, float* x1, float* x_final, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x*2;

    while (idx < N) {

        float4 *x_final4 = reinterpret_cast<float4*>(&(x_final[idx*4]));
        float2 *M6_2 = reinterpret_cast<float2*>(&(M[idx*6]));
        float4 *x1_4 = reinterpret_cast<float4*>(&(x1[idx*2]));

        float4 temp_final1;
        float4 temp_final2;
        float2 temp_M6 = M6_2[0];
        float2 temp_M12 = M6_2[3];
        float4 temp_x1 = x1_4[0];

        float quotient1 = temp_M6.y/temp_M6.x;
        float quotient2 = temp_M12.y/temp_M12.x;

        temp_final1.x = quotient1 + temp_x1.x;
        temp_final1.y = temp_final1.x;
        temp_final1.z = quotient1 + temp_x1.y;
        temp_final1.w = temp_final1.z;
        temp_final2.x = quotient2 + temp_x1.z;
        temp_final2.y = temp_final2.x;
        temp_final2.z = quotient2 + temp_x1.w;
        temp_final2.w = temp_final2.z;

        x_final4[0] = temp_final1;
        x_final4[1] = temp_final2;
        
        // printf("[thread %d] temp_x1: %f %f %f %f \n" , idx, temp_x1.x, temp_x1.y, temp_x1.z, temp_x1.w);
        // printf("[thread %d] quotient: %f \n",idx, quotient2);
        // printf("[thread %d] temp_final2: %f %f %f %f \n" , idx, temp_final2.x, temp_final2.y, temp_final2.z, temp_final2.w);
        // x_final[4*idx] = M[6*idx+1]/M[6*idx] + x1[2*idx];
        // x_final[4*idx+1] = M[6*idx+1]/M[6*idx] + x1[2*idx];
        // x_final[4*idx+2] = M[6*idx+1]/M[6*idx] + x1[2*idx+1];
        // x_final[4*idx+3] = M[6*idx+1]/M[6*idx] + x1[2*idx+1];
        idx += 2*blockDim.x;
    };
};

__global__ void y_kernel(float* M, float* nu, float* x2, float* y_final, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x*2;

    while (idx < N) {

        float4 *y_final_4 = reinterpret_cast<float4*>(&(y_final[idx*4]));
        float2 *M6_2 = reinterpret_cast<float2*>(&(M[idx*6]));
        float4 *x2_4 = reinterpret_cast<float4*>(&(x2[idx*2]));
        float4 *nu_4 = reinterpret_cast<float4*>(&(nu[idx*2]));

        float4 temp_final1;
        float4 temp_final2;
        float2 temp_M6_21 = M6_2[0];
        float2 temp_M6_22 = M6_2[1];
        float2 temp_M12_21 = M6_2[3];
        float2 temp_M12_22 = M6_2[4];
        float4 temp_x2 = x2_4[0];
        float4 temp_nu = nu_4[0];

        float quotient1 = temp_M6_22.x/temp_M6_21.x;
        float quotient2 = temp_M12_22.x/temp_M12_21.x;

        temp_final1.x = quotient1 + temp_nu.x + temp_x2.x;
        // printf("temp_final1.x: %f\n", temp_final1.x);
        temp_final1.y = quotient1 + temp_nu.x + temp_x2.y;
        temp_final1.z = quotient1 + temp_nu.y + temp_x2.x;
        temp_final1.w = quotient1 + temp_nu.y + temp_x2.y;

        temp_final2.x = quotient2 + temp_nu.z + temp_x2.z;
        temp_final2.y = quotient2 + temp_nu.z + temp_x2.w;
        temp_final2.z = quotient2 + temp_nu.w + temp_x2.z;
        temp_final2.w = quotient2 + temp_nu.w + temp_x2.w;

        y_final_4[0] = temp_final1;
        y_final_4[1] = temp_final2;
        
        // printf("[thread %d] M6: %f %f %f %f\n" , idx, temp_M6_21.x, temp_M6_21.y, temp_M6_22.x, temp_M6_22.y);
        // printf("[thread %d] M12: %f %f %f %f\n" , idx, M[(idx+1)*6], M[(idx+1)*6+1], M[(idx+1)*6+2], M[(idx+1)*6+3]);
        // printf("[thread %d] temp_x2: %f %f %f %f \n" , idx, temp_x2.x, temp_x2.y, temp_x2.z, temp_x2.w);
        // printf("[thread %d] temp_nu: %f %f %f %f \n" , idx, temp_nu.x, temp_nu.y, temp_nu.z, temp_nu.w);
        // printf("[thread %d] quotient: %f \n",idx, quotient2);
        // printf("[thread %d] temp_final1: %f %f %f %f \n" , idx, temp_final1.x, temp_final1.y, temp_final1.z, temp_final1.w);
        idx += 2*blockDim.x;
    };
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // while (idx < N) {
    //     y_final[4*idx] = M[6*idx+2]/M[6*idx] + nu[2*idx] + x2[2*idx];
    //     y_final[4*idx+1] = M[6*idx+2]/M[6*idx] + nu[2*idx] + x2[2*idx+1];
    //     y_final[4*idx+2] = M[6*idx+2]/M[6*idx] + nu[2*idx+1] + x2[2*idx];
    //     y_final[4*idx+3] = M[6*idx+2]/M[6*idx] + nu[2*idx+1] + x2[2*idx+1];
    //     idx += blockDim.x;
    // };
};

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
    float *c20, *c11, *c02; 
    float *M_inter, *w_inter_1, *w_inter_2, *x_inter_1, *x_inter_2;
    float *nu, *mu;
    float *w_final_gpu, *x_final_gpu, *y_final_gpu;

    //// allocate device memory 
    // input
    gpuErrchk(cudaMalloc(&moments_gpu, sizeof(float)*num_moments*6));
    // Central moments
    gpuErrchk(cudaMalloc(&c20, sizeof(float)*num_moments));
    gpuErrchk(cudaMalloc(&c11, sizeof(float)*num_moments));
    gpuErrchk(cudaMalloc(&c02, sizeof(float)*num_moments));
    // intermediate M, w, x as input and output of HyQMOM2
    gpuErrchk(cudaMalloc(&M_inter, sizeof(float)*num_moments*3));
    gpuErrchk(cudaMalloc(&w_inter_1, sizeof(float)*num_moments*2));
    gpuErrchk(cudaMalloc(&w_inter_2, sizeof(float)*num_moments*2));
    gpuErrchk(cudaMalloc(&x_inter_1, sizeof(float)*num_moments*2));
    gpuErrchk(cudaMalloc(&x_inter_2, sizeof(float)*num_moments*2));
    // intermediate values 
    gpuErrchk(cudaMalloc(&nu, sizeof(float)*num_moments*2));
    gpuErrchk(cudaMalloc(&mu, sizeof(float)*num_moments));
    // final weight, abscissas: 
    gpuErrchk(cudaMalloc(&w_final_gpu, sizeof(float)*num_moments*4));
    gpuErrchk(cudaMalloc(&x_final_gpu, sizeof(float)*num_moments*4));
    cudaMalloc(&y_final_gpu, sizeof(float)*num_moments*4);  
    //copy input from host to device 
    gpuErrchk(cudaMemcpy(moments_gpu, moments, 
                    sizeof(float)*num_moments*6, cudaMemcpyHostToDevice)
    );


    // thread block is set to be 1D,
    int num_threads = 1024;
    int num_blocks = 1;

    // set up three streams for concurrent kernels
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);

    //// Calculating 
    printf("[CUDA] starting calculation. Timer on ... \n");
    cudaEventRecord(start); //start the timer
    cudaProfilerStart();

    // Central moments set_M_kernel
    c11_kernel<<<num_blocks, num_threads, 0, stream1>>>(moments_gpu, c11, num_moments);
    c20_kernel<<<num_blocks, num_threads, 0, stream2>>>(moments_gpu, c20, num_moments);
    c02_kernel<<<num_blocks, num_threads, 0, stream3>>>(moments_gpu, c02, num_moments);
    init_M<<<num_blocks, num_threads, 0, stream3>>>(c02, M_inter, num_moments);

    hyqmom2_kernel<<<num_blocks, num_threads, 0, stream3>>>(M_inter, w_inter_1, x_inter_1, num_moments);
    nu_kernel<<<num_blocks, num_threads, 0, stream3>>>(c11, c20, x_inter_1, nu, num_moments);
    mu_kernel2<<<num_blocks, num_threads, 0, stream2>>>(c02, c11, c20, mu, num_moments);
    init_M<<<num_blocks, num_threads, 0, stream2>>>(mu, M_inter, num_moments);

    // second hyqmom2
    hyqmom2_kernel<<<num_blocks, num_threads, 0, stream2>>>(M_inter, w_inter_2, x_inter_2, num_moments);

    // final results
    cudaStreamSynchronize(stream3);
    weight_kernel<<<num_blocks, num_threads, 0, stream2>>>(moments_gpu, w_inter_1, w_inter_2, w_final_gpu, num_moments);
    x_kernel<<<num_blocks, num_threads, 0, stream3>>>(moments_gpu, x_inter_1, x_final_gpu, num_moments);
    y_kernel<<<num_blocks, num_threads, 0, stream1>>>(moments_gpu, nu, x_inter_2, y_final_gpu, num_moments);
    cudaProfilerStop();
    cudaEventRecord(stop); //stop the timer
    printf("[CUDA] Finished calculation. Timer off... \n");

    // copy result from device to host 
    gpuErrchk(cudaMemcpyAsync(wout, w_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost, stream1));
    gpuErrchk(cudaMemcpyAsync(xout, x_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost, stream2));
    gpuErrchk(cudaMemcpyAsync(yout, y_final_gpu, sizeof(float)*num_moments*4, cudaMemcpyDeviceToHost, stream3));
    // --TODO-- verify the result somehow? 

    float calc_duration; 
    cudaEventElapsedTime(&calc_duration, start, stop);

    cudaFree(moments_gpu);
    cudaFree(c20);
    cudaFree(c02);
    cudaFree(c11);
    cudaFree(M_inter);
    cudaFree(w_inter_1);
    cudaFree(w_inter_2);
    cudaFree(x_inter_1);
    cudaFree(x_inter_2);
    cudaFree(mu);
    cudaFree(nu);
    cudaFree(w_final_gpu);
    cudaFree(x_final_gpu);
    cudaFree(y_final_gpu);
    return calc_duration;
}