import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from inversion_vectorized import chyqmom27 as chyqmom27_cpu

import time
from numba import njit, config, set_num_threads, threading_layer

HELPER = '''
    // set a segment of memory to a specific value
    __global__ void float_value_set(float *addr, float value, int size, int offset) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            addr[idx + offset] = value;
            //printf("[%d], setting addr to %f \\n", idx, value);
        }
    }

    // set a segment of memory to a specific array
    __global__ void float_array_set(float *addr, float *value, int size, int loc_d, int loc_s) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {

            addr[idx + loc_d] = value[idx + loc_s];
            //printf("[%d]loc_s: %d, setting addr[%d] to %f \\n", idx, loc_s, loc_d+idx, value[idx]);
        }
    }
'''

# Hyqmom kernels
HYQMOM = '''
    __global__ void hyqmom3(float moments[], float x[], float w[], const int size){
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {
            // copy moments to local registers
            float mom[5];
            mom[0] = moments[idx];
            // printf("[tIdx %d] hyqmom3 mom[0] = %.9f\\n", idx, mom[0]);
            for (int n = 1; n < 5; n++) {
                mom[n] = moments[n * size + idx] / mom[0];
                // printf("[tIdx %d] hyqmom3 mom[%d] = %.9f\\n", idx, n, mom[n]);
            }
            // central moments
            float c_moments[3];
            c_moments[0] = mom[2] - mom[1]*mom[1];
            c_moments[1] = mom[3] - 3*mom[1]*mom[2] + 2*mom[1]*mom[1]*mom[1];
            c_moments[2] = mom[4] - 4*mom[1]*mom[2] + 6*mom[1]*mom[1]*mom[2] -
                            3*mom[1]*mom[1]*mom[1]*mom[1];
            
            float scale = sqrt(c_moments[0]);
            float q = c_moments[1]/scale/c_moments[0];
            float eta = c_moments[2]/c_moments[0]/c_moments[0];
            float eta_test = c_moments[2]/c_moments[0];

            // printf("[hyqmom2] c_moments[0] = %.9f, c_moments[1] = %.9f, c_moments[2] = %.9f \\n", c_moments[0], c_moments[1], c_moments[2]);
            // printf("[hyqmom2] scale = %.9f, q = %.9f, eta = %.9f eta_test = %.9f\\n",scale, q, eta, eta_test);
            
            // xps
            float xps[3]; 
            float sqrt_term = sqrt(4*eta - 3*q*q);
            xps[0] = (q - sqrt_term)/2.0;
            xps[1] = 0;
            xps[2] = (q + sqrt_term)/2.0;

            // rho 
            float rho[3], rho_sum;
            float prod = -xps[0] * xps[2];
            rho[0] = -1.0/sqrt_term/xps[0];
            rho[1] = 1.0 - 1.0/prod;
            rho[2] = 1.0/sqrt_term/xps[2];
            rho_sum = rho[0] + rho[1] + rho[2]; 

            float scales = 0;
            for (int n = 0; n < 3; n++) {
                scales += rho[n]/rho_sum * xps[n] * xps[n];
            }
            scales /= rho_sum;

            // x 
            x[idx] = mom[1] + xps[0] * scale / sqrt(scales);
            x[size + idx] = mom[1] + xps[1] * scale / sqrt(scales);
            x[2*size + idx] = mom[1] + xps[2] * scale / sqrt(scales);
            //w 
            w[idx] = mom[0] * rho[0];
            w[size + idx] = mom[0] * rho[1];
            w[2*size + idx] = mom[0] * rho[2];
        }
    }

    __global__ void hyqmom2(float moments[], float x[], float w[], const int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float moments_local[3];
        while (idx < size) {
            for (int i = 0; i < 3; i++) {
                moments_local[i] = moments[i * size + idx];
                //printf("[tIdx %d] hyqmom3 mom[%d] = %f\\n", idx, i, moments_local[i]);
            }
            float C2 = ((moments_local[0] * moments_local[2]) - (moments_local[1] * moments_local[1])) 
                / (moments_local[0] * moments_local[0]);
            for (int i=0; i<2; i++) {
                w[i*size+idx] = moments_local[0]/2;
            }
            x[idx] = (moments_local[1]/moments_local[0]) - sqrt(C2);
            x[size + idx] = (moments_local[1]/moments_local[0]) + sqrt(C2);
            //printf("[hyqmom2] x[%d] = %f, x[%d] = %f \\n", idx, x[idx], size+idx, x[size+idx]);
            idx+= blockDim.x*gridDim.x;
        }
    }
'''

# Chyqmom4 specific kernels
CHYQMOM4 = '''
    // a helper function for calculating nth moment 
    __device__ float sum_pow(float rho[], float yf[], float n, const int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += rho[i] * pow(yf[i], n); 
        }
        return sum;
    }

    __global__ void chyqmom4_cmoments(const float moments[], float c_moments[], const int size) {
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {
            float mom[6];
            mom[0] = moments[idx];
            // printf("[tIdx %d] mom[0] = %f\\n", idx, mom[0]);
            // normalize mom by mom[0];
            // mom[i] = mom[i]/mom[0] for i !=0
            for (int n=1; n<6; n++) {
                mom[n] = moments[n * size + idx] / mom[0];
                //printf("[tIdx %d] mom[%d] = %f\\n", idx, n, mom[n]);
            }

            c_moments[0*size + idx] = mom[3] - mom[1] * mom[1];
            c_moments[1*size + idx] = mom[4] - mom[1] * mom[2];
            c_moments[2*size + idx] = mom[5] - mom[2] * mom[2];

            //printf("[%d] c_moment[%d] = %f \\n", idx, idx, c_moments[idx]);
            //printf("[%d] c_moment[%d] = %f \\n", idx, 1*size + idx, c_moments[1*size + idx]);
            //printf("[%d] c_moment[%d] = %f \\n", idx, 2*size + idx, c_moments[2*size + idx]);

        };
    };

    __global__ void chyqmom4_mu_yf(
        const float c_moments[], 
        const float xp[], 
        const float rho[],
        float yf[], 
        float mu[], 
        const int size)
    {
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x*2;
        for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {

            float c_11 = c_moments[size + idx];
            float c_20 = c_moments[2*size + idx];
            float coef = c_11/c_20;

            float yf_local[2] = {
                coef * xp[0*size + idx],
                coef * xp[1*size + idx],
            };

            float rho_local[2] = {
                rho[idx],          
                rho[1*size + idx], 
            };
            float mu_avg = c_20 - sum_pow(rho_local, yf_local, 2, 3);
            yf[0*size + idx] = yf_local[0];
            yf[1*size + idx] = yf_local[1];
            //printf("mu: %f\\n", mu_avg);
            mu[idx] = mu_avg;
        };
    };

    __global__ void chyqmom4_wout(
        const float moments[], 
        const float r1[], 
        const float r2[], 
        float w[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float mom = moments[idx];
            for (int row = 0; row < 2; row ++) {
                for (int col = 0; col < 2; col ++) {
                    w[(2*row + col) * size + idx] = 
                        r1[row * size + idx] * r2[col* size + idx] * mom;
                    // printf("[tIdx %d] w[%d] = %f \\n", tIdx, (3*row + col) * size + idx, w[(3*row + col) * size + idx]);
                }
            }
        }
    }

    __global__ void chyqmom4_xout(
        float moments[], 
        float xp[],
        float x[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float x_local[2];
            float bx = moments[size + idx] / moments[idx];
            for (int n = 0; n < 2; n++) {
                x_local[n] = xp[n * size + idx];
            }
            for (int row = 0; row < 2; row ++) {
                float val = x_local[row] + bx;
                for (int col = 0; col < 2; col ++) {
                    x[(2*row + col) * size + idx] = val;
                    // printf("[tIdx %d] x[%d] = %f \\n", tIdx, (2*row + col) * size + idx, x[(2*row + col) * size + idx]);
                }
            }
        }
    }

    __global__ void chyqmom4_yout(
        float moments[], 
        float xp3[],
        float yf[],
        float y[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float x_local[2];
            float yf_local[2];
            
            for (int n = 0; n < 2; n++) {
                x_local[n] = xp3[n * size + idx];
                yf_local[n]= yf[n * size + idx];
            }
            float by = moments[2*size + idx] / moments[idx];

            for (int row = 0; row < 2; row ++) {
                for (int col = 0; col < 2; col ++) {
                    y[(2*row + col) * size + idx] = yf_local[row] + x_local[col] + by;
                }
            }
        }
    }
'''

# Chyqmom9 specific kernels
CHYQMOM9 = '''
    // a helper function for calculating nth moment 
    __device__ float sum_pow(float rho[], float yf[], float n, const int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += rho[i] * pow(yf[i], n); 
        }
        return sum;
    }
    // set a segment of memory to a specific array
    __global__ void float_array_set(float *addr, float *value, int size, int loc_d, int loc_s) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {

            addr[idx + loc_d] = value[idx + loc_s];
            // printf("[%d]loc_s: %d, setting addr[%d] to %f \\n", idx, loc_s, loc_d+idx, value[idx]);
        }
    }

     __global__ void chyqmom9_cmoments(
        const float moments[], 
        float c_moments[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            // copy moments to local registers
            float mom[10], cmom[7];
            mom[0] = moments[idx];
            // normalize mom by mom[0];
            // mom[i] = mom[i]/mom[0] for i !=0
            for (int n=1; n<10; n++) {
                mom[n] = moments[n * size + idx] / mom[0];
                // // printf("[tIdx %d] mom[%d] = %f\\n", idx, n, mom[n]);
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
            c_moments[1*size + idx] =cmom[1];
            c_moments[2*size + idx] =cmom[2];
            c_moments[3*size + idx] =cmom[3];
            c_moments[4*size + idx] =cmom[4];
            c_moments[5*size + idx] =cmom[5];
            c_moments[6*size + idx] =cmom[6];

            // printf("[%d] c_moment[%d] = %f \\n", idx, idx, c_moments[idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 1*size + idx, c_moments[1*size + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 2*size + idx, c_moments[2*size + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 3*size + idx, c_moments[3*size + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 4*size + idx, c_moments[4*size + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 5*size + idx, c_moments[5*size + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 6*size + idx, c_moments[6*size + idx]);

        }
    }

     __global__ void chyqmom9_mu_yf(
        const float c_moments[], 
        const float xp[], 
        const float rho[],
        float yf[], 
        float mu[], 
        const int size) 
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float c_local[5] = {
                c_moments[idx],           // c02
                c_moments[1*size + idx],  // c11
                c_moments[2*size + idx],  // c20
                c_moments[4*size + idx],  // c03
                c_moments[6*size + idx]   // c04
            };
            float mu_avg = c_local[2] - c_local[1]*c_local[1]/c_local[0];
            float rho_local[3] = {
                rho[idx],          
                rho[1*size + idx], 
                rho[2*size + idx]  
            };
            float yf_local[3] = {
                c_local[1] * xp[idx] / c_local[2],
                c_local[1] * xp[size + idx] / c_local[2],
                c_local[1] * xp[2*size + idx] / c_local[2]
            };
            yf[idx] = yf_local[0];
            yf[size + idx] = yf_local[1];
            yf[2*size + idx] = yf_local[2];

            // if mu > csmall
            float q = (c_local[3] - sum_pow(rho_local, yf_local, 3.0, 3)) / 
                        powf(mu_avg, (3.0 / 2.0));
            float eta = (c_local[4] - sum_pow(rho_local, yf_local, 4.0, 3) - 
                        6 * sum_pow(rho_local, yf_local, 2.0, 3) * mu_avg) / 
                        powf(mu_avg, 2.0);

            float mu3 = q * powf(mu_avg, 3/2);
            float mu4 = eta * mu_avg * mu_avg;

            mu[idx] = mu_avg;
            mu[size + idx] = mu3;

            mu[2*size + idx] = mu4;
        }
    }

     __global__ void chyqmom9_wout(
        float moments[], 
        float rho_1[], 
        float rho_2[], 
        float w[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float r1[3], r2[3];
            float mom = moments[idx];
            for (int n=0; n<3; n++) {
                r1[n] = rho_1[n * size + idx];
                r2[n] = rho_2[n * size + idx];
            }
            
            for (int row = 0; row < 3; row ++) {
                for (int col = 0; col < 3; col ++) {
                    w[(3*row + col) * size + idx] = r1[row] * r2[col] * mom;
                }
            }
        }
    }

     __global__ void chyqmom9_xout(
        float moments[], 
        float xp[],
        float x[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float x_local[3];
            float bx = moments[size + idx] / moments[idx];
            for (int n = 0; n < 3; n++) {
                x_local[n] = xp[n * size + idx];
            }
            for (int row = 0; row < 3; row ++) {
                float val = x_local[row] + bx;
                for (int col = 0; col < 3; col ++) {
                    x[(3*row + col) * size + idx] = val;
                }
            }
        }
    }

     __global__ void chyqmom9_yout(
        float moments[], 
        float xp3[],
        float yf[],
        float y[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float x_local[3];
            float yf_local[3];
            
            for (int n = 0; n < 3; n++) {
                x_local[n] = xp3[n * size + idx];
                yf_local[n]= yf[n * size + idx];
            }
            float by = moments[2*size + idx] / moments[idx];

            for (int row = 0; row < 3; row ++) {
                for (int col = 0; col < 3; col ++) {
                    y[(3*row + col) * size + idx] = yf_local[row] + x_local[col] + by;
                }
            }
        }
    }
'''

# Chyqmom27 specific kernels
CHYQMOM27 = '''
    // a helper function for calculating nth moment 
    __device__ float sum_pow(float rho[], float yf[], float n, const int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += rho[i] * powf(yf[i], n); 
        }
        return sum;
    }

    // set a segment of memory to a specific value
    __global__ void float_value_set(float addr[], float value, const int size, const int loc) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            addr[idx + loc] = value;
            // printf("[%d]loc: %d, setting addr[%d] to %f \\n", idx, loc, loc+idx, value);
        }
    }

    // set a segment of memory to a specific array
    __global__ void float_array_set(float *addr, float *value, int size, int loc_d, int loc_s) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {

            addr[idx + loc_d] = value[idx + loc_s];
            // printf("[%d]loc_s: %d, setting addr[%d] to %f \\n", idx, loc_s, loc_d+idx, value[idx + loc_s]);
        }
    }

    // print out the content of an array 
    __global__ void print_device(float* addr, int size) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            printf("[%d], %.9f \\n", idx, addr[idx]);
        }
    }

    __global__ void chyqmom27_set_m(
        float* m,
        float* c_moments,
        const int size) 
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            m[0*size + idx] = 1;
            m[1*size + idx] = 0;
            m[2*size + idx] = 0;
            m[3*size + idx] = c_moments[0*size + idx];
            m[4*size + idx] = c_moments[1*size + idx];
            m[5*size + idx] = c_moments[3*size + idx];
            m[6*size + idx] = c_moments[7*size + idx];
            m[7*size + idx] = c_moments[8*size + idx];
            m[8*size + idx] = c_moments[10*size + idx];
            m[9*size + idx] = c_moments[11*size + idx];
        }
    }

    __global__ void chyqmom27_cmoments(
        const float moments[], 
        float c_moments[],
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            // copy moments to local registers
            float mom[16];
            mom[0] = moments[idx];
            // normalize mom by mom[0];
            // mom[i] = mom[i]/mom[0] for i !=0
            for (int n=1; n<16; n++) {
                mom[n] = moments[n * size + idx] / mom[0];
                // printf("[tIdx %d] mom[%d] = %.9f\\n", idx, n, mom[n]);
            }
            
            //compute central moments
            c_moments[idx] = mom[4] - mom[1] * mom[1]; // c200 = d200 - bx ** 2
            c_moments[1*size + idx] = mom[5] - mom[1] * mom[2]; // c110 = d200 - bx * by
            c_moments[2*size + idx] = mom[6] - mom[1] * mom[3]; // c101 = d101 - bx * bz
            c_moments[3*size + idx] = mom[7] - mom[2] * mom[2]; // c020 = d020 - by ** 2
            c_moments[4*size + idx] = mom[8] - mom[2] * mom[3]; // c011 = d011 - by * bz
            c_moments[5*size + idx] = mom[9] - mom[3] * mom[3]; // c002 = d002 - bz ** 2

            // c300 = d300 - 3 * bx * d200 + 2 * bx ** 3
            c_moments[6*size + idx] = mom[10] - 3*mom[1]*mom[4] + 2*mom[1]*mom[1]*mom[1];
            // c030 = d030 - 3 * by * d020 + 2 * by ** 3
            c_moments[7*size + idx] = mom[11] - 3*mom[2]*mom[7] + 2*mom[2]*mom[2]*mom[2];
            // c003 = d003 - 3 * bz * d002 + 2 * bz ** 3
            c_moments[8*size + idx] = mom[12] - 3*mom[3]*mom[9] + 2*mom[3]*mom[3]*mom[3];

            // d400 - 4 * bx * d300 + 6 * bx ** 2 * d200 - 3 * bx ** 4
            c_moments[9*size + idx] = mom[13] - 4*mom[1]*mom[10] + 6*mom[1]*mom[1]*mom[4] -
            3*mom[1]*mom[1]*mom[1]*mom[1];
            // d040 - 4 * by * d030 + 6 * by ** 2 * d020 - 3 * by ** 4
            c_moments[10*size + idx] = mom[14] - 4*mom[2]*mom[11] + 6*mom[2]*mom[2]*mom[7] -
            3*mom[2]*mom[2]*mom[2]*mom[2];
            // d004 - 4 * bz * d003 + 6 * bz ** 2 * d002 - 3 * bz ** 4
            c_moments[11*size + idx] = mom[15] - 4*mom[3]*mom[12] + 6*mom[3]*mom[3]*mom[9] -
            3*mom[3]*mom[3]*mom[3]*mom[3];

            // printf("[%d] c_moment[%d] = %.9f \\n", idx, idx, c_moments[idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 1*size + idx, c_moments[1*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 2*size + idx, c_moments[2*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 3*size + idx, c_moments[3*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 4*size + idx, c_moments[4*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 5*size + idx, c_moments[5*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 6*size + idx, c_moments[6*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 7*size + idx, c_moments[7*size + idx]);
            // printf("[%d] c_moment[%d] = %.9f \\n", idx, 8*size + idx, c_moments[8*size + idx]);
            // printf("[%d] c_moment[%d] = %.18f \\n", idx, 9*size + idx, c_moments[9*size + idx]);
            // printf("[%d] c_moment[%d] = %.18f \\n", idx, 10*size + idx, c_moments[10*size + idx]);
            // printf("[%d] c_moment[%d] = %.18f \\n", idx, 11*size + idx, c_moments[11*size + idx]);
        }
    }

    __global__ void chyqmom27_rho_yf(
        float* c_moments, 
        float* y2,  // chyqmom9 output
        float* w2,  // chyqmom9 output 
        float* rho,
        float* yf, 
        float* yp,
        const int size) 
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float rho_local[9], yf_local[3];

            rho_local[0] = w2[idx] / (w2[idx] +  w2[1*size + idx] +  w2[2*size + idx]);
            rho_local[1] = w2[1*size + idx] / (w2[idx] +  w2[1*size + idx] +  w2[2*size + idx]);
            rho_local[2] = w2[2*size + idx] / (w2[idx] +  w2[1*size + idx] +  w2[2*size + idx]);
            rho_local[3] = w2[3*size + idx] / (w2[3*size + idx] +  w2[4*size + idx] +  w2[5*size + idx]);
            rho_local[4] = w2[4*size + idx] / (w2[3*size + idx] +  w2[4*size + idx] +  w2[5*size + idx]);
            rho_local[5] = w2[5*size + idx] / (w2[3*size + idx] +  w2[4*size + idx] +  w2[5*size + idx]);
            rho_local[6] = w2[6*size + idx] / (w2[6*size + idx] +  w2[7*size + idx] +  w2[8*size + idx]);
            rho_local[7] = w2[7*size + idx] / (w2[6*size + idx] +  w2[7*size + idx] +  w2[8*size + idx]);
            rho_local[8] = w2[8*size + idx] / (w2[6*size + idx] +  w2[7*size + idx] +  w2[8*size + idx]);

            yf_local[0] = rho[0] * y2[0*size + idx] + rho[1] * y2[1*size + idx] + rho[2] * y2[2*size + idx];
            yf_local[1] = rho[3] * y2[3*size + idx] + rho[4] * y2[4*size + idx] + rho[5] * y2[5*size + idx];
            yf_local[2] = rho[6] * y2[6*size + idx] + rho[7] * y2[7*size + idx] + rho[8] * y2[8*size + idx];

            yp[0*size + idx] = y2[0*size + idx] - yf_local[0];
            yp[1*size + idx] = y2[1*size + idx] - yf_local[0];
            yp[2*size + idx] = y2[2*size + idx] - yf_local[0];
            yp[3*size + idx] = y2[3*size + idx] - yf_local[1];
            yp[4*size + idx] = y2[4*size + idx] - yf_local[1];
            yp[5*size + idx] = y2[5*size + idx] - yf_local[1];
            yp[6*size + idx] = y2[6*size + idx] - yf_local[2];
            yp[7*size + idx] = y2[7*size + idx] - yf_local[2];
            yp[8*size + idx] = y2[8*size + idx] - yf_local[2];

            rho[0*size + idx] = rho_local[0];
            rho[1*size + idx] = rho_local[1];
            rho[2*size + idx] = rho_local[2];
            rho[3*size + idx] = rho_local[3];
            rho[4*size + idx] = rho_local[4];
            rho[5*size + idx] = rho_local[5];
            rho[6*size + idx] = rho_local[6];
            rho[7*size + idx] = rho_local[7];
            rho[8*size + idx] = rho_local[8];
        }
    }

    __global__ void chyqmom27_zf(
        float* c_moments,
        float* x1,
        float* zf,
        const int size) 
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float scale1 = sqrt(c_moments[0*size + idx]); // c200
            float scale2 = sqrt(c_moments[3*size + idx]); // c020
            float c101_s = c_moments[2*size + idx] / scale1;

            // Zf = b1 * Yc1, line 1052 in inversion.py
            zf[0*size + idx] = x1[idx] / scale1 * c101_s;
            zf[1*size + idx] = x1[1*size + idx] / scale1 * c101_s;
            zf[2*size + idx] = x1[2*size + idx] / scale1 * c101_s;
        }
    }    

    __global__ void chyqmom27_mu(
        float* c_moments, 
        float* rho,
        float* zf,
        float* mu,
        const int size) 
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float rho_local[3];
            float Zf_local[3];
            float mu_local[3];

            rho_local[0] = rho[idx];
            rho_local[1] = rho[1*size + idx];
            rho_local[2] = rho[2*size + idx];

            Zf_local[0] = zf[idx];
            Zf_local[1] = zf[1*size + idx];
            Zf_local[2] = zf[2*size + idx];

            // mu2 = c002 - sum(RAB * ZF ^ 2)
            float SUM002 = sum_pow(rho_local, Zf_local, 2, 3);
            mu_local[0] = c_moments[5*size + idx] - SUM002;

            float q = c_moments[8*size + idx] - sum_pow(rho_local, Zf_local, 3, 3);
            q /= powf(mu_local[0], 3/2);

            float eta = c_moments[11*size + idx] - (sum_pow(rho_local, Zf_local, 4, 3) + 6 * SUM002 * mu_local[0]);
            eta /= (mu_local[0] * mu_local[0]);

            mu[idx] = mu_local[0];
            mu[1*size + idx] = q * powf(mu_local[0], 3/2);
            mu[2*size + idx] = eta * mu_local[0] * mu_local[0];
        }
    }

    __global__ void chyqmom27_wout(
        float* moments,
        float* rho1, 
        float* rho2, 
        float* rho3,
        float* w,
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {

            for (int i = 0; i < 3; i++) {
                for (int j=0; j < 3; j++) {
                    for (int k=0; k < 3; k++) {
                        w[(i*9 + j*3 + k) * size + idx] = rho1[i] * rho2[i*3 + j] * rho3[k];
                    }
                }
            }
        }
    }

    __global__ void chyqmom27_xout(
        float* moments,
        float* x1, 
        float* x,
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float bx = moments[1] / moments[0];
            for (int i = 0; i < 3; i++) {
                for (int j=0; j < 3; j++) {
                    for (int k=0; k < 3; k++) {
                        x[(i*9 + j*3 + k) * size + idx] = x1[i] + bx;
                    }
                }
            }
        }
    }

    __global__ void chyqmom27_yout(
        float* moments,
        float* yf, 
        float* yp,
        float* y,
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float by = moments[2] / moments[0];
            for (int i = 0; i < 3; i++) {
                for (int j=0; j < 3; j++) {
                    for (int k=0; k < 3; k++) {
                        y[(i*9 + j*3 + k) * size + idx] = yf[i] + yp[i*3 + j] + by;
                    }
                }
            }
        }
    }

    __global__ void chyqmom27_zout(
        float* moments,
        float* zf, 
        float* zp,
        float* z,
        const int size)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            float bz = moments[3] / moments[0];
            for (int i = 0; i < 3; i++) {
                for (int j=0; j < 3; j++) {
                    for (int k=0; k < 3; k++) {
                        z[(i*9 + j*3 + k) * size + idx] = zf[i*3 + j] + zp[k] + bz;
                    }
                }
            }
        }
    }
'''

this_context = pycuda.autoinit.context


def init_moment_27(size: int):
    one_moment = np.asarray([1,    1,      1,      1, 
                             1.01, 1,      1,      1.01, 
                             1,    1.01,   1.03,   1.03,
                             1.03, 1.0603, 1.0603, 1.0603], 
                             dtype=np.float32)
    moments = cuda.aligned_zeros((16, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments

def chyqmom9(
    moments: cuda.DeviceAllocation, 
    size: int, 
    w: cuda.DeviceAllocation,
    x: cuda.DeviceAllocation, 
    y: cuda.DeviceAllocation):

    mem_d_size_in_byte = np.ones(size).astype(np.float32).nbytes
    sizeof_float = np.int32(np.dtype(np.float32).itemsize)
    size = np.int32(size)

    BlockSize = (256, 1, 1)
    GridSize = (size +BlockSize[0] - 1) /BlockSize[0];
    GridSize = (int(GridSize), 1, 1)

    # compile kernel 
    CHYQMOM9_KERNEL = SourceModule(CHYQMOM9)
    CHY27 = SourceModule(CHYQMOM27)
    HYQ = SourceModule(HYQMOM)
    HELP = SourceModule(HELPER)
    print_device = CHY27.get_function('print_device')
    c_kernel = CHYQMOM9_KERNEL.get_function('chyqmom9_cmoments')
    float_value_set = HELP.get_function('float_value_set')
    float_array_set = HELP.get_function('float_array_set')
    chyqmom9_mu_yf = CHYQMOM9_KERNEL.get_function('chyqmom9_mu_yf')
    chyqmom9_wout = CHYQMOM9_KERNEL.get_function('chyqmom9_wout')
    chyqmom9_xout = CHYQMOM9_KERNEL.get_function('chyqmom9_xout')
    chyqmom9_yout = CHYQMOM9_KERNEL.get_function('chyqmom9_yout')

    hyqmom3 = HYQ.get_function('hyqmom3')


    c_moments = cuda.mem_alloc(int(sizeof_float * size * 7))
    mu = cuda.mem_alloc(int(sizeof_float * size * 3))
    yf = cuda.mem_alloc(int(sizeof_float * size * 3))

    m1 = cuda.mem_alloc(int(sizeof_float * size * 5))
    float_value_set(m1, np.float32(1), size, np.int32(0),
        block=BlockSize, grid=GridSize)
    float_value_set(m1, np.float32(0), size, size,
            block=BlockSize, grid=GridSize)

    x1 = cuda.mem_alloc(int(sizeof_float * size * 3))
    w1 = cuda.mem_alloc(int(sizeof_float * size * 3))
    x2 = cuda.mem_alloc(int(sizeof_float * size * 3))
    w2 = cuda.mem_alloc(int(sizeof_float * size * 3))

    c_kernel(moments, c_moments, size,
                block=BlockSize, grid=GridSize)
            
    float_array_set(m1, c_moments, 
            np.int32(size), np.int32(size * 2), np.int32(0),
            block=BlockSize, grid=GridSize)

    float_array_set(m1, c_moments, 
            np.int32(size * 2), np.int32(size * 3), np.int32(size * 4),
            block=BlockSize, grid=GridSize)

    hyqmom3(m1, x1, w1, size,
            block=BlockSize, grid=GridSize)
    chyqmom9_mu_yf(c_moments, 
            x1, w1, 
            yf, mu, size,
            block=BlockSize, grid=GridSize)
            
    # this_context.synchronize()
    # print("CHYQMOM9- What is xq?")
    # print_device(x1, np.int32(3), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("CHYQMOM9- What is rho?")
    # print_device(w1, np.int32(3), block=BlockSize, grid=GridSize)
    
    float_array_set(m1, mu, 
            np.int32(size * 3), np.int32(size * 2), np.int32(0),
            block=BlockSize, grid=GridSize)
    hyqmom3(m1, x2, w2, size, size,
            block=BlockSize, grid=GridSize)
    
    chyqmom9_wout(moments, w1, 
                w2, w, size,
                block=BlockSize, grid=GridSize)
    
    chyqmom9_xout(moments, x1, 
                x, size,
                block=BlockSize, grid=GridSize)

    chyqmom9_yout(moments, x2, 
                yf, y, size,
                block=BlockSize, grid=GridSize)

    c_moments.free()
    mu.free()
    yf.free()
    m1.free()
    w1.free()
    x1.free()
    w2.free()
    x2.free()


def chyqmom27(
    moments: np.ndarray, 
    size: int):

    mem_d_size_in_byte = np.ones(size).astype(np.float32).nbytes
    sizeof_float = np.int32(np.dtype(np.float32).itemsize)
    size = np.int32(size)

    BlockSize = (256, 1, 1)
    GridSize = (size +BlockSize[0] - 1) /BlockSize[0];
    GridSize = (int(GridSize), 1, 1)

    # compile kernel
    HYQ = SourceModule(HYQMOM)
    CHY27 = SourceModule(CHYQMOM27)
    hyqmom3 = HYQ.get_function('hyqmom3')

    c_kernel = CHY27.get_function('chyqmom27_cmoments')
    chyqmom27_rho_yf = CHY27.get_function('chyqmom27_rho_yf')
    chyqmom27_zf = CHY27.get_function('chyqmom27_zf')
    chyqmom27_mu = CHY27.get_function('chyqmom27_mu')
    float_value_set = CHY27.get_function('float_value_set')
    float_array_set = CHY27.get_function('float_array_set')
    chyqmom27_set_m = CHY27.get_function('chyqmom27_set_m')
    print_device = CHY27.get_function('print_device')
    chyqmom27_wout = CHY27.get_function('chyqmom27_wout')
    chyqmom27_xout = CHY27.get_function('chyqmom27_xout')
    chyqmom27_yout = CHY27.get_function('chyqmom27_yout')
    chyqmom27_zout = CHY27.get_function('chyqmom27_zout')

    w = cuda.aligned_zeros((27, int(size)), dtype=np.float32)
    x = cuda.aligned_zeros((27, int(size)), dtype=np.float32)
    y = cuda.aligned_zeros((27, int(size)), dtype=np.float32)
    z = cuda.aligned_zeros((27, int(size)), dtype=np.float32)

    # Allocate memory 
    moments_device = cuda.mem_alloc(int(sizeof_float * size * 16))
    c_moments = cuda.mem_alloc(int(sizeof_float * size * 12))

    m = cuda.mem_alloc(int(sizeof_float * size * 10))
    float_value_set(m, np.float32(1), size, np.int32(0), block=BlockSize, grid=GridSize)
    float_value_set(m, np.float32(0), size, size, block=BlockSize, grid=GridSize)

    w1 = cuda.mem_alloc(int(sizeof_float * size * 3))
    x1 = cuda.mem_alloc(int(sizeof_float * size * 3))

    w2 = cuda.mem_alloc(int(sizeof_float * size * 9))
    x2 = cuda.mem_alloc(int(sizeof_float * size * 9))
    y2 = cuda.mem_alloc(int(sizeof_float * size * 9))

    rho = cuda.mem_alloc(int(sizeof_float * size * 9))
    yf = cuda.mem_alloc(int(sizeof_float * size * 3))
    yp = cuda.mem_alloc(int(sizeof_float * size * 9))
    zf = cuda.mem_alloc(int(sizeof_float * size * 3))

    w3 = cuda.mem_alloc(int(sizeof_float * size * 3))
    x3 = cuda.mem_alloc(int(sizeof_float * size * 3))

    mu = cuda.mem_alloc(int(sizeof_float * size * 3))

    w_dev = cuda.mem_alloc(int(sizeof_float * size * 27))
    x_dev = cuda.mem_alloc(int(sizeof_float * size * 27))
    y_dev = cuda.mem_alloc(int(sizeof_float * size * 27))
    z_dev = cuda.mem_alloc(int(sizeof_float * size * 27))

    cuda.memcpy_htod(moments_device, moments)
    # Is this faster? 

    time_before = cuda.Event()
    time_after = cuda.Event()

    time_before.record()

    c_kernel(moments_device, c_moments, size, block=BlockSize, grid=GridSize)
    float_array_set(m, c_moments, size, np.int32(2) * size, np.int32(0), block=BlockSize, grid=GridSize)
    float_array_set(m, c_moments, size, np.int32(3) * size, np.int32(6) * size, block=BlockSize, grid=GridSize)
    float_array_set(m, c_moments, size, np.int32(4) * size, np.int32(9) * size, block=BlockSize, grid=GridSize)

    # print("What is m1?")
    # print_device(m, np.int32(5), block=BlockSize, grid=GridSize)

    hyqmom3(m, x1, w1, size, block=BlockSize, grid=GridSize)

    # Is this faster? 
    chyqmom27_set_m(m, c_moments, size, block=BlockSize, grid=GridSize)

    # this_context.synchronize()
    # print_device(m, np.int32(10), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Entering CHYQMOM9")
    chyqmom9(m, size, w2, x2, y2)

    # this_context.synchronize()
    # print("What is w2?")
    # print_device(w2, np.int32(10), block=BlockSize, grid=GridSize)


    chyqmom27_rho_yf(c_moments, y2, w2, rho, yf, yp, size, block=BlockSize, grid=GridSize)
    chyqmom27_zf(c_moments, x1, zf, size, block=BlockSize, grid=GridSize) 
    chyqmom27_mu(c_moments, rho, zf, mu, size, block=BlockSize, grid=GridSize)

    float_array_set(m, mu, size, np.int32(2) * size, np.int32(0), block=BlockSize, grid=GridSize)
    float_array_set(m, mu, size, np.int32(3) * size, np.int32(1) * size, block=BlockSize, grid=GridSize)
    float_array_set(m, mu, size, np.int32(4) * size, np.int32(2) * size, block=BlockSize, grid=GridSize)
    hyqmom3(m, x3, w3, size, block=BlockSize, grid=GridSize)

    chyqmom27_wout(moments_device, w1, rho, w3, w_dev, size, block=BlockSize, grid=GridSize)
    chyqmom27_xout(moments_device, x1, x_dev, size, block=BlockSize, grid=GridSize)
    chyqmom27_yout(moments_device, yf, yp, y_dev, size, block=BlockSize, grid=GridSize)
    chyqmom27_zout(moments_device, zf, x3, z_dev, block=BlockSize, grid=GridSize)

    time_after.record()
    time_after.synchronize()
    elapsed_time =  time_after.time_since(time_before)
    
    cuda.memcpy_dtoh(w, w_dev)
    cuda.memcpy_dtoh(x, x_dev)
    cuda.memcpy_dtoh(y, y_dev)
    cuda.memcpy_dtoh(z, z_dev)

    # this_context.synchronize()
    # print("Entering rho")
    # print_device(rho, np.int32(9*2), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Entering mu")
    # print_device(mu, np.int32(3*2), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Entering w1")
    # print_device(w1, np.int32(3*2), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Entering rho")
    # print_device(rho, np.int32(9*2), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Entering w3")
    # print_device(w3, np.int32(3*2), block=BlockSize, grid=GridSize)
    # this_context.synchronize()
    # print("Final w_dev")
    # print_device(w_dev, np.int32(27*1), block=BlockSize, grid=GridSize)

    moments_device.free()
    c_moments.free()

    m.free()
    w1.free()
    x1.free()

    w2.free()
    x2.free()
    y2.free()

    rho.free()
    yf.free()
    yp.free()
    zf.free()

    w3.free()
    x3.free()

    mu.free()

    return elapsed_time, w_dev, x_dev, y_dev, z_dev

def time_script():
    res_file_name = 'chyqmom4_res_2.csv' # output data file name
    max_input_size_mag = 6             # max number of input point (power of 10)
    num_points = 200                   # number of runs collected 
    trial = 5                          # For each run, the number of trials run. 
    num_device = 1                     # number of GPUs used

    config.THREADING_LAYER = 'threadsafe'
    set_num_threads(12)                # numba: number of concurrent CPU threads 
    print("Threading layer chosen: %s" % threading_layer())
    
    ## Header: 
    #  [num input, cpu_result (ms), gpu_result (ms)] 
    result = np.zeros((num_points, 4))

    this_result_cpu = np.zeros(trial)
    this_result_gpu = np.zeros(trial)

    # generate a set of input data size, linear in log space between 1 and maximum 
    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
    # for idx, in_size in enumerate(np.linspace(1e5, 1e6, num_points)):
        result[idx, 0] = idx
        result[idx, 1] = int(in_size)

        this_moment = init_moment_27(int(in_size))
        # output from GPU

        for i in range(0, trial, 1):
            # GPU time
            try: 
                this_result_gpu[i], w, x,y,z = chyqmom27(this_moment, int(in_size))
            except: 
                pass
            # chyqmom27(this_moment, int(in_size))
            # numba time: 
            start_time = time.perf_counter()
            chyqmom27_cpu(this_moment.transpose(), int(in_size))
            stop_time = time.perf_counter()
            this_result_cpu[i] = (stop_time - start_time) * 1e3 #ms

            w.free()
            x.free()
            y.free()
            z.free()
        result[idx, 1] = np.min(this_result_cpu)
        result[idx, 2] = np.min(this_result_gpu)
        print("[{}/{}] running on {} inputs, CPU: {:4f}, GPU: {:4f}".format(
            idx, num_points, int(in_size), result[idx, 1], result[idx, 2]))

    np.savetxt(res_file_name, result, delimiter=',')
    
if __name__ == "__main__":
    time_script()
   
    # in_size = 5e5

    # for i in range(200):
    #     this_moment = init_moment_27(int(in_size))
    #     t1 = chyqmom27(this_moment, int(in_size))
    #     t2 = chyqmom27(this_moment, int(in_size))
    #     t3 = chyqmom27(this_moment, int(in_size))
    #     t4 = chyqmom27(this_moment, int(in_size))
    #     t5 = chyqmom27(this_moment, int(in_size))
    #     print(i, t1, t2, t3, t4, t5)

    # w, x, y, z = chyqmom27_cpu(this_moment.transpose(), int(in_size))
    # print(w)