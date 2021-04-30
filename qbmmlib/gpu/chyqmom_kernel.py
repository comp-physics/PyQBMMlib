## GPU kernel source code
import numpy as np

# size of 32bit float in bytes 
SIZEOF_FLOAT = np.int32(np.dtype(np.float32).itemsize)

# number of helper functions used by GPU kernels
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
            // printf("[tIdx %d] hyqmom3 mom[0] = %f\\n", idx, mom[0]);
            for (int n = 1; n < 5; n++) {
                mom[n] = moments[n * size + idx] / mom[0];
                //printf("[tIdx %d] hyqmom3 mom[%d] = %f\\n", idx, n, mom[n]);
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
                c_moments[idx],             // c02
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
