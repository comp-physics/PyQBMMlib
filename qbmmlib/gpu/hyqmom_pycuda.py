import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import time

HYQMOM = SourceModule('''
    __global__ void hyqmom3(float moments[], float x[], float w[], const int size, const int stride){
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx += blockDim.x*gridDim.x) {
            // copy moments to local registers
            float mom[5];
            mom[0] = moments[idx];
            for (int n = 1; n < 5; n++) {
                mom[n] = moments[n * stride + idx] / mom[0];
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
            x[stride + idx] = mom[1] + xps[1] * scale / sqrt(scales);
            x[2*stride + idx] = mom[1] + xps[2] * scale / sqrt(scales);
            //w 
            w[idx] = mom[0] * rho[0];
            w[stride + idx] = mom[0] * rho[1];
            w[2*stride + idx] = mom[0] * rho[2];
        }
    }

    __global__ void hyqmom2(float moments[], float x[], float w[], const int size, const int stride) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float moments_local[3];
        while (idx < size) {
            for (int i = 0; i < 3; i++) {
                moments_local[i] = moments[i * stride + idx];
            }
            float C2 = ((moments_local[0] * moments_local[2]) - (moments_local[1] * moments_local[1])) 
                / (moments_local[0] * moments_local[0]);
            for (int i=0; i<2; i++) {
                w[i*stride+idx] = moments_local[0]/2;
            }
            x[idx] = (moments_local[1]/moments_local[0]) - sqrt(C2);
            x[stride + idx] = (moments_local[1]/moments_local[0]) + sqrt(C2);
            idx+= blockDim.x*gridDim.x;
        }
    }
''')


class Hyqmom: 

    def __init__(self, block_size: int, grid_size: int) -> None:
        self.hyqmom2 = HYQMOM.get_function('hyqmom2')
        self.hyqmom3 = HYQMOM.get_function('hyqmom3')

        self.grid_size = (grid_size, 1, 1)
        self.block_size = (block_size, 1, 1)
    
    def hyqmom2(moments: np.ndarray, 
                x: np.ndarray, 
                w: np.ndarray, 
                size: int,
                stride: int):
        self.hyqmom2(moments, w, x, size, stride,
                     block=self.block_size,
                     grid=self.grid_size)
    
    def hyqmom3(moments: np.ndarray, 
                x: np.ndarray, 
                w: np.ndarray, 
                size: int,
                stride: int):
        self.hyqmom3(moments, w, x, size, stride,
                     block=self.block_size,
                     grid=self.grid_size)
    
