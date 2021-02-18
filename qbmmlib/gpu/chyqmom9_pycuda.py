import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import ctypes

import numpy as np
import time

import hyqmom_pycuda as hyqmom

CHYQMOM9 = SourceModule('''
    // a helper function for calculating nth moment 
    __device__ float sum_pow(float rho[], float yf[], float n, const int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += rho[i] * pow(yf[i], n); 
        }
        return sum;
    }

    // set a segment of memory to a specific value
    __global__ void float_value_set(float addr[], float value, const int size, const int loc) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            addr[idx + loc] = value;
            //printf("[%d]loc: %d, setting addr[%d] to %f \\n", idx, loc, loc+idx, value);
        }
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
        const int size, 
        const int stride)
    {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            // copy moments to local registers
            float mom[10], cmom[7];
            mom[0] = moments[idx];
            // normalize mom by mom[0];
            // mom[i] = mom[i]/mom[0] for i !=0
            for (int n=1; n<10; n++) {
                mom[n] = moments[n * stride + idx] / mom[0];
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
            c_moments[1*stride + idx] =cmom[1];
            c_moments[2*stride + idx] =cmom[2];
            c_moments[3*stride + idx] =cmom[3];
            c_moments[4*stride + idx] =cmom[4];
            c_moments[5*stride + idx] =cmom[5];
            c_moments[6*stride + idx] =cmom[6];

            // printf("[%d] c_moment[%d] = %f \\n", idx, idx, c_moments[idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 1*stride + idx, c_moments[1*stride + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 2*stride + idx, c_moments[2*stride + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 3*stride + idx, c_moments[3*stride + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 4*stride + idx, c_moments[4*stride + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 5*stride + idx, c_moments[5*stride + idx]);
            // printf("[%d] c_moment[%d] = %f \\n", idx, 6*stride + idx, c_moments[6*stride + idx]);

        }
    }

     __global__ void chyqmom9_mu_yf(
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

     __global__ void chyqmom9_wout(
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
                }
            }
        }
    }

     __global__ void chyqmom9_xout(
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

     __global__ void chyqmom9_yout(
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
''')

def chyqmom9_cuda(
    moments: np.ndarray, 
    size: int, 
    w: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray,
    batch_size: int):

    mem_d_size_in_byte = np.ones(size).astype(np.float32).nbytes
    sizeof_float = np.int32(np.dtype(np.float32).itemsize)
    size = np.int32(size)

    # register host memory as page-locked to enable Asych mem transfer
    cuda.register_host_memory(moments, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(w, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(x, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(y, cuda.mem_host_register_flags.PORTABLE)

    # Allocate 1 concurrent streams to each batch
    num_stream = batch_size
    streams = []
    for i in range(num_stream):
        streams.append(cuda.Stream())

    BlockSize = (256, 1, 1)
    GridSize = (size +BlockSize[0] - 1) /BlockSize[0];
    GridSize = (int(GridSize), 1, 1)

    # timers 
    event_start = cuda.Event()
    event_stop = cuda.Event()

    size_per_batch = np.int32(np.ceil(float(size)/batch_size))
    print("size_per_batch: ", size_per_batch)

    # initialize kernels
    c_kernel = CHYQMOM9.get_function('chyqmom9_cmoments')
    float_value_set = CHYQMOM9.get_function('float_value_set')
    float_array_set = CHYQMOM9.get_function('float_array_set')
    chyqmom9_mu_yf = CHYQMOM9.get_function('chyqmom9_mu_yf')
    chyqmom9_wout = CHYQMOM9.get_function('chyqmom9_wout')
    chyqmom9_xout = CHYQMOM9.get_function('chyqmom9_xout')
    chyqmom9_yout = CHYQMOM9.get_function('chyqmom9_yout')

    moments_d = []
    w_out_d = []
    x_out_d = []
    y_out_d = []

    c_moments = []
    mu = []
    yf = []

    m1 = []
    x1 = []
    w1 = []
    x2 = []
    w2 = []

    for i in range(batch_size):
        # allocate memory on device
        moments_d.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 10)))
        w_out_d.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 9)))
        x_out_d.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 9)))
        y_out_d.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 9)))

        c_moments.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 7)))
        mu.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))
        yf.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))

        m1.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 5)))
        x1.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))
        w1.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))
        x2.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))
        w2.append(cuda.mem_alloc(int(sizeof_float * size_per_batch * 3)))

    hyq = hyqmom.Hyqmom(BlockSize, GridSize)

    event_start.record()
    for i in np.arange(0, num_stream, 1):
        loc = np.int32((i) * size_per_batch)
        if loc + size_per_batch > size: 
            size_per_batch = size - loc

        this_moment = np.ascontiguousarray(moments[:, loc:loc+size_per_batch], dtype=np.float32)
        cuda.memcpy_htod_async(moments_d[i], this_moment,
                stream=streams[i])
    
        c_kernel(moments_d[i], c_moments[i], size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])

        float_value_set(m1[i], np.float32(1), size_per_batch, np.int32(0),
                block=BlockSize, grid=GridSize, stream=streams[i])
        float_value_set(m1[i], np.float32(0), size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])
        float_array_set(m1[i], c_moments[i], 
                np.int32(size_per_batch), np.int32(size_per_batch * 2), np.int32(0),
                block=BlockSize, grid=GridSize, stream=streams[i])
        float_array_set(m1[i], c_moments[i], 
                np.int32(size_per_batch * 2), np.int32(size_per_batch * 3), np.int32(size_per_batch * 4),
                block=BlockSize, grid=GridSize, stream=streams[i])

        
        hyq.hyqmom3(m1[i], x1[i], w1[i], size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])
        chyqmom9_mu_yf(c_moments[i], 
                x1[i], w1[i], 
                yf[i], mu[i], size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])
        float_array_set(m1[i], mu[i], 
                np.int32(size_per_batch * 3), np.int32(size_per_batch * 2), np.int32(0),
                block=BlockSize, grid=GridSize, stream=streams[i])

        hyq.hyqmom3(m1[i], x2[i], w2[i], size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])

        chyqmom9_wout(moments_d[i], w1[i], 
                w2[i], w_out_d[i], size_per_batch, size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i])
        this_w = np.ascontiguousarray(np.zeros_like(w[:, loc:loc+size_per_batch]))
        cuda.memcpy_dtoh_async(this_w, w_out_d[i], stream=streams[i])

        chyqmom9_xout(moments_d[i], x1[i], 
                    x_out_d[i], size_per_batch, size_per_batch,
                    block=BlockSize, grid=GridSize, stream=streams[i])
        this_x = np.ascontiguousarray(np.zeros_like(x[:, loc:loc+size_per_batch]))
        cuda.memcpy_dtoh_async(this_x, x_out_d[i], stream=streams[i])
        
        chyqmom9_yout(moments_d[i], x2[i], 
                    yf[i], y_out_d[i], size_per_batch, size_per_batch,
                    block=BlockSize, grid=GridSize, stream=streams[i])
        this_y = np.ascontiguousarray(np.zeros_like(y[:, loc:loc+size_per_batch]))
        cuda.memcpy_dtoh_async(this_y, y_out_d[i], stream=streams[i])

        w[:, loc:loc+size_per_batch] = this_w
        x[:, loc:loc+size_per_batch] = this_x
        y[:, loc:loc+size_per_batch] = this_y

    
    event_stop.record()
    event_stop.synchronize()
    calc_time = event_stop.time_since(event_start)
    
    print('### w')
    print(w)
    return calc_time


def init_moment_10(size: int):
    one_moment = np.asarray([1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603], 
                        dtype=np.float32)
    moments = np.zeros((10, size))
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments


if __name__ == '__main__':
    num_moments = 5

    moments = init_moment_10(num_moments)
    # flatten to 1d array 
    # moments = moments.flatten()

    # outputs 
    w = np.zeros((9, num_moments), dtype=np.float32)
    x = np.zeros((9, num_moments), dtype=np.float32)
    y = np.zeros((9, num_moments), dtype=np.float32)

    time = chyqmom9_cuda(moments, num_moments, w, x, y, 3)
    print("## Time")
    print(time)
