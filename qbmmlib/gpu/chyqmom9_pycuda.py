import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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
     __global__ void float_value_set(float *addr, float value, int size) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int idx = tIdx; idx < size; idx+=blockDim.x*gridDim.x) {
            addr[idx] = value;
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

class cudaMemcpy2D:
    def __init__(self):
        self.copy2D = cuda.Memcpy2D()

    def copy_htod(self, dst, d_stride,
                src, s_stride, 
                width, height, stream=None):

        self.copy2D.set_src_host(src)
        self.copy2D.src_x_in_bytes = s_stride
        self.copy2D.set_dst_device(dst)
        self.copy2D.dst_x_in_bytes = d_stride
        self.copy2D.width_in_bytes = width
        self.copy2D.height = height
        self.copy2D(stream)
    
    def copy_dtoh(self, dst, d_stride,
                src, s_stride, 
                width, height, stream=None):

        self.copy2D.set_src_device(src)
        self.copy2D.src_x_in_bytes = s_stride
        self.copy2D.set_dst_host(dst)
        self.copy2D.dst_x_in_bytes = d_stride
        self.copy2D.width_in_bytes = width
        self.copy2D.height = height
        self.copy2D(stream)

    def copy_dtod(self, dst, d_stride,
                src, s_stride, 
                width, height, stream=None):

        self.copy2D.set_src_device(src)
        self.copy2D.src_x_in_bytes = s_stride
        self.copy2D.set_dst_device(dst)
        self.copy2D.dst_x_in_bytes = d_stride
        self.copy2D.width_in_bytes = width
        self.copy2D.height = height
        self.copy2D(stream)

def chyqmom9_cuda(
    moments: np.ndarray, 
    size: int, 
    w: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray,
    batch_size: int):

    mem_d_size_in_byte = np.ones(size).astype(np.float32).nbytes
    sizeof_float = np.dtype(np.float32).itemsize
    size = np.int32(size)

    # allocate memory on device
    moments_d = cuda.mem_alloc(moments.nbytes)
    w_out_d = cuda.mem_alloc(mem_d_size_in_byte * 9)
    x_out_d = cuda.mem_alloc(mem_d_size_in_byte * 9)
    y_out_d = cuda.mem_alloc(mem_d_size_in_byte * 9)

    c_moments = cuda.mem_alloc(mem_d_size_in_byte * 7)
    mu = cuda.mem_alloc(mem_d_size_in_byte * 3)
    yf = cuda.mem_alloc(mem_d_size_in_byte * 3)

    m1 = cuda.mem_alloc(mem_d_size_in_byte * 5)
    x1 = cuda.mem_alloc(mem_d_size_in_byte * 3)
    w1 = cuda.mem_alloc(mem_d_size_in_byte * 3)
    x2 = cuda.mem_alloc(mem_d_size_in_byte * 3)
    w2 = cuda.mem_alloc(mem_d_size_in_byte * 3)

    # register host memory as page-locked to enable Asych mem transfer
    cuda.register_host_memory(moments, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(w, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(x, cuda.mem_host_register_flags.PORTABLE)
    cuda.register_host_memory(y, cuda.mem_host_register_flags.PORTABLE)

    # Allocate 3 concurrent streams to each batch
    num_stream = batch_size * 3
    streams = []
    for i in range(num_stream):
        streams.append(cuda.Stream())

    blockSize = (1024, 1, 1)
    GridSize = (size + blockSize[0] - 1) / blockSize[0];
    GridSize = (GridSize, 1, 1)

    # timers 
    event_start = cuda.Event()
    event_stop = cuda.Event()

    size_per_batch = np.int32(np.ceil(size/batch_size))

    # initialize kernels
    c_kernel = CHYQMOM9.get_function('chyqmom9_cmoments')
    float_value_set = CHYQMOM9.get_function('float_value_set');
    chyqmom9_mu_yf = CHYQMOM9.get_function('chyqmom9_mu_yf')
    chyqmom9_wout = CHYQMOM9.get_function('chyqmom9_wout')
    chyqmom9_xout = CHYQMOM9.get_function('chyqmom9_xout')
    chyqmom9_yout = CHYQMOM9.get_function('chyqmom9_yout')

    copy2D = cudaMemcpy2D()
    hyqmom3 = hyqmom.Hyqmom.hyqmom3


    event_start.record()
    for i in np.arange(0, num_stream, 3):
        loc = (i/3) * size_per_batch
        copy2D.copy_htod(int(moments_d) + loc, size*sizeof_float,
                int(moments) + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 10,
                streams=streams[i])
    
        c_kernel(int(moments_d) + loc, int(c_moments) + loc, size_per_batch, size,
                block=BlockSize, grid=GridSize, stream=streams[i])

        float_value_set(int(m1) + loc, np.float32(1), size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i+1])
        float_value_set(int(m1) + loc, np.float32(0), size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i+2])
        cuda.memcpy_dtod_async(int(mi) + 3*size + loc, 
                int(c_moments) + loc, size_per_batch*sizeof_float, 
                stream=streams[i])
        copy2D.copy_dtod(int(m1) + 3*size + loc, size*sizeof_float,
                int(c_moments) + 4*size + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 2,
                streams=streams[i])
        
        hyqmom3(int(m1) + loc, int(x1) + loc, int(w1) + loc, size_per_batch, size,
                block=BlockSize, grid=GridSize, stream=streams[i+2])
        chyqmom9_mu_yf(int(c_moments) + loc, 
                int(x1) + loc, int(w1) + loc, 
                int(yf) + loc, int(mu) + loc, size_per_batch, size,
                block=BlockSize, grid=GridSize, stream=streams[i+2])
        
        float_value_set(int(m1) + loc, np.float32(1), size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i+1])
        float_value_set(int(m1) + size + loc, np.float32(0), size_per_batch,
                block=BlockSize, grid=GridSize, stream=streams[i+2])
        copy2D.copy_dtod(int(m1)+ 2*size + loc, size*sizeof_float,
                int(mu) + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 3,
                streams=streams[i])

        hyqmom3(int(m1) + loc, int(x2) + loc, int(w2) + loc, size_per_batch, size,
                block=BlockSize, grid=GridSize, stream=streams[i+2])

        chyqmom9_wout(int(moments_d) + loc, int(w1) + loc, 
                int(w2) + loc, int(w_out_d) + loc, size_per_batch, size,
                block=BlockSize, grid=GridSize, stream=streams[i])
        copy2D.copy_dtoh(int(w) + loc, size*sizeof_float,
                int(w_out_d) + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 9,
                streams=streams[i])

        chyqmom9_xout(int(moments_d) + loc, int(x1) + loc, 
                    int(x_out_d) + loc, size_per_batch, size,
                    block=BlockSize, grid=GridSize, stream=streams[i])
        copy2D.copy_dtoh(int(x) + loc, size*sizeof_float,
                int(x_out_d) + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 9,
                streams=streams[i+1])
        
        chyqmom9_yout(int(moments_d) + loc, int(x2) + loc, 
                    int(yf) + loc, int(y_out_d) + loc, size_per_batch, size,
                    block=BlockSize, grid=GridSize, stream=streams[i+2])
        copy2D.copy_dtoh(int(y) + loc, size*sizeof_float,
                int(y_out_d) + loc, size*sizeof_float,
                size_per_batch*sizeof_float, 9,
                streams=streams[i+2])
    
    event_stop.record()
    calc_time = event_stop.time_since(event_start)
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
    num_moments = 2

    moments = init_moment_10(num_moments)
    # flatten to 1d array 
    moments = moments.flatten()
    print(moments)

    # outputs 
    w = np.zeros((num_moments * 9, 1), dtype=np.float32)
    x = np.zeros((num_moments * 9, 1), dtype=np.float32)
    y = np.zeros((num_moments * 9, 1), dtype=np.float32)

    time = chyqmom9_cuda(moments, num_moments, w, x, y, 1)
