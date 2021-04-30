import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import time

import qbmmlib.gpu.qbmm_tests as qbmm

C_KERNEL = SourceModule('''
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
    }
''')

INTER_KERNEL = SourceModule('''

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
''')

HYQMOM2_KERNEL = SourceModule('''
    __global__ void hyqmom2_kernel(float* M, float* w, float* x, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            float C2 = ((M[3*idx] * M[3*idx+2]) - (M[3*idx+1] * M[3*idx+1])) 
                        / (M[3*idx] * M[3*idx]);
            w[2*idx] = M[3*idx]/2;
            w[2*idx+1] = M[3*idx]/2;
            x[2*idx] = (M[3*idx+1]/M[3*idx]) - sqrt(C2);
            x[2*idx+1] = (M[3*idx+1]/M[3*idx]) + sqrt(C2);
            idx += blockDim.x;
        };
    };
''')

FINAL_KERNEL = SourceModule('''
    __global__ void weight_kernel(float* M, float* w1, float* w2, float* w_final, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            w_final[4*idx] = M[6*idx] * w1[2*idx] * w2[2*idx];
            w_final[4*idx+1] = M[6*idx] * w1[2*idx] * w2[2*idx+1];
            w_final[4*idx+2] = M[6*idx] * w1[2*idx+1] * w2[2*idx];
            w_final[4*idx+3] = M[6*idx] * w1[2*idx+1] * w2[2*idx+1];
            idx += blockDim.x;
        };
    };

    __global__ void x_kernel(float* M, float* x1, float* x_final, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            x_final[4*idx] = M[6*idx+1]/M[6*idx] + x1[2*idx];
            x_final[4*idx+1] = M[6*idx+1]/M[6*idx] + x1[2*idx];
            x_final[4*idx+2] = M[6*idx+1]/M[6*idx] + x1[2*idx+1];
            x_final[4*idx+3] = M[6*idx+1]/M[6*idx] + x1[2*idx+1];
            idx += blockDim.x;
        };
    };

    __global__ void y_kernel(float* M, float* nu, float* x2, float* y_final, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            y_final[4*idx] = M[6*idx+2]/M[6*idx] + nu[2*idx] + x2[2*idx];
            y_final[4*idx+1] = M[6*idx+2]/M[6*idx] + nu[2*idx] + x2[2*idx+1];
            y_final[4*idx+2] = M[6*idx+2]/M[6*idx] + nu[2*idx+1] + x2[2*idx];
            y_final[4*idx+3] = M[6*idx+2]/M[6*idx] + nu[2*idx+1] + x2[2*idx+1];
            idx += blockDim.x;
        };
    };

''') 

class TestGPU4Node:

    def __init__(self, batch_size: int, block_size: int) -> None:
        self.batch_size = np.int32(batch_size)
        self.block_size = (block_size, 1, 1)
        self.cpu_bench = qbmm.TestInversion()
        self.input_moments = self.cpu_bench.init_batch_input(batch_size)
        print(self.input_moments[0])
        self.input_moments = self.input_moments.astype(np.float32)

        self.w_final = np.empty(self.batch_size*4).astype(np.float32)
        self.x_final = np.empty(self.batch_size*4).astype(np.float32)
        self.y_final = np.empty(self.batch_size*4).astype(np.float32)

    def setup_memory(self) -> None:
        '''
        Allocate and copy memory on device
        '''
        # input
        self.input_moments_gpu = cuda.mem_alloc(self.input_moments.nbytes)
        cuda.memcpy_htod(self.input_moments_gpu, self.input_moments)

        c_mem_size = np.ones(self.batch_size).astype(np.float32).nbytes

        # Central moments
        self.c20 = cuda.mem_alloc(c_mem_size)
        self.c11 = cuda.mem_alloc(c_mem_size)
        self.c02 = cuda.mem_alloc(c_mem_size)
        
        # intermediate M, w, x as input and output of HyQMOM2
        self.M_inter = cuda.mem_alloc(c_mem_size * 3)
        self.w_inter_1 = cuda.mem_alloc(c_mem_size * 2)
        self.w_inter_2 = cuda.mem_alloc(c_mem_size * 2)
        self.x_inter_1 = cuda.mem_alloc(c_mem_size * 2)
        self.x_inter_2 = cuda.mem_alloc(c_mem_size * 2)

        # intermediate values 
        self.nu = cuda.mem_alloc(c_mem_size * 2)
        self.mu = cuda.mem_alloc(c_mem_size)

        # final weight, abscissas: 
        self.w_final_gpu = cuda.mem_alloc(self.w_final.nbytes)
        self.x_final_gpu = cuda.mem_alloc(self.x_final.nbytes)
        self.y_final_gpu = cuda.mem_alloc(self.y_final.nbytes)

    def setup_kernel(self) -> None:
        '''
        Initialize GPU kernels
        '''
        self.c20_kernel = C_KERNEL.get_function('c20_kernel')
        self.c11_kernel = C_KERNEL.get_function('c11_kernel')
        self.c02_kernel = C_KERNEL.get_function('c02_kernel')

        self.set_M_kernel = INTER_KERNEL.get_function('init_M')
        self.hyqmom2_kernel = HYQMOM2_KERNEL.get_function('hyqmom2_kernel')

        self.nu_kernel = INTER_KERNEL.get_function('nu_kernel')
        self.mu_kernel = INTER_KERNEL.get_function('mu_kernel')

        self.weight_kernel = FINAL_KERNEL.get_function('weight_kernel')
        self.x_kernel = FINAL_KERNEL.get_function('x_kernel')
        self.y_kernel = FINAL_KERNEL.get_function('y_kernel')


    def compute_gpu(self) -> None:
        '''
        Run chyqmom4 on predefined inputs
        '''

        # Central moments
        self.c20_kernel(self.input_moments_gpu, self.c20, 
                        self.batch_size, block=self.block_size)
        self.c11_kernel(self.input_moments_gpu, self.c11, 
                        self.batch_size, block=self.block_size)
        self.c02_kernel(self.input_moments_gpu, self.c02, 
                        self.batch_size, block=self.block_size)

        # first hyqmom2
        self.set_M_kernel(self.c02, self.M_inter, 
                        self.batch_size, block=self.block_size)
        self.hyqmom2_kernel(self.M_inter, self.w_inter_1, self.x_inter_1, 
                        self.batch_size, block=self.block_size)

        # intermediate vaues
        self.nu_kernel(self.c11, self.c20, self.x_inter_1, self.nu, 
                        self.batch_size, block=self.block_size)
        self.mu_kernel(self.c02, self.nu, self.w_inter_1, self.mu, 
                        self.batch_size, block=self.block_size)

        # second hyqmom2
        self.set_M_kernel(self.mu, self.M_inter, 
                        self.batch_size, block = self.block_size)
        self.hyqmom2_kernel(self.M_inter, self.w_inter_2, self.x_inter_2, 
                        self.batch_size, block=self.block_size)

        # final results
        self.weight_kernel(self.input_moments_gpu, self.w_inter_1, self.w_inter_2, 
                        self.w_final_gpu, self.batch_size, block=self.block_size)
        self.x_kernel(self.input_moments_gpu, self.x_inter_1, self.x_final_gpu, 
                        self.batch_size, block=self.block_size)
        self.y_kernel(self.input_moments_gpu, self.nu, self.x_inter_2, 
                        self.y_final_gpu, self.batch_size, block=self.block_size)
        
    def compute_cpu(self) -> None:
        '''
        Make the same calculation without GPU
        '''
        result = self.cpu_bench.compute_batch(
                        self.input_moments, self.batch_size)
        self.w_cpu, self.x_cpu, self.y_cpu = result

    def verify(self) -> None:
        '''
        Verify that the GPU result is correct
        '''
        # copy result back to host
        cuda.memcpy_dtoh(self.w_final, self.w_final_gpu)
        cuda.memcpy_dtoh(self.x_final, self.x_final_gpu)
        cuda.memcpy_dtoh(self.y_final, self.y_final_gpu)

        ### verify results
        self.w_final = np.reshape(self.w_final, (-1, 4))
        self.x_final = np.reshape(self.x_final, (-1, 4))
        self.y_final = np.reshape(self.y_final, (-1, 4))
        print(self.w_final)
        w_cpu, x_cpu, y_cpu = self.cpu_bench.compute_batch(
                                        self.input_moments, self.batch_size)
        assert(np.all(np.abs(self.w_final - self.w_cpu) <= 1e-7))
        assert(np.all(np.abs(self.x_final - self.x_cpu) <= 1e-7))
        assert(np.all(np.abs(self.y_final - self.y_cpu) <= 1e-7))


if __name__ == "__main__":
    print("Initializing ...")
    N = int(10000000)
    GPU = TestGPU4Node(N, 1024)
    print("Initialization finished")
    
    start = pycuda.driver.Event()
    done = pycuda.driver.Event()

    gpu_begin = time.perf_counter()
    GPU.setup_memory()
    GPU.setup_kernel()

    start.record()
    gpu_init_end = time.perf_counter()
    pycuda.driver.start_profiler()
    GPU.compute_gpu()

    done.record()
    done.synchronize()
    time_by_event = start.time_till(done)

    gpu_end = time.perf_counter()
    pycuda.driver.stop_profiler()

    # GPU.compute_cpu()
    # cpu_end = time.perf_counter()
    print("calculation finished, verifying results ...")
    # GPU.verify()

    gpu_init_time = (gpu_init_end - gpu_begin)
    gpu_compute_time = (gpu_end - gpu_init_end)
    # cpu_compute_time = (cpu_end - gpu_end)

    # print("CPU compute time: {:3e} s".format(cpu_compute_time/N))
    print("GPU total time:   ", ((gpu_compute_time + gpu_init_time))/N)
    print("GPU compute time: ", (gpu_compute_time/N))
    print("GPU init time     ", (gpu_init_time/N))
    print("Event time: ", time_by_event/N)

    