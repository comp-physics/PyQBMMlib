## basic example
# used to verify that pycuda is working properly by calling two simple kernels


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

# Check whether gpu threads are properly launched, and input argument properly passed.
# Each thread will simply print its thread index.
TEST = SourceModule('''
    __global__ void test_kernel(int N) {
        // 1D block thread indexing 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        printf("thread_idx: %d got integer: %d\\n", idx, N);
    }
''')

# Add two arrays element-wise.
# Each thread prints the array element index its in charge of adding, and the element sum
ADD = SourceModule('''
    __global__ void add(int N, float* a, float* b, float* c) {
        // 1D block thread indexing 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            c[idx] = a[idx] + b[idx];
            printf("thread_idx: %d c[idx]: %f\\n", idx, c[idx]);
            idx += blockDim.x * gridDim.x;
        }
    }  
    ''')

def simple_test(N: int) -> None:
    N = np.int32(N)
    test = TEST.get_function('test_kernel')
    test(N, block=(10, 10, 1))

def add_two_Identity(dim: int) -> None:
    dim = 10
    a = np.ones([dim, dim])
    b = np.ones([dim, dim])
    N = np.int32(dim * dim)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # cudaMalloc
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(a.nbytes)
    c_gpu = cuda.mem_alloc(a.nbytes)
    # cudaMemcpy, host to device
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # cuda kernel 
    kernel = ADD.get_function('add')
    block_size = (16, 1, 1)
    kernel(N, a_gpu, b_gpu, c_gpu, block=block_size)

    # cudaMemcpy, device to host
    c_result_gpu = np.empty_like(a)
    cuda.memcpy_dtoh(c_result_gpu, c_gpu)

    assert(np.all(c_result_gpu == a+b))

if __name__ == '__main__':
    simple_test(10)
    add_two_Identity(10)




    