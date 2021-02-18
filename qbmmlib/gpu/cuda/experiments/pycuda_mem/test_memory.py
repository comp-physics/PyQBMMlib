import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

KERNEL = SourceModule('''
    __global__ void print_val(float *addr, const int i) {
        const int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
        printf("[%d] i = %d printing %f \\n", tIdx, i, addr[i]);
    }

''')


if __name__ == '__main__':

    array = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    array = np.vstack([array, array])
    array_size = array.nbytes
    i = np.int32(0)

    blockSize = (1, 1, 1)

    mu = cuda.mem_alloc(array_size)

    a = array[:, 1:]
    a = np.ascontiguousarray(a, dtype=np.float32)
    print(a)

    cuda.memcpy_htod_async(mu, a)
    print_kernel = KERNEL.get_function('print_val')
    print_kernel(mu, i, block=blockSize)

    mu.free()