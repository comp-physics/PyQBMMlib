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

def memcpy_to_device_cmp():

    event_start_1 = cuda.Event()
    event_stop_1 = cuda.Event()

    event_start_2 = cuda.Event()
    event_stop_2 = cuda.Event()

    array = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    array = np.vstack([array, array])
    array_size = array.nbytes

    mu = cuda.mem_alloc(array_size)
    event_start_1.record()
    cuda.memcpy_htod(mu, array)
    event_stop_1.record()

    event_start_2.record()
    mu2 = cuda.to_device(array)
    event_stop_2.record()

    event_stop_1.synchronize()
    event_stop_2.synchronize()

    t1 = event_stop_1.time_since(event_start_1)
    t2 = event_stop_2.time_since(event_start_2)
    print(t1)
    print(t2)

def batch_memcpy_cmp(size: int, batch: int):
    event_start_1 = cuda.Event()
    event_stop_1 = cuda.Event()
    event_start_2 = cuda.Event()
    event_stop_2 = cuda.Event()

    array = np.random.rand(size, 9)
    array.astype(np.float32)    

    mem = cuda.aligned_zeros_like(array)
    mem = cuda.register_host_memory(mem, cuda.mem_host_register_flags.DEVICEMAP)

    mem_d = cuda.mem_alloc_like(mem)

    event_start_1.record()
    cuda.memcpy_htod(mem_d, mem)
    event_stop_1.record()
    event_stop_1.synchronize()

    mem2 = []
    this_mem = []
    size_per_batch = int(size/batch)
    for i in range(batch):
        mem2.append(cuda.mem_alloc_like(array[i*size_per_batch:(i+1)* size_per_batch]))
        this_mem.append(array[i*size_per_batch:(i+1)* size_per_batch])
        this_mem[i] = cuda.register_host_memory(this_mem[i], cuda.mem_host_register_flags.DEVICEMAP)
    
    event_start_2.record()
    for i in range(batch):
        cuda.memcpy_htod(mem2[i], this_mem[i])
    event_stop_2.record()
    event_stop_2.synchronize()

    t1 = event_stop_1.time_since(event_start_1)
    t2 = event_stop_2.time_since(event_start_2)
    print("batch_memcpy_cmp size", size, " batch ", batch)
    print(t1)
    print(t2)


if __name__ == '__main__':
    batch_memcpy_cmp(10000000, 10)