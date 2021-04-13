
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import threading
import numpy as np
TEST_CODE = '''

    __global__ void test_kernel(int N) {
        // 1D block thread indexing 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        printf("thread_idx: %d got integer: %d\\n", idx, N);
    }
'''

ADD_CODE = '''
    __global__ void add(int N, float* a, float* b, float* c) {
        // 1D block thread indexing 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            c[idx] = a[idx] + b[idx];
            printf("thread_idx: %d c[idx]: %f\\n", idx, c[idx]);
            idx += blockDim.x * gridDim.x;
        }
    }  
'''

class TestClass: 

    def __init__(self, num_device: int, input_size: int):

        self.dev_num = num_device
        self.in_size = input_size
        print("Using {} GPU devices".format(self.dev_num))
        self.context_list = []

        for i in range(self.dev_num):
            dev = cuda.Device(i)
            ctx = dev.make_context()
            self.context_list.append(ctx)
            print("context created for {}, {}".format(ctx.get_device().name(), ctx.get_device().pci_bus_id()))

        self.kernel = []
        for i, ctx in enumerate(self.context_list): 
            thread = threading.Thread(target=self.init_thread_kernel, args=(i, ctx))
            thread.start()
            thread.join()
        self.init_memory()
    
    def set_args(self, a: np.ndarray, b:np.ndarray):
        in_shape = a.shape
        self.a = []
        self.b = []
        self.c = []

        for i, ctx in enumerate(self.context_list):
            self.c.append(cuda.aligned_zeros(in_shape, dtype=np.float32))
            self.a.append(np.ascontiguousarray(a, dtype=np.float32))
            self.b.append(np.ascontiguousarray(a, dtype=np.float32))
    
    def init_memory(self):
        self.a_gpu = [None for ctx in self.context_list]
        self.b_gpu = [None for ctx in self.context_list]
        self.c_gpu = [None for ctx in self.context_list]

        # floats_per_thread = self.in_size // self.dev_num + 1 # round up 
        size_per_float = np.int32(np.dtype(np.float32).itemsize)
        size_total = self.in_size * size_per_float

        for i, ctx in enumerate(self.context_list): 
            thread = threading.Thread(target=self.init_thread_memory, args=(i, ctx, size_total))
            thread.start()
    
    def init_thread_kernel(self, i, ctx):
        ctx.push()
        ADD = SourceModule(ADD_CODE)
        kernel = ADD.get_function('add')
        self.kernel.append(kernel)
        ctx.pop()

    def init_thread_memory(self, i, ctx, size):
        ctx.push()
        print(int(size))
        self.a_gpu[i] = cuda.mem_alloc(int(size))
        self.b_gpu[i] = cuda.mem_alloc(int(size))
        self.c_gpu[i] = cuda.mem_alloc(int(size))
        ctx.pop()
    
    def thread_run(self, i, ctx, size):
        ctx.push()

        cuda.memcpy_htod(self.a_gpu[i], self.a[i])
        cuda.memcpy_htod(self.b_gpu[i], self.b[i])

        block_size = (16, 1, 1)
        self.kernel[i](np.int32(size), self.a_gpu[i], self.b_gpu[i], self.c_gpu[i], block=block_size)

        cuda.memcpy_dtoh(self.c[i], self.c_gpu[i])
        ctx.pop()
    
    def start(self):
        for i, ctx in enumerate(self.context_list):
            t = threading.Thread(target=self.thread_run, args=(i, ctx, self.in_size))
            t.run()
    
    def get_c(self):
        for i in self.c:
            print(i)

    def __del__(self):
        for i, ctx in enumerate(self.context_list): 
            self.a_gpu[i].free()
            self.b_gpu[i].free()
            self.c_gpu[i].free()
            ctx.pop()
        

def test_run(ctx):

    cuda.init()
    dev_num = cuda.Device.count()
    print("Detected {} GPU devices".format(dev_num))

    context_list = []
    thread_list = []
    for i in range(dev_num):
        dev = cuda.Device(i)
        ctx = dev.make_context()
        context_list.append(ctx)
        x = threading.Thread(target=test_run, args=(ctx, ))
        thread_list.append(x)
    
    for i, ctx in enumerate(context_list): 
        thread_list[i].start()
    
    for ctx in context_list:
        ctx.pop()



if __name__ == "__main__":
    cuda.init()

    dev_num = cuda.Device.count()
    print("Detected {} GPU devices".format(dev_num))

    a = np.ones([3, 3])
    b = np.ones([3, 3])

    T = TestClass(2, 3*3)
    T.set_args(a, b)
    T.start()
    T.get_c()


