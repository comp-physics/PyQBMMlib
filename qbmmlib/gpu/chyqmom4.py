

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import ctypes

import numpy as np

import threading
import time

from chyqmom_kernel import HELPER, CHYQMOM4, HYQMOM, SIZEOF_FLOAT
from chyqmom import Chyqmom

from util import init_moment_6

class Chyqmom4(Chyqmom):     
    """
    The Chyqmom4 method.
    """

    def __init__(self, num_gpu: int, 
        in_size: int, 
        stream: int = 1, 
        block: tuple = (1024, 1, 1)) -> None:
        ''' 
        Constructor 
        
        Initailize resources based on desired number of device. Note that 
        this construct an empty class, and data will have to be loaded in 
        seperately through a different method

        Parameters
        -----------------------------------------------------------------------
        num_gpu: int 
            Number of GPUs (devices) used for this class 
        in_size: int 
            Number of input moments
        stream: int 
            Number of concurrent stream initialize for each device
        block: tuple
            Thread block size for GPU kernels

        '''
        super().__init__(num_gpu, in_size, stream=stream, block=block)

    def _init_thread_memory(self, dev_id:int, ctx:cuda.Context, alloc_size: int) -> None:
        '''
        Single thread that initializes the memory for all the stream for a single 
        GPU. 
        '''
        ctx.push()
        size_per_batch = np.int32(np.ceil(alloc_size / self.num_stream))

        # Initialize streams
        for i in range(self.num_stream):
            self.streams[dev_id].append(cuda.Stream())

        for i in range(0, self.num_stream, 1):
            # allocate memory on device
            self.moments_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 6))))
            self.w_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 4))))
            self.x_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 4))))
            self.y_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 4))))

            # set host memory for returned output
            self.c_moments[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.mu[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 1))))
            self.yf[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 2))))

            self.m1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.float_value_set[dev_id](self.m1[dev_id][i], np.float32(0), size_per_batch, size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
            self.float_value_set[dev_id](self.m1[dev_id][i], np.float32(1), size_per_batch, np.int32(0),
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.x1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 2))))
            self.w1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 2))))
            self.x2[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 2))))
            self.w2[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 2))))

        ctx.synchronize()
        ctx.pop()

    def _init_thread_kernel(self, i: int, ctx: cuda.Context) -> None:
        '''
        Single thread that compiles GPU kernel for each GPU
        '''
        self.context_list[i].push()

        helper_func = SourceModule(HELPER)
        core_func = SourceModule(CHYQMOM4)
        hyq = SourceModule(HYQMOM)
        self.float_value_set[i] = helper_func.get_function('float_value_set')
        self.float_array_set[i] = helper_func.get_function('float_array_set')

        self.c_kernel[i] = core_func.get_function('chyqmom4_cmoments')
        self.mu_yf[i] = core_func.get_function('chyqmom4_mu_yf')
        self.wout_kernel[i] = core_func.get_function('chyqmom4_wout')
        self.xout_kernel[i] = core_func.get_function('chyqmom4_xout')
        self.yout_kernel[i] = core_func.get_function('chyqmom4_yout')
        self.hyq[i] = hyq.get_function("hyqmom2")
        self.context_list[i].pop()

    def _set_thread_args(self, dev_id: int, ctx: cuda.Context,
                         moment:np.ndarray, 
                         w_out: np.ndarray,  
                         x_out: np.ndarray,  
                         y_out: np.ndarray):
        '''
        Set the input moment for all the stream for a specific GPU
        '''

        ctx.push()
        # number of input for this GPU
        max_size = moment.shape[1]

        # loop through the streams to set their input 
        for i in range(0, self.num_stream, 1):
            # Size of input allocated for each stream
            size_per_batch = int(np.ceil(max_size / self.num_stream))

            # location on the original input array where the input to this stream starts
            loc = np.int32((i) * size_per_batch)
            if loc + size_per_batch > max_size: 
                size_per_batch = max_size - loc
            
            self.moment_chunk_host[dev_id].append(
                np.ascontiguousarray(moment[:, loc:loc+size_per_batch], dtype=np.float32)
            )
            self.moment_chunk_host[dev_id][i] = cuda.register_host_memory(
                self.moment_chunk_host[dev_id][i], cuda.mem_host_register_flags.PORTABLE
            )
            self.w_chunk_host[dev_id].append(np.ascontiguousarray(np.zeros_like(w_out[:, loc:loc+size_per_batch])))
            self.w_chunk_host[dev_id][i] = cuda.register_host_memory(self.w_chunk_host[dev_id][i], cuda.mem_host_register_flags.PORTABLE)

            self.x_chunk_host[dev_id].append(np.ascontiguousarray(np.zeros_like(x_out[:, loc:loc+size_per_batch])))
            self.x_chunk_host[dev_id][i] = cuda.register_host_memory(self.x_chunk_host[dev_id][i], cuda.mem_host_register_flags.PORTABLE)

            self.y_chunk_host[dev_id].append(np.ascontiguousarray(np.zeros_like(y_out[:, loc:loc+size_per_batch])))
            self.y_chunk_host[dev_id][i] = cuda.register_host_memory(self.y_chunk_host[dev_id][i], cuda.mem_host_register_flags.PORTABLE)


        ctx.synchronize()
        ctx.pop()

    def _run_thread(self, dev_id, ctx) -> tuple:
        '''
        A thread that runs CHYQMOM4 on a single GPU
        '''
        ctx.push()

        for i in range(0, self.num_stream, 1):
            size_per_batch = np.int32(np.ceil(self.num_moments_per_thread / self.num_stream))         
            loc = np.int32((i) * size_per_batch)
            if loc + size_per_batch > self.num_moments_per_thread: 
                size_per_batch = self.num_moments_per_thread - loc

            cuda.memcpy_htod_async(self.moments_device[dev_id][i], self.moment_chunk_host[dev_id][i],
                    stream=self.streams[dev_id][i])
        
            self.c_kernel[dev_id](self.moments_device[dev_id][i], self.c_moments[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
                
            self.float_array_set[dev_id](self.m1[dev_id][i], self.c_moments[dev_id][i], 
                    np.int32(size_per_batch), np.int32(size_per_batch * 2), np.int32(0),
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.hyq[dev_id](self.m1[dev_id][i], self.x1[dev_id][i], self.w1[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.mu_yf[dev_id](self.c_moments[dev_id][i], 
                    self.x1[dev_id][i], self.w1[dev_id][i], 
                    self.yf[dev_id][i], self.mu[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.float_array_set[dev_id](self.m1[dev_id][i], self.mu[dev_id][i], 
                    np.int32(size_per_batch), np.int32(size_per_batch * 2), np.int32(0),
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
            self.hyq[dev_id](self.m1[dev_id][i], self.x2[dev_id][i], self.w2[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

        for i in range(0, self.num_stream, 1):
            self.streams[dev_id][i].synchronize()
            self.wout_kernel[dev_id](self.moments_device[dev_id][i], self.w1[dev_id][i], 
                    self.w2[dev_id][i], self.w_device[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            cuda.memcpy_dtoh_async(self.w_chunk_host[dev_id][i], self.w_device[dev_id][i], stream=self.streams[dev_id][i])
            
            self.xout_kernel[dev_id](self.moments_device[dev_id][i], self.x1[dev_id][i], 
                        self.x_device[dev_id][i], size_per_batch,
                        block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
            cuda.memcpy_dtoh_async(self.x_chunk_host[dev_id][i], self.x_device[dev_id][i], stream=self.streams[dev_id][i])
            
            self.yout_kernel[dev_id](self.moments_device[dev_id][i], self.x2[dev_id][i], 
                        self.yf[dev_id][i], self.y_device[dev_id][i], size_per_batch,
                        block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
            cuda.memcpy_dtoh_async(self.y_chunk_host[dev_id][i], self.y_device[dev_id][i], stream=self.streams[dev_id][i])
        
        ctx.synchronize()
        ctx.pop()

def time_runtime(fname: str, max_input_size_mag: int, num_points: int, trial: int, num_device: int):

    result = np.zeros((num_points, trial + 1))

    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
        this_result = np.zeros(trial + 1)
        this_result[0] = int(in_size)

        T2 = Chyqmom4(num_device, int(in_size))
        this_moment = init_moment_6(int(in_size))
        T2.set_args(this_moment)
        for i in range(1, trial, 1):
            start_time = time.perf_counter()
            T2.run()
            stop_time = time.perf_counter()
            this_result[i] = (stop_time - start_time) * 1e3 #ms

        result[idx] = this_result
        print(int(in_size), ": {:.6f}".format(this_result[1]))
    np.savetxt(fname, result, delimiter=',')

if __name__ == "__main__":
    in_size = int(3e6)
    dev_size = 1
    stream_size = 2

    T = Chyqmom4(dev_size, in_size, stream=stream_size)
    moment = init_moment_6(in_size)
    T.set_args(moment)
    # print(T.moment_chunk_host[0][0].shape)

    start_time = time.perf_counter()
    res = T.run()
    print(T.w_chunk_host)
    print(T.x_chunk_host)
    print(T.y_chunk_host)
    stop_time = time.perf_counter()
    this_result = (stop_time - start_time) * 1e3 # ms
    print("[Chyqmom4] input moment size: {} \nN GPU: {}, N Stream: {} \ntime (ms): {}".format(
        in_size, dev_size, stream_size, this_result))

    # time_runtime("result_2gpu.csv", 7, 200, 5, 2)


