#CHYQMOM9

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import ctypes

import numpy as np

import threading
import time

from chyqmom_kernel import HELPER, CHYQMOM9, HYQMOM, SIZEOF_FLOAT

class Chyqmom9: 

    def __init__(self, 
        num_gpu: int, 
        in_size: int, 
        stream: int = 1, 
        block: tuple = (1024, 1, 1)) -> None:
        ''' Constructor '''
        cuda.init()

        max_dev = cuda.Device.count()
        if num_gpu > max_dev:
            raise ValueError("Error: not enough GPU available")
        if num_gpu > in_size:
            raise ValueError("Error: More GPU assigned than input size")

        self.num_device = num_gpu
        self.in_size = in_size
        self.context_list = []
        self.num_stream = stream
        self.input_moment = None

        # GPU block and grid sizes 
        self.block_size = block
        self.grid_size = ((self.in_size)//self.block_size[0] + 1, 1, 1)

        # based on the number of GPU, initialize their context
        for i in range(self.num_device):
            dev = cuda.Device(i)
            ctx = dev.make_context()
            self.context_list.append(ctx)
        
        self.init_kernel()
        self.init_memory()
    
        # if either init_kernel or init_memory fails, this will be false, 
        # and memory deallocation in the deconstructor will not proceed
        self.initialized = True
    
    def init_kernel(self) -> None:
        ''' 
        Compile the GPU kernels 
        Each GPU context gets its own compilation 
        A seperate thread is launched to do the actual compilation for each GPU
        '''
        # Initialize lists of kernels. 
        self.float_value_set = [None for i in range(self.num_device)]
        self.float_array_set = [None for i in range(self.num_device)]

        self.c_kernel = [None for i in range(self.num_device)]
        self.mu_yf = [None for i in range(self.num_device)]
        self.wout_kernel = [None for i in range(self.num_device)]
        self.xout_kernel = [None for i in range(self.num_device)]
        self.yout_kernel = [None for i in range(self.num_device)]
        self.hyq = [None for i in range(self.num_device)]

        # Launch kernel threads to init kernels for each GPU
        kernel_threads = []
        for i, ctx in enumerate(self.context_list):
            kernel_threads.append(threading.Thread(target=self.init_thread_kernel, args=(i, ctx)))
            kernel_threads[i].start()

        # wait for the threads to finish
        for t in kernel_threads: 
            t.join()
    
    def init_memory(self) -> None:
        '''
        Initialize GPU memory 
        each GPU gets its own number of streams, and each stream gets its own
        memory allocation
        '''
        # initialize memory lists
        self.moments_device = [[] for i in range(self.num_device)]
        self.moment_chunk_host = [[] for i in range(self.num_device)]
        self.x_chunk_host = [[] for i in range(self.num_device)]
        self.y_chunk_host = [[] for i in range(self.num_device)]
        self.w_chunk_host = [[] for i in range(self.num_device)]
        self.x_device = [[] for i in range(self.num_device)]
        self.y_device = [[] for i in range(self.num_device)]
        self.w_device = [[] for i in range(self.num_device)]

        self.c_moments = [[] for i in range(self.num_device)]
        self.mu = [[] for i in range(self.num_device)]
        self.yf = [[] for i in range(self.num_device)]

        self.m1 = [[] for i in range(self.num_device)]
        self.x1 = [[] for i in range(self.num_device)]
        self.w1 = [[] for i in range(self.num_device)]
        self.x2 = [[] for i in range(self.num_device)]
        self.w2 = [[] for i in range(self.num_device)]

        # Host memory that stores the output
        self.w_out = cuda.aligned_zeros((4, self.in_size), dtype=np.float32)
        self.x_out = cuda.aligned_zeros((4, self.in_size), dtype=np.float32)
        self.y_out = cuda.aligned_zeros((4, self.in_size), dtype=np.float32)

        self.streams = [[] for i in range(self.num_device)]
        # number of input allocated to each thread
        size_per_thread = np.ceil(self.in_size / self.num_device)

        mem_thread = []
        for i, ctx in enumerate(self.context_list):
            mem_thread.append(threading.Thread(
                        target=self.init_thread_memory, 
                        args=(i, ctx, size_per_thread)))
            mem_thread[i].start()
        
        for t in mem_thread:
            t.join()

    def init_thread_memory(self, dev_id:int, ctx:cuda.Context, alloc_size: int) -> None:
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
            self.moments_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 10))))
            self.w_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 9))))
            self.x_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 9))))
            self.y_device[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 9))))

            # set host memory for returned output
            self.c_moments[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 7))))
            self.mu[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.yf[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))

            self.m1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 5))))
            self.float_value_set[dev_id](self.m1[dev_id][i], np.float32(0), size_per_batch, size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])
            self.float_value_set[dev_id](self.m1[dev_id][i], np.float32(1), size_per_batch, np.int32(0),
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.x1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.w1[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.x2[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))
            self.w2[dev_id].append((cuda.mem_alloc(int(SIZEOF_FLOAT * size_per_batch * 3))))

        ctx.synchronize()
        ctx.pop()

    def init_thread_kernel(self, i: int, ctx: cuda.Context) -> None:
        '''
        Single thread that compiles GPU kernel for each GPU
        '''
        self.context_list[i].push()

        helper_func = SourceModule(HELPER)
        core_func = SourceModule(CHYQMOM9)
        hyq = SourceModule(HYQMOM)
        self.float_value_set[i] = helper_func.get_function('float_value_set')
        self.float_array_set[i] = helper_func.get_function('float_array_set')

        self.c_kernel[i] = core_func.get_function('chyqmom9_cmoments')
        self.mu_yf[i] = core_func.get_function('chyqmom9_mu_yf')
        self.wout_kernel[i] = core_func.get_function('chyqmom9_wout')
        self.xout_kernel[i] = core_func.get_function('chyqmom9_xout')
        self.yout_kernel[i] = core_func.get_function('chyqmom9_yout')
        self.hyq[i] = hyq.get_function("hyqmom3")
        self.context_list[i].pop()

    def set_args(self, moment: np.ndarray) -> None:
        '''
        Set the input moment 
        '''
        self.input_moment = moment
        self.num_moments_per_thread = int(np.ceil(self.in_size / self.num_device))
        start_loc = 0

        self.moment_chunk_host = [[] for i in range(self.num_device)]
        self.x_chunk_host = [[] for i in range(self.num_device)]
        self.y_chunk_host = [[] for i in range(self.num_device)]
        self.w_chunk_host = [[] for i in range(self.num_device)]

        
        thread_list = []
        for i, ctx in enumerate(self.context_list): 
            end_loc = start_loc + self.num_moments_per_thread
            if end_loc > self.in_size:
                end_loc = self.in_size - 1

            t = threading.Thread(target=self.set_thread_args, args=(i, ctx,
                        self.input_moment[:, start_loc:end_loc],
                        self.w_out[:, start_loc:end_loc], 
                        self.x_out[:, start_loc:end_loc],
                        self.y_out[:, start_loc:end_loc]))

            t.start()
            thread_list.append(t)
        
        for t in thread_list:
            t.join()
    
    def run(self) -> None:
        ''' 
        Run CHYQMOM9 on the input moment 
        '''
        if self.input_moment is not None: 
            run_list = []
            for i, ctx in enumerate(self.context_list):
                run = threading.Thread(target=self.run_thread, args=(i, ctx))
                run.start()
                run_list.append(run)
            
            for t in run_list:
                t.join()



    def set_thread_args(self, dev_id: int, ctx: cuda.Context,
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

    def run_thread(self, dev_id, ctx) -> tuple:
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
            self.float_array_set[dev_id](self.m1[dev_id][i], self.c_moments[dev_id][i], 
                    np.int32(size_per_batch * 2), np.int32(size_per_batch * 3), np.int32(size_per_batch * 4),
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.hyq[dev_id](self.m1[dev_id][i], self.x1[dev_id][i], self.w1[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.mu_yf[dev_id](self.c_moments[dev_id][i], 
                    self.x1[dev_id][i], self.w1[dev_id][i], 
                    self.yf[dev_id][i], self.mu[dev_id][i], size_per_batch,
                    block=self.block_size, grid=self.grid_size, stream=self.streams[dev_id][i])

            self.float_array_set[dev_id](self.m1[dev_id][i], self.mu[dev_id][i], 
                    np.int32(size_per_batch * 3), np.int32(size_per_batch * 2), np.int32(0),
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

    def __del__(self):
        '''
        Deconstructor

        Free all GPU resources 
        '''
        for i, ctx in enumerate(self.context_list):
            for j in range(0, self.num_stream, 1):
                # print("what?")
                # print("device: {}, stream: {}".format(i, j))

                self.moment_chunk_host[i][j].base.unregister()
                self.x_chunk_host[i][j].base.unregister()
                self.y_chunk_host[i][j].base.unregister()
                self.w_chunk_host[i][j].base.unregister()

                self.moments_device[i][j].free()
                self.w_device[i][j].free()
                self.x_device[i][j].free()
                self.y_device[i][j].free()

                self.c_moments[i][j].free()
                self.mu[i][j].free()
                self.yf[i][j].free()

                self.m1[i][j].free()
                self.x1[i][j].free()
                self.w1[i][j].free()
                self.x2[i][j].free()
                self.w2[i][j].free()

            ctx.synchronize()
            ctx.pop()

def init_moment_10(size: int):
    one_moment = np.asarray([1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603], 
                        dtype=np.float32)
    moments = cuda.aligned_zeros((10, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments

def time_runtime(fname: str, max_input_size_mag: int, num_points: int, trial: int, num_device: int):

    result = np.zeros((num_points, trial + 1))

    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
        this_result = np.zeros(trial + 1)
        this_result[0] = int(in_size)

        T2 = Chyqmom9(num_device, int(in_size))
        this_moment = init_moment_10(int(in_size))
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
    # in_size = int(3e7)
    # dev_size = 1
    # stream_size = 1

    # T = Chyqmom9(dev_size, in_size, stream=stream_size)
    # moment = init_moment_10(in_size)
    # T.set_args(moment)
    # # print(T.moment_chunk_host[0][0].shape)

    # start_time = time.perf_counter()
    # res = T.run()
    # print(T.w_chunk_host)
    # print(T.x_chunk_host)
    # print(T.y_chunk_host)
    # stop_time = time.perf_counter()
    # this_result = (stop_time - start_time) * 1e3 # ms
    # print("[Chyqmom4] input moment size: {} \nN GPU: {}, N Stream: {} \ntime (ms): {}".format(
    #     in_size, dev_size, stream_size, this_result))

    time_runtime("chyqmom9_result_1gpu.csv", 7, 200, 5, 1)

    # del T

