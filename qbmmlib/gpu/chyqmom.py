

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import ctypes

import numpy as np

import threading
import time

from qbmmlib.gpu.chyqmom_kernel import SIZEOF_FLOAT

class Chyqmom:
    """
    The base class for all CHyqmom methods. Spcific CHyqmom algorithms
    from this class

    Allcate and manage resources required by chyqmom methods

    Warning: 
        This class is meant to be inherited by CHyqmom classes only. 
        It is not meant to be called directly by the user
    """

    def __init__(self, 
        num_gpu: int, 
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
        self.initialized = False

        cuda.init()
        max_dev = cuda.Device.count()
        if num_gpu > max_dev:
            raise ValueError("Error: not enough GPU available")
        if num_gpu > in_size:
            raise ValueError("Error: More GPU assigned than input size")

        self.num_device = num_gpu
        self.in_size = in_size    # number of moment input 
        self.context_list = []    # list of GPU context 
        self.num_stream = stream  # numer of streams on each GPU
        self.input_moment = None  # Input 

        # GPU block and grid sizes 
        self.block_size = block
        self.grid_size = ((self.in_size)//self.block_size[0] + 1, 1, 1)

        # based on the number of GPU, initialize their context
        for i in range(self.num_device):
            dev = cuda.Device(i)
            ctx = dev.make_context()
            self.context_list.append(ctx)
    
        self._init_kernel()
        self._init_memory()

        # if either init_kernel or init_memory fails, this will be false, 
        # and memory deallocation in the deconstructor will not proceed
        self.initialized = True
    
    def _init_kernel(self) -> None:
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
            kernel_threads.append(threading.Thread(target=self._init_thread_kernel, args=(i, ctx)))
            kernel_threads[i].start()
        
        # wait for the threads to finish
        for t in kernel_threads: 
            t.join()
    
    def _init_memory(self) -> None:
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
                        target=self._init_thread_memory, 
                        args=(i, ctx, size_per_thread)))
            mem_thread[i].start()
        
        for t in mem_thread:
            t.join()

    def _init_thread_memory(self, dev_id:int, ctx:cuda.Context, alloc_size: int) -> None:
        '''
        Single thread that initializes the memory for all the stream for a single 
        GPU. 
        Only implemented in the subclasses
        '''
        raise NotImplementedError("Base class does not have this method implemented")

    def _init_thread_kernel(self, i: int, ctx: cuda.Context) -> None:
        '''
        Single thread that compiles GPU kernel for each GPU

        Only implemented in the subclasses
        '''
        raise NotImplementedError("Base class does not have this method implemented")

    def set_args(self, moment: np.ndarray) -> None:
        '''
        Set the input moment, once the class is created 

        The exact size of the input array is set during the initialization 
        of the class. The size of the input moment here must match the
        specified size. 

        Parameters
        -----------------------------------------------------------------------
        moment: numpy.ndarray
            the numpy array containing the actual data. 

        '''
        self.input_moment = moment
        self.num_moments_per_thread = int(np.ceil(self.in_size / self.num_device))
        start_loc = 0

        # host memory, divided into chunks for each GPU streams
        self.moment_chunk_host = [[] for i in range(self.num_device)]
        self.x_chunk_host = [[] for i in range(self.num_device)]
        self.y_chunk_host = [[] for i in range(self.num_device)]
        self.w_chunk_host = [[] for i in range(self.num_device)]

        # Allocate workload to different GPU 
        thread_list = []
        for i, ctx in enumerate(self.context_list): 
            end_loc = start_loc + self.num_moments_per_thread
            if end_loc > self.in_size:
                end_loc = self.in_size - 1

            t = threading.Thread(target=self._set_thread_args, args=(i, ctx,
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
        Run CHyqmom method on the  specified input moment 

        '''
        if self.input_moment is not None: 
            run_list = []
            for i, ctx in enumerate(self.context_list):
                run = threading.Thread(target=self._run_thread, args=(i, ctx))
                run.start()
                run_list.append(run)
            
            for t in run_list:
                t.join()
        else: 
            raise ValueError("Input moment not set!")

    def _set_thread_args(self, dev_id: int, ctx: cuda.Context,
                         moment:np.ndarray, 
                         w_out: np.ndarray,  
                         x_out: np.ndarray,  
                         y_out: np.ndarray):
        '''
        Set the input moment for all the stream for a specific GPU
        '''
        raise NotImplementedError("Base class does not have this method implemented")

    def _run_thread(self, dev_id, ctx) -> tuple:
        '''
        A thread that runs CHYQMOM4 on a single GPU
        '''
        raise NotImplementedError("Base class does not have this method implemented")

    def __del__(self):
        '''
        Deconstructor

        Free all GPU resources 
        '''
        if self.initialized:
            for i, ctx in enumerate(self.context_list):
                for j in range(0, self.num_stream, 1):

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



