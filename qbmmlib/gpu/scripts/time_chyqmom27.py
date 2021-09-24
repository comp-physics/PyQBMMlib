def init_moment_10(size: int):
    one_moment = np.asarray([1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603], 
                        dtype=np.float32)
    moments = cuda.aligned_zeros((10, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import time
from numba import njit, config, set_num_threads, threading_layer

from inversion_vectorized import chyqmom27 as chyqmom27_cpu
from qbmmlib.gpu.chyqmom9 import Chyqmom27 

def init_moment_27(size: int):
    one_moment = np.asarray([1,    1,      1,      1, 
                             1.01, 1,      1,      1.01, 
                             1,    1.01,   1.03,   1.03,
                             1.03, 1.0603, 1.0603, 1.0603], 
                             dtype=np.float32)
    moments = cuda.aligned_zeros((16, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments
