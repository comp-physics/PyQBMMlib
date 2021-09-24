# File of utility functions, used by tests and examples

import numpy as np
from numpy.core.arrayprint import dtype_is_implied
import pycuda.driver as cuda 
import pycuda.autoinit

def init_moment_6(size: int):
    '''
    Initialize a dummy input of specified size for Chyqmom4
    '''
    one_moment = np.asarray([1.0, 1.0, 1.0, 1.01,  
                        1, 1.01], 
                        dtype=np.float32)
    moments = cuda.aligned_zeros((6, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments

def init_moment_10(size: int):
    '''
    Initialize a dummy input of specified size for Chyqmom4
    '''
    one_moment = np.asarray([1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603], 
                        dtype=np.float32)
    moments = cuda.aligned_zeros((10, size), dtype=np.float32)
    for i in range(size):
        moments[:, i] = one_moment
    
    return moments

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

