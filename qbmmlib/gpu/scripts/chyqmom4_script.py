import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import time
from numba import njit, config, set_num_threads, threading_layer

from inversion_vectorized import chyqmom4 as chyqmom4_cpu
from qbmmlib.gpu.chyqmom4 import Chyqmom4 

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

if __name__ == "__main__": 
    res_file_name = 'chyqmom4_res_2.csv' # output data file name
    max_input_size_mag = 7             # max number of input point (power of 10)
    num_points = 200                   # number of runs collected 
    trial = 5                          # For each run, the number of trials run. 
    num_device = 1                     # number of GPUs used

    config.THREADING_LAYER = 'threadsafe'
    set_num_threads(12)                # numba: number of concurrent CPU threads 
    print("Threading layer chosen: %s" % threading_layer())
    
    ## Header: 
    #  [num input, cpu_result (ms), gpu_result (ms)] 
    result = np.zeros((num_points, 4))

    this_result_cpu = np.zeros(trial)
    this_result_gpu = np.zeros(trial)

    # generate a set of input data size, linear in log space between 1 and maximum 
    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
        result[idx, 0] = idx
        result[idx, 1] = int(in_size)

        T2 = Chyqmom4(num_device, int(in_size), stream=2)
        this_moment = init_moment_6(int(in_size))
        T2.set_args(this_moment)

        for i in range(0, trial, 1):
            # GPU time
            start_time = time.perf_counter()
            T2.run()
            stop_time = time.perf_counter()
            this_result_gpu[i] = (stop_time - start_time) * 1e3 #ms

            # numba time: 
            start_time = time.perf_counter()
            chyqmom4_cpu(this_moment.transpose(), int(in_size))
            stop_time = time.perf_counter()
            this_result_cpu[i] = (stop_time - start_time) * 1e3 #ms
        
        result[idx, 1] = np.min(this_result_cpu)
        result[idx, 2] = np.min(this_result_gpu)
        print("[{}/{}] running on {} inputs, CPU: {:4f}, GPU: {:4f}".format(
            idx, num_points, int(in_size), result[idx, 1], result[idx, 2]))

    np.savetxt(res_file_name, result, delimiter=',')
    