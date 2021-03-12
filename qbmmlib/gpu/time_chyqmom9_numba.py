from inversion_vectorized import hyqmom2, hyqmom3, chyqmom4, chyqmom9

import numpy as np
import time

def init_moment_10(size: int):
    one_moment = np.asarray([1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603], 
                        dtype=np.float32)
    moments = np.zeros((size, 10), dtype=np.float32)
    for i in range(size):
        moments[i, :] = one_moment
    
    return moments

if __name__ == "__main__":
    res_file_name = 'results.csv'
    max_input_size_mag = 6
    num_points = 10
    trial = 5
    
    result = np.zeros((num_points, trial + 1))

    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
        print(int(in_size))
        this_result = np.zeros(trial + 1)
        this_result[0] = int(in_size)

        this_moment = init_moment_10(int(in_size))
        for i in range(1, trial+1, 1):
            t_begin = time.perf_counter()
            this_result[i] = chyqmom9(this_moment, int(in_size))
            t_end = time.perf_counter()
            this_result[i] = t_end - t_begin
        result[idx] = this_result

    np.savetxt(res_file_name, result, delimiter=',')
    

