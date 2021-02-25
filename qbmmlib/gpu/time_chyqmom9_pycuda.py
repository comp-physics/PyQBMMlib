from chyqmom9_pycuda import init_moment_10, chyqmom9_pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

if __name__ == "__main__":

    res_file_name = 'result.csv'
    max_input_size_mag = 6
    num_points = 100
    trial = 5


    result = np.zeros((num_points, trial + 1))

    for idx, in_size in enumerate(np.logspace(1, max_input_size_mag, num=num_points)):
        this_result = np.zeros(trial + 1)
        this_result[0] = int(in_size)

        w = cuda.aligned_zeros((9, int(in_size)), dtype=np.float32)
        x = cuda.aligned_zeros((9, int(in_size)), dtype=np.float32)
        y = cuda.aligned_zeros((9, int(in_size)), dtype=np.float32)

        this_moment = init_moment_10(int(in_size))
        for i in range(1, trial, 1):
            this_result[i] = chyqmom9_pycuda(this_moment, int(in_size), w, x, y, 1)
        result[idx] = this_result
    np.savetxt(res_file_name, result, delimiter=',')




