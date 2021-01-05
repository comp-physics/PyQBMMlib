from numba import config, njit, threading_layer, prange
import numpy as np
import psutil
import time

@njit(parallel=True)
def prange_test(A,n):
    s = 0
    for i in prange(n):
        s += A[i] * A[i] + A[i] + A[i] - A[i]
    return s


@njit(parallel=False)
def range_test(A,n):
    s = 0
    for i in range(n):
        s += A[i] * A[i] + A[i] + A[i] - A[i]
    return s

if __name__ == "__main__":
    # Can force the 'treading layers' here
    # Options are 'tbb', 'omp', or 'workspace'
    # config.THREADING_LAYER = 'tbb'

    # print(available_cpu_count())
    # exit()
    
    ntests = 10
    ncpus = psutil.cpu_count(logical = False)
    print('It seems that you have %i physical CPU cores and %i tests' % (ncpus, ntests))
    n = int(1e7)
    A = np.zeros(n)
    # A = A + 1.
    # A = A/float(n)

    serial_times = np.zeros(ntests)
    parallel_times = np.zeros(ntests)
    t_start = time.perf_counter()
    s = prange_test(A,n)
    t_end = time.perf_counter()
    for i in range(ntests):
        t_start = time.perf_counter()
        s = range_test(A,n)
        t_end = time.perf_counter()
        serial_times[i] = (t_end - t_start)

        t_start = time.perf_counter()
        s = prange_test(A,n)
        t_end = time.perf_counter()
        parallel_times[i] = (t_end - t_start)

    print("Serial Times   [s]",serial_times)
    print("Parallel Times [s]",parallel_times)

    print("Avg serial time   [s]", np.mean(serial_times))
    print("Avg parallel time [s]", np.mean(parallel_times))

    print("Perfect speed up would be be:", np.mean(serial_times)/float(ncpus))
    print("Threading layer chosen: %s" % threading_layer())

