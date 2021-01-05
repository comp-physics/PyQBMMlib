from numba import config, njit, threading_layer, prange
import numpy as np
import psutil
import time

@njit(parallel=False)
def f(x):
    return np.exp(x)

@njit(parallel=False)
def range_deriv(f,dx,nx):
    df = np.zeros(nx)
    for i in range(nx-2):
        df[i+1] = (f[i+2]-f[i])/(2.*dx)
    df[0] = (f[1]-f[0])/dx
    df[nx-1] = (f[nx-1]-f[nx-2])/dx
    return df

@njit(parallel=True)
def prange_deriv(f,dx,nx):
    df = np.zeros(nx)
    for i in prange(nx-2):
        df[i+1] = (f[i+2]-f[i])/(2.*dx)
    df[0] = (f[1]-f[0])/dx
    df[nx-1] = (f[nx-1]-f[nx-2])/dx
    return df

if __name__ == "__main__":
    ntests = 5
    ncpus = psutil.cpu_count(logical = False)
    nx = int(1e9)
    a = 0.
    b = 1.
    x = np.linspace(a,b,nx)
    y = f(x)
    dx = (b-a)/float(nx)

    serial_times = np.zeros(ntests)
    parallel_times = np.zeros(ntests)
    t_start = time.perf_counter()
    prange_deriv(y,dx,nx)
    t_end = time.perf_counter()
    for i in range(ntests):
        t_start = time.perf_counter()
        dy_s = range_deriv(y,dx,nx)    
        t_end = time.perf_counter()
        serial_times[i] = (t_end - t_start)

        t_start = time.perf_counter()
        dy_p = prange_deriv(y,dx,nx)    
        t_end = time.perf_counter()
        parallel_times[i] = (t_end - t_start)
    
    np.testing.assert_almost_equal(dy_s,dy_p)

    print("Serial Times   [s]",serial_times)
    print("Parallel Times [s]",parallel_times)

    print("Avg serial time   [s]", np.mean(serial_times))
    print("Avg parallel time [s]", np.mean(parallel_times))

    print("Perfect speed up would be be:", np.mean(serial_times)/float(ncpus))
    print("Threading layer chosen: %s" % threading_layer())

