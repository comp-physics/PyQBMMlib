from numba import config, njit, threading_layer, prange, objmode
import sys
import numpy as np
import psutil
# import qbmmlib.src.advancer as advancer
# import qbmmlib.src.inversion as inv
# import qbmmlib.src.config_manager as cfg
import qbmmlib.utils.stats_util as stats
import time

def init_batch_input(indices):
    mu = [1.0, 1.0]
    sig = [0.1, 0.1]
    moments = stats.raw_gaussian_moments_bivar(indices,
            mu[0], mu[1],
            sig[0], sig[1])
    return np.asarray(moments)

@njit
def hyqmom2(moments):
    n = 2
    w = np.zeros(n)
    x = np.zeros(n)

    w[0] = moments[0] / 2.0
    w[1] = w[0]

    bx = moments[1] / moments[0]
    d2 = moments[2] / moments[0]
    c2 = d2 - bx ** 2.0

    x[0] = bx - np.sqrt(c2)
    x[1] = bx + np.sqrt(c2)

    return
    # return x, w

@njit
def chyqmom4(moments, indices, max_skewness=30):
    mom00 = moments[0]
    mom10 = moments[1]
    mom01 = moments[2]
    mom20 = moments[3]
    mom11 = moments[4]
    mom02 = moments[5]

    n = 4
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    bx = mom10 / mom00
    by = mom01 / mom00
    d20 = mom20 / mom00
    d11 = mom11 / mom00
    d02 = mom02 / mom00

    c20 = d20 - bx ** 2.0
    c11 = d11 - bx * by
    c02 = d02 - by ** 2.0

    M1 = np.array([1, 0, c20])
    xp, rho = hyqmom2(M1)
    yf = c11 * xp / c20
    mu2avg = c02 - np.sum(rho * yf ** 2)
    mu2avg = max(mu2avg, 0)
    mu2 = mu2avg
    M3 = np.array([1, 0, mu2])
    xp3, rh3 = hyqmom2(M3)
    yp21 = xp3[0]
    yp22 = xp3[1]
    rho21 = rh3[0]
    rho22 = rh3[1]

    w[0] = rho[0] * rho21
    w[1] = rho[0] * rho22
    w[2] = rho[1] * rho21
    w[3] = rho[1] * rho22
    w = mom00 * w

    x[0] = xp[0]
    x[1] = xp[0]
    x[2] = xp[1]
    x[3] = xp[1]
    x = bx + x

    y[0] = yf[0] + yp21
    y[1] = yf[0] + yp22
    y[2] = yf[1] + yp21
    y[3] = yf[1] + yp22
    y = by + y

    return

@njit(parallel=False)
def s_compute_batch(moments, n, indices): 
    for i in range(n): 
        # chyqmom4(moments, indices)
        hyqmom2(moments)
    return

@njit(parallel=True)
def p_compute_batch(moments, n, indices): 
    for i in prange(n): 
        # chyqmom4(moments, indices)
        hyqmom2(moments)
    return



# @njit
@njit(parallel=False,fastmath=True)
def compute_batch(n):
    w = np.zeros((2,n))
    x = np.zeros((2,n))
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    for i in range(n): 
        w[0:1,i] = moments[0] / 2.0
        bx = moments[1] / moments[0]
        d2 = moments[2] / moments[0]
        c = np.sqrt(d2 - np.power(bx,2))
        x[0,i] = bx - c
        x[1,i] = bx + c
    with objmode(time2='f8'):
        time2 = time.perf_counter() - time1
    return time2

if __name__ == "__main__":

    indices = np.array(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    moments = init_batch_input(indices)
    print(moments)
    exit()
    moments = np.array([ 1, 0.0001, 0.01])

    ntests = 4
    ncpus = psutil.cpu_count(logical = False)
    print('It seems that you have %i physical CPU cores and %i tests' % (ncpus, ntests))
    N = int(1e3)

    serial_times = np.zeros(ntests)
    parallel_times = np.zeros(ntests)
    # Kick off one parallel run to get cache 
    # p_compute_batch(moments, N, indices)
    dt = compute_batch(N)
    for i in range(ntests):
        dt = compute_batch(N)
        serial_times[i] = dt

        # t_begin = time.perf_counter()
        # s_compute_batch(moments, N, indices)
        # t_end = time.perf_counter()
        # serial_times[i] = (t_end - t_begin)

        # t_begin = time.perf_counter()
        # p_compute_batch(moments, N, indices)
        # t_end = time.perf_counter()
        # parallel_times[i] = (t_end - t_begin)

    print("Serial Times   [s]",serial_times)
    # print("Parallel Times [s]",parallel_times)

    print("Avg serial time   [s]", np.mean(serial_times))
    # print("Avg parallel time [s]", np.mean(parallel_times))

    # print("Ideal speed up %i" % ncpus)
    # print("Actual speed up", np.mean(serial_times)/np.mean(parallel_times))
    # print("Threading layer chosen: %s" % threading_layer())
