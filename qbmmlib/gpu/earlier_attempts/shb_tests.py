from numba import config, njit, threading_layer, prange, objmode
import sys
import numpy as np
import psutil
# import qbmmlib.src.advancer as advancer
# import qbmmlib.src.inversion as inv
# import qbmmlib.src.config_manager as cfg
import qbmmlib.utils.stats_util as stats
import time
# import cProfile
# import pstats
# from pstats import SortKey
import random

# from meliae import scanner
# scanner.dump_all_objects( 'blah.dat' ) # you can pass a file-handle if you prefer


def init_batch_input(indices):
    mu = [1.0, 1.0]
    sig = [0.1, 0.1]
    moments = stats.raw_gaussian_moments_bivar(indices,
            mu[0], mu[1],
            sig[0], sig[1])
    return np.asarray(moments)

@njit
def hyqmom2(moments,x,w):
    # n = 2
    # w = np.zeros(n)
    # x = np.zeros(n)
    w[0] = moments[0] / 2.0
    w[1] = w[0]

    bx = moments[1] / moments[0]
    d2 = moments[2] / moments[0]
    c2 = d2 - bx ** 2.0

    x[0] = bx - np.sqrt(c2)
    x[1] = bx + np.sqrt(c2)

    # return x, w
    return


# @njit
def chyqmom4_p(moments, N):
    x = np.zeros(4)
    y = np.zeros(4)
    w = np.zeros(4)

    M1 = np.zeros(3)
    xp = np.zeros(2)
    rho = np.zeros(2)

    M3 = np.zeros(3)
    xp3 = np.zeros(2)
    rh3 = np.zeros(2)

    yf = np.zeros(2)

    for i in range(N):
        mom00 = moments[0]
        mom10 = moments[1]
        mom01 = moments[2]
        mom20 = moments[3]
        mom11 = moments[4]
        mom02 = moments[5]

        bx = mom10 / mom00
        by = mom01 / mom00
        d20 = mom20 / mom00
        d11 = mom11 / mom00
        d02 = mom02 / mom00

        c20 = d20 - bx ** 2.0
        c11 = d11 - bx * by
        c02 = d02 - by ** 2.0

        M1[0] = 1. 
        M1[1] = 0.
        M1[2] = c20

        hyqmom2(M1, xp, rho)
        print(c11,c20)
        quit()
        # yf[:] = c11 * xp[:] / c20
        # yf = 2.*xp


        # mu2avg = c02 - np.sum(rho * yf ** 2)
        # if mu2avg < 0:
        #     mu2avg = 0.
        # mu2 = mu2avg

        # M3[0] = 1.
        # M3[1] = 0.
        # M3[2] = mu2
        # hyqmom2(M3, xp3, rh3)

        # yp21 = xp3[0]
        # yp22 = xp3[1]
        # rho21 = rh3[0]
        # rho22 = rh3[1]

        # x[i,0] = xp[0]
        # x[i,1] = xp[0]
        # x[i,2] = xp[1]
        # x[i,3] = xp[1]
        # x = bx + x

        # y[i,0] = yf[0] + yp21
        # y[i,1] = yf[0] + yp22
        # y[i,2] = yf[1] + yp21
        # y[i,3] = yf[1] + yp22
        # y = by + y

        # w[i,0] = rho[0] * rho21
        # w[i,1] = rho[0] * rho22
        # w[i,2] = rho[1] * rho21
        # w[i,3] = rho[1] * rho22
        # w = mom00 * w

    return

@njit
def chyqmom4(moments, x, y, w, i):
    mom00 = moments[0]
    mom10 = moments[1]
    mom01 = moments[2]
    mom20 = moments[3]
    mom11 = moments[4]
    mom02 = moments[5]

    bx = mom10 / mom00
    by = mom01 / mom00
    d20 = mom20 / mom00
    d11 = mom11 / mom00
    d02 = mom02 / mom00

    c20 = d20 - bx ** 2.0
    c11 = d11 - bx * by
    c02 = d02 - by ** 2.0

    M1 = np.array([1, 0, c20])
    xp = np.zeros(2)
    rho = np.zeros(2)

    # hyqmom2(M1, xp, rho)
    # yf = c11 * xp / c20
    # mu2avg = c02 - np.sum(rho * yf ** 2)
    # if mu2avg < 0:
    #     mu2avg = 0.
    # mu2 = mu2avg

    # M3 = np.array([1, 0, mu2])
    # xp3 = np.zeros(2)
    # rh3 = np.zeros(2)
    # hyqmom2(M3, xp3, rh3)

    # yp21 = xp3[0]
    # yp22 = xp3[1]
    # rho21 = rh3[0]
    # rho22 = rh3[1]

    # x[i,0] = xp[0]
    # x[i,1] = xp[0]
    # x[i,2] = xp[1]
    # x[i,3] = xp[1]
    # x = bx + x

    # y[i,0] = yf[0] + yp21
    # y[i,1] = yf[0] + yp22
    # y[i,2] = yf[1] + yp21
    # y[i,3] = yf[1] + yp22
    # y = by + y

    # w[i,0] = rho[0] * rho21
    # w[i,1] = rho[0] * rho22
    # w[i,2] = rho[1] * rho21
    # w[i,3] = rho[1] * rho22
    # w = mom00 * w

    return

@njit(parallel=False)
def s_compute_batch(moments, n): 
    x = np.zeros(2)
    w = np.zeros(2)
    for i in range(n): 
        # chyqmom4(moments)
        hyqmom2(moments,x,w)
    return

@njit(parallel=True)
def p_compute_batch(moments, n): 
    x = np.zeros(2)
    w = np.zeros(2)
    for i in prange(n): 
        # chyqmom4(moments)
        hyqmom2(moments,x,w)
    return


@njit(parallel=False)
def s_compute_batch_c(moments, n, x, y, w): 
    for i in range(n):
        chyqmom4(moments, x, y, w, i)
    return


@njit(parallel=True)
def p_compute_batch_c(moments, n, x, y, w): 
    for i in prange(n):
        chyqmom4(moments, x, y, w, i)
    return


# @njit(parallel=False,fastmath=True)
# def compute_batch(n):
#     w = np.zeros((2,n))
#     x = np.zeros((2,n))
#     with objmode(time1='f8'):
#         time1 = time.perf_counter()
#     for i in range(n): 
#         w[0:1,i] = moments[0] / 2.0
#         bx = moments[1] / moments[0]
#         d2 = moments[2] / moments[0]
#         c = np.sqrt(d2 - np.power(bx,2))
#         x[0,i] = bx - c
#         x[1,i] = bx + c
#     with objmode(time2='f8'):
#         time2 = time.perf_counter() - time1
#     return time2

if __name__ == "__main__":

    indices = np.array(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    moments = init_batch_input(indices)
    moments = moments * (1. + random.random()*1)
    # moments = np.array([ 1, 0.0001, 0.01])
    config.THREADING_LAYER = 'workqueue'
    print('moments = ',moments)

    ntests = 5
    ncpus = psutil.cpu_count(logical = False)
    print('It seems that you have %i physical CPU cores and %i tests' % (ncpus, ntests))
    N = int(1e5)

    serial_times = np.zeros(ntests)
    parallel_times = np.zeros(ntests)
    for i in range(ntests):
        # x = np.zeros((N,4))
        # y = np.zeros((N,4))
        # w = np.zeros((N,4))
        t_begin = time.process_time()
        # t_begin = time.perf_counter()
        # s_compute_batch(moments, N)
        # s_compute_batch_c(moments, N, x, y, w)
        chyqmom4_p(moments, N)
        # t_end = time.perf_counter()
        t_end = time.process_time()
        serial_times[i] = (t_end - t_begin)

        # x = np.zeros((N,4))
        # y = np.zeros((N,4))
        # w = np.zeros((N,4))
        # t_begin = time.perf_counter()
        # # p_compute_batch(moments, N)
        # p_compute_batch_c(moments, N, x, y, w)
        # t_end = time.perf_counter()
        # parallel_times[i] = (t_end - t_begin)

    # cProfile.run('chyqmom4_p(moments, N)','restats')
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(-1).print_stats(10)


    print("Serial Times   [s]",serial_times)
    print("Parallel Times [s]",parallel_times)

    print("Min serial time   [s]", np.min(serial_times))
    print("Min parallel time [s]", np.min(parallel_times))

    # print("Ideal speed up %i" % ncpus)
    # print("Actual speed up", np.min(serial_times)/np.min(parallel_times))
    # print("Threading layer chosen: %s" % threading_layer())
