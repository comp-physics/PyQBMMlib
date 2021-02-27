import numpy as np
import sys
import time
import psutil
import qbmmlib.utils.stats_util as stats
from numba import njit
from inversion_vectorized import hyqmom2, hyqmom3, chyqmom4, chyqmom9

# from numba import config, threading_layer, prange, objmode
# import cProfile
# import pstats
# from pstats import SortKey
# import random
# from meliae import scanner
# scanner.dump_all_objects( 'blah.dat' ) # you can pass a file-handle if you prefer


def init_batch_input(indices):
    mu = [1.0, 1.0]
    sig = [0.1, 0.1]
    moments = stats.raw_gaussian_moments_bivar(indices,
            mu[0], mu[1],
            sig[0], sig[1])
    return np.asarray(moments)

if __name__ == "__main__":

    # oned = True
    # twod = False

    oned = False
    twod = True

    Npt = 3

    if oned:
        print('Algorithm: HyQMOM%i ' % (Npt))
    if twod:
        print('Algorithm: CHyQMOM%i ' % (Npt**2))

    Ninputs = int(1e6)
    Ntests = 3
    if Ntests <= 1: raise ValueError('Ntests > 1 required due to Numba compilation')
    print('You are running %i test iterations on %i inputs' % (Ntests-1, Ninputs))

    # config.THREADING_LAYER = 'workqueue'
    # ncpus = psutil.cpu_count(logical = False)
    # print('You have %i cores' % (ncpus))

    if oned:
        #HyQMOM2 and 3
        if Npt == 2:
            # I think this represents a log-normal distribution of some kind
            moments = np.array(
                    [ 1, 0.0001, 0.01])
        elif Npt == 3:
            # Normal distribution (1.1, 0.1)
            moments = np.array(
                    [ 1, 1.1, 1.22, 1.364, 1.537])
    elif twod:
        #CHyQMOM4 and 9
        if Npt == 2:
            indices = np.array(
                    [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
            moments = init_batch_input(indices)
        if Npt == 3:
            indices = np.array(
                    [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], 
                     [3, 0], [0, 3], [4, 0], [0, 4]])
            moments = init_batch_input(indices)

    moments_full = np.zeros((Ninputs,len(moments)))
    for i in range(len(moments)):
        moments_full[:,i] = moments[i]

    times = np.zeros(Ntests)
    for i in range(Ntests):
        t_begin = time.process_time()
        if oned:
            w = np.zeros((Ninputs,Npt))
            x = np.zeros((Ninputs,Npt))
            if Npt == 2:
                hyqmom2(moments_full,Ninputs,w,x)
            elif Npt == 3:
                hyqmom3(moments_full,Ninputs,w,x)
        if twod:
            if Npt == 2:
                chyqmom4(moments_full,Ninputs)
            if Npt == 3:
                chyqmom9(moments_full,Ninputs)
        t_end = time.process_time()
        times[i] = t_end - t_begin

    print("Times   [s]", times[1:])
    print("Min time   [s]", np.min(times[1:]))

    # cProfile.run('chyqmom4_p(moments, N)','restats')
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(-1).print_stats(10)

    # print("Ideal speed up %i" % ncpus)
    # print("Actual speed up", np.min(serial_times)/np.min(parallel_times))
    # print("Threading layer chosen: %s" % threading_layer())
