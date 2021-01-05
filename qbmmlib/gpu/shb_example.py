from numba import njit
import sys
import numpy as np
import qbmmlib.utils.stats_util as stats
import time
import random
import matplotlib.pyplot as plt

def init_batch_input(indices):
    mu = [1.0, 1.0]
    sig = [0.1, 0.1]
    moments = stats.raw_gaussian_moments_bivar(indices,
            mu[0], mu[1],
            sig[0], sig[1])
    return np.asarray(moments)

# @njit(fastmath=True)
def hyqmom2(m,x,w):
    w[0] = m[0] / 2.0
    w[1] = w[0]

    bx = m[1] / m[0]
    d2 = m[2] / m[0]
    c = np.sqrt(d2 - bx ** 2.0)

    x[0] = bx - c
    x[1] = bx + c

    return

# @profile
# @njit(fastmath=True)
def myfunc_1d(mom2d,N):
    M1 = mom2d
    xp = np.zeros(2)
    rho = np.zeros(2)
    for i in range(N):
        hyqmom2(M1, xp, rho)
    return


# @profile
@njit(fastmath=True)
def myfunc_2d(mom2d,N):
    mom00 = mom2d[0]
    mom10 = mom2d[1]
    mom01 = mom2d[2]
    mom20 = mom2d[3]
    mom11 = mom2d[4]
    mom02 = mom2d[5]

    x = np.zeros(4)
    y = np.zeros(4)
    w = np.zeros(4)

    M1  = np.zeros(3)
    xp  = np.zeros(2)
    rho = np.zeros(2)

    M3  = np.zeros(3)
    xp3 = np.zeros(2)
    rh3 = np.zeros(2)

    for i in range(N):
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
        # --------------------
        # Do Hyqmom2 in place
        # hyqmom2(M1, xp, rho)
        rho[0] = M1[0] / 2.0
        rho[1] = rho[0]
        bx2 = M1[1] / M1[0]
        d22 = M1[2] / M1[0]
        c = np.sqrt(d22 - bx2 ** 2.0)
        xp[0] = bx2 - c
        xp[1] = bx2 + c
        # --------------------

        yf = c11 * xp / c20

        mu2avg = c02 
        for j in range(2):
            mu2avg -= rho[j] * (yf[j]**2.)

        if mu2avg < 0:
            mu2avg = 1.e-14
        mu2 = mu2avg

        M3[0] = 1.
        M3[1] = 0.
        M3[2] = mu2
        # --------------------
        # Do Hyqmom2 in place
        # hyqmom2(M3, xp3, rh3)
        rh3[0] = M3[0] / 2.0
        rh3[1] = rh3[0]
        bx2 = M3[1] / M3[0]
        d22 = M3[2] / M3[0]
        c = np.sqrt(d22 - bx2 ** 2.0)
        xp3[0] = bx2 - c
        xp3[1] = bx2 + c
        # --------------------

        yp21 = xp3[0]
        yp22 = xp3[1]
        rho21 = rh3[0]
        rho22 = rh3[1]

        x[0] = bx + xp[0]
        x[1] = bx + xp[0]
        x[2] = bx + xp[1]
        x[3] = bx + xp[1]

        y[0] = by + yf[0] + yp21
        y[1] = by + yf[0] + yp22
        y[2] = by + yf[1] + yp21
        y[3] = by + yf[1] + yp22

        w[0] = mom00 * rho[0] * rho21
        w[1] = mom00 * rho[0] * rho22
        w[2] = mom00 * rho[1] * rho21
        w[3] = mom00 * rho[1] * rho22

    return

if __name__ == "__main__":
    indices = np.array(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    mom1d = np.array([1.,0.001,0.03])
    mom2d = init_batch_input(indices)

    run = ''
    # run = 'profile'

    if ( run == 'profile'):
        import psutil
        import cProfile
        import pstats
        from pstats import SortKey

        N = int(1e4)
        # myfunc_1d(mom1d,N)
        myfunc_2d(mom2d,N)
        myfunc_2d(mom2d,N)
        # cProfile.run('myfunc_2d(mom2d, N)','restats')
        # p = pstats.Stats('restats')
        # p.dump_stats('profile.prof')
        # p.strip_dirs().sort_stats(-1).print_stats(10)
    else:
        ns = np.logspace(2,5,num=4).astype(int)
        my_times = np.zeros((len(ns),2))
        for j,N in enumerate(ns):
            ntests=5
            times = np.zeros(ntests)
            for i in range(ntests):
                t_begin = time.perf_counter()
                # myfunc_1d(mom1d,N)
                myfunc_2d(mom2d,N)
                t_end = time.perf_counter()
                times[i] = (t_end - t_begin)

            my_times[j,:] = [N, np.min(times)]
            # my_times[j,:] = [N, np.min(times)/float(N)]

        print("Full times [s]",my_times)
        
        # plt.loglog(my_times[:,0],my_times[:,1])
        # plt.xlabel('N')
        # plt.ylabel('Time/N')
        # plt.show()


