from numba import config, njit, threading_layer, prange, objmode
import numpy as np

@njit
def hyqmom2(moments, N, w, x):
    bx = np.zeros(N)
    d2 = np.zeros(N)
    c = np.zeros(N)

    w[:,0] = moments[:,0] / 2.0
    w[:,1] = w[:,0]

    bx = moments[:,1] / moments[:,0]
    d2 = moments[:,2] / moments[:,0]
    c = np.sqrt(d2 - bx ** 2.0)

    x[:,0] = bx[:] - c[:]
    x[:,1] = bx[:] + c[:]

@njit
def hyqmom3(moments, N, w, x):
    w = np.zeros((N,3))
    x = np.zeros((N,3))
    xp = np.zeros((N,3))
    xps = np.zeros((N,3))
    rho = np.zeros((N,3))

    bx = np.zeros(N)
    d2 = np.zeros(N)
    d3 = np.zeros(N)
    d4 = np.zeros(N)

    bx[:] = moments[:,1] / moments[:,0]
    d2[:] = moments[:,2] / moments[:,0]
    d3[:] = moments[:,3] / moments[:,0]
    d4[:] = moments[:,4] / moments[:,0]

    c2 = d2 - bx ** 2
    c3 = d3 - 3 * bx * d2 + 2 * bx ** 3
    c4 = d4 - 4 * bx * d3 + 6 * (bx ** 2) * d2 - 3 * bx ** 4

    scale = np.sqrt(c2)
    q = c3 / scale / c2
    eta = c4 / c2 / c2

    term = np.sqrt(4 * eta - 3 * q ** 2)
    xps[:,0] = (q[:] - term[:]) / 2.0
    xps[:,1] = 0.0
    xps[:,2] = (q[:] + term[:]) / 2.0

    dem = 1.0 / term
    prod = -xps[:,0] * xps[:,2]
    # prod = max(prod, 1 + realsmall)

    rho[:,0] = -dem[:] / xps[:,0]
    rho[:,1] = - 1. / prod[:] + 1.
    rho[:,2] = dem[:] / xps[:,2]

    srho = rho[:,0] + rho[:,1] + rho[:,2]
    for i in range(3):
        rho[:,i] = rho[:,i] / srho[:]

    num = (
          rho[:,0] * xps[:,0] ** 2 + 
          rho[:,1] * xps[:,1] ** 2 + 
          rho[:,2] * xps[:,2] ** 2
          )
    denom = rho[:,0] + rho[:,1] + rho[:,2]
    scales = num / denom
    for i in range(3):
        xp[:,i] = xps[:,i] * scale[:] / np.sqrt(scales[:])

    for i in range(3):
        w[:,i] = moments[:,0] * rho[:,i]
        x[:,i] = xp[:,i] + bx[:]

@njit
def chyqmom4(moments, N):
    x = np.zeros((N,4))
    y = np.zeros((N,4))
    w = np.zeros((N,4))

    M1 = np.zeros((N,3))
    xp = np.zeros((N,2))
    rho = np.zeros((N,2))

    yf = np.zeros((N,2))

    M3 = np.zeros((N,3))
    xp3 = np.zeros((N,2))
    rh3 = np.zeros((N,2))

    mom00 = np.zeros(N)
    mom10 = np.zeros(N)
    mom01 = np.zeros(N)
    mom20 = np.zeros(N)
    mom11 = np.zeros(N)
    mom02 = np.zeros(N)
    mu2avg = np.zeros(N)

    yp21 = np.zeros(N)
    yp22 = np.zeros(N)
    rho21 = np.zeros(N)
    rho22 = np.zeros(N)

    mom00[:] = moments[:,0]
    mom10[:] = moments[:,1]
    mom01[:] = moments[:,2]
    mom20[:] = moments[:,3]
    mom11[:] = moments[:,4]
    mom02[:] = moments[:,5]

    bx = mom10 / mom00
    by = mom01 / mom00
    d20 = mom20 / mom00
    d11 = mom11 / mom00
    d02 = mom02 / mom00

    c20 = d20 - bx ** 2.0
    c11 = d11 - bx * by
    c02 = d02 - by ** 2.0

    M1[:,0] = 1.
    M1[:,1] = 0.
    M1[:,2] = c20[:]
    hyqmom2(M1, N, rho, xp)

    for i in range(2):
        yf[:,i] = c11[:] * xp[:,i] / c20[:]

    mu2avg[:] = c02[:] - (
                    rho[:,0] * yf[:,0] ** 2.0 + 
                    rho[:,1] * yf[:,1] ** 2.0
                )

    M3[:,0] = 1.
    M3[:,1] = 0.
    M3[:,2] = mu2avg[:]
    hyqmom2(M3, N, rh3, xp3)

    yp21[:] = xp3[:,0]
    yp22[:] = xp3[:,1]
    rho21[:] = rh3[:,0]
    rho22[:] = rh3[:,1]

    x[:,0] = bx[:] + xp[:,0]
    x[:,1] = bx[:] + xp[:,0]
    x[:,2] = bx[:] + xp[:,1]
    x[:,3] = bx[:] + xp[:,1]

    y[:,0] = by[:] + yf[:,0] + yp21[:]
    y[:,1] = by[:] + yf[:,0] + yp22[:]
    y[:,2] = by[:] + yf[:,1] + yp21[:]
    y[:,3] = by[:] + yf[:,1] + yp22[:]

    w[:,0] = mom00[:] * rho[:,0] * rho21[:]
    w[:,1] = mom00[:] * rho[:,0] * rho22[:]
    w[:,2] = mom00[:] * rho[:,1] * rho21[:]
    w[:,3] = mom00[:] * rho[:,1] * rho22[:]


@njit
def chyqmom9(moments, N):
    x = np.zeros((N,9))
    y = np.zeros((N,9))
    w = np.zeros((N,9))

    M1 = np.zeros((N,5))
    xp = np.zeros((N,3))
    rho = np.zeros((N,3))

    yf = np.zeros((N,3))

    M3 = np.zeros((N,5))
    xp3 = np.zeros((N,3))
    rh3 = np.zeros((N,3))

    mom00 = np.zeros(N)
    mom10 = np.zeros(N)
    mom01 = np.zeros(N)
    mom20 = np.zeros(N)
    mom11 = np.zeros(N)
    mom02 = np.zeros(N)
    mom30 = np.zeros(N)
    mom03 = np.zeros(N)
    mom40 = np.zeros(N)
    mom04 = np.zeros(N)
    mu2avg = np.zeros(N)

    yp21 = np.zeros(N)
    yp22 = np.zeros(N)
    rho21 = np.zeros(N)
    rho22 = np.zeros(N)

    mom00[:] = moments[:,0]
    mom10[:] = moments[:,1]
    mom01[:] = moments[:,2]
    mom20[:] = moments[:,3]
    mom11[:] = moments[:,4]
    mom02[:] = moments[:,5]
    mom30[:] = moments[:,6]
    mom03[:] = moments[:,7]
    mom40[:] = moments[:,8]
    mom04[:] = moments[:,9]



    bx = mom10 / mom00
    by = mom01 / mom00
    d20 = mom20 / mom00
    d11 = mom11 / mom00
    d02 = mom02 / mom00
    d30 = mom30 / mom00
    d03 = mom03 / mom00
    d40 = mom40 / mom00
    d04 = mom04 / mom00

    c20 = d20 - bx ** 2.0
    c11 = d11 - bx * by
    c02 = d02 - by ** 2.0
    c30 = d30 - 3.0 * bx * d20 + 2.0 * bx ** 3.0
    c03 = d03 - 3.0 * by * d02 + 2.0 * by ** 3.0
    c40 = d40 - 4.0 * bx * d30 + 6 * (bx ** 2.0) * d20 - 3.0 * bx ** (4)
    c04 = d04 - 4.0 * by * d03 + 6 * (by ** 2.0) * d02 - 3.0 * by ** (4)

    M1[:,0] = 1.
    M1[:,1] = 0.
    M1[:,2] = c20[:]
    M1[:,3] = c30[:]
    M1[:,4] = c40[:]
    hyqmom3(M1, N, rho, xp)

    # ----------------
    for i in range(3):
        yf[:,i] = c11[:] * xp[:,i] / c20[:]


    ry2 = (
        rho[:,0] * yf[:,0] ** 2.0 +
        rho[:,1] * yf[:,1] ** 2.0 +
        rho[:,2] * yf[:,2] ** 2.0 )
    mu2avg = c02 - ry2 
    # mu2avg = max(mu2avg, 0.0)

    mu2 = mu2avg
    mu3 = 0. * mu2
    mu4 = mu2 ** 2.0

    ry3 = (
        rho[:,0] * yf[:,0] ** 3.0 +
        rho[:,1] * yf[:,1] ** 3.0 +
        rho[:,2] * yf[:,2] ** 3.0 )
    q = (c03 - ry3) / mu2 ** (3.0 / 2.0)

    ry4 = (
        rho[:,0] * yf[:,0] ** 4.0 +
        rho[:,1] * yf[:,1] ** 4.0 +
        rho[:,2] * yf[:,2] ** 4.0 )
    eta = (c04 - ry4 - 6. * ry2 * mu2) / mu2 ** 2.0
    mu3 = q * mu2 ** (3.0 / 2.0)
    mu4 = eta * mu2 ** 2.0

    M3[:,0] = 1.
    M3[:,1] = 0.
    M3[:,2] = mu2[:]
    M3[:,3] = mu3[:]
    M3[:,4] = mu4[:]
    hyqmom3(M3, N, rh3, xp3)

    yp21 = xp3[:,0]
    yp22 = xp3[:,1]
    yp23 = xp3[:,2]
    rho21 = rh3[:,0]
    rho22 = rh3[:,1]
    rho23 = rh3[:,2]

    w[:,0] = mom00[:] * rho[:,0] * rho21[:]
    w[:,1] = mom00[:] * rho[:,0] * rho22[:]
    w[:,2] = mom00[:] * rho[:,0] * rho23[:]
    w[:,3] = mom00[:] * rho[:,1] * rho21[:]
    w[:,4] = mom00[:] * rho[:,1] * rho22[:]
    w[:,5] = mom00[:] * rho[:,1] * rho23[:]
    w[:,6] = mom00[:] * rho[:,2] * rho21[:]
    w[:,7] = mom00[:] * rho[:,2] * rho22[:]
    w[:,8] = mom00[:] * rho[:,2] * rho23[:]

    x[:,0] = bx[:] + xp[:,0]
    x[:,1] = bx[:] + xp[:,0]
    x[:,2] = bx[:] + xp[:,0]
    x[:,3] = bx[:] + xp[:,1]
    x[:,4] = bx[:] + xp[:,1]
    x[:,5] = bx[:] + xp[:,1]
    x[:,6] = bx[:] + xp[:,2]
    x[:,7] = bx[:] + xp[:,2]
    x[:,8] = bx[:] + xp[:,2]

    y[:,0] = by[:] + yf[:,0] + yp21[:]
    y[:,1] = by[:] + yf[:,0] + yp22[:]
    y[:,2] = by[:] + yf[:,0] + yp23[:]
    y[:,3] = by[:] + yf[:,1] + yp21[:]
    y[:,4] = by[:] + yf[:,1] + yp22[:]
    y[:,5] = by[:] + yf[:,1] + yp23[:]
    y[:,6] = by[:] + yf[:,2] + yp21[:]
    y[:,7] = by[:] + yf[:,2] + yp22[:]
    y[:,8] = by[:] + yf[:,2] + yp23[:]

