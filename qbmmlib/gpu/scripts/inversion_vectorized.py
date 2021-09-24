from numba import config, njit, threading_layer, prange, objmode
import numpy as np

@njit(parallel=True)
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

@njit(parallel=True)
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

@njit(parallel=True)
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

@njit(parallel=True)
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

    return w, x, y

@njit(parallel=True)
def chyqmom27(moments, N):
    x = np.zeros((N,27))
    y = np.zeros((N,27))
    z = np.zeros((N,27))
    w = np.zeros((N,27))

    M1 = np.zeros((N,5))
    M4 = np.zeros((N,10))
    M5 = np.zeros((N,5))
    xp1 = np.zeros((N,3))
    xp2 = np.zeros((N,3))
    rho0 = np.zeros((N,3))
    rho1 = np.zeros((N,3,3))
    rho2 = np.zeros((N,3))

    yf = np.zeros((N,3))
    zf = np.zeros((N,3,3))

    yp = np.zeros((N,3,3))

    mom000 = np.zeros(N)
    mom100 = np.zeros(N)
    mom010 = np.zeros(N)
    mom001 = np.zeros(N)
    mom200 = np.zeros(N)
    mom110 = np.zeros(N)
    mom101 = np.zeros(N)
    mom020 = np.zeros(N)
    mom011 = np.zeros(N)
    mom002 = np.zeros(N)
    mom300 = np.zeros(N)
    mom030 = np.zeros(N)
    mom003 = np.zeros(N)
    mom400 = np.zeros(N)
    mom040 = np.zeros(N)
    mom004 = np.zeros(N)

    mom000[:] = moments[:,0]
    mom100[:] = moments[:,1]
    mom010[:] = moments[:,2]
    mom001[:] = moments[:,3]
    mom200[:] = moments[:,4]
    mom110[:] = moments[:,5]
    mom101[:] = moments[:,6]
    mom020[:] = moments[:,7]
    mom011[:] = moments[:,8]
    mom002[:] = moments[:,9]
    mom300[:] = moments[:,10]
    mom030[:] = moments[:,11]
    mom003[:] = moments[:,12]
    mom400[:] = moments[:,13]
    mom040[:] = moments[:,14]
    mom004[:] = moments[:,15]

    bx = mom100 / mom000
    by = mom010 / mom000
    bz = mom001 / mom000

    d200 = mom200 / mom000
    d110 = mom110 / mom000
    d101 = mom101 / mom000
    d020 = mom020 / mom000
    d011 = mom011 / mom000
    d002 = mom002 / mom000
    d300 = mom300 / mom000
    d030 = mom030 / mom000
    d003 = mom003 / mom000
    d400 = mom400 / mom000
    d040 = mom040 / mom000
    d004 = mom004 / mom000

    c200 = d200 - bx ** 2
    c110 = d110 - bx * by
    c101 = d101 - bx * bz
    c020 = d020 - by ** 2
    c011 = d011 - by * bz
    c002 = d002 - bz ** 2
    c300 = d300 - 3 * bx * d200 + 2 * bx ** 3
    c030 = d030 - 3 * by * d020 + 2 * by ** 3
    c003 = d003 - 3 * bz * d002 + 2 * bz ** 3
    c400 = d400 - 4 * bx * d300 + 6 * bx ** 2 * d200 - 3 * bx ** 4
    c040 = d040 - 4 * by * d030 + 6 * by ** 2 * d020 - 3 * by ** 4
    c004 = d004 - 4 * bz * d003 + 6 * bz ** 2 * d002 - 3 * bz ** 4

    M1[:,0] = 1.
    M1[:,1] = 0.
    M1[:,2] = c200[:]
    M1[:,3] = c300[:]
    M1[:,4] = c400[:]

    hyqmom3(M1, N, rho0, xp1)

    M4[:,0] = 1.
    M4[:,1] = 0.
    M4[:,2] = 0.
    M4[:,3] = c200[:]
    M4[:,4] = c110[:]
    M4[:,5] = c020[:]
    M4[:,6] = c300[:]
    M4[:,7] = c030[:]
    M4[:,8] = c400[:]
    M4[:,9] = c040[:]

    w4, x4, y4 = chyqmom9(M4, N)

    for i in range(3):
        for j in range(3):
            rho1[:, i, j] = w4[:, (i)*3+j] / w4[:, (i-1)*3 + j]
    
    yf[:, 0] = rho1[:, 0, 0] * y4[:, 0] + rho1[:, 0, 1] * y4[:, 1] + rho1[:, 0, 2] * y4[:, 2]
    yf[:, 1] = rho1[:, 1, 0] * y4[:, 3] + rho1[:, 1, 1] * y4[:, 4] + rho1[:, 1, 2] * y4[:, 5]
    yf[:, 2] = rho1[:, 2, 0] * y4[:, 6] + rho1[:, 2, 1] * y4[:, 7] + rho1[:, 2, 2] * y4[:, 8]

    for i in range(3):
        for j in range(3):
            yp[:, i, j] = y4[:, i*3+j] - yf[:, i]
        
    scale1 = np.sqrt(c200)
    scale2 = np.sqrt(c020)
    c101s = c101/scale1

    zf[:, 0, 0] = xp1[:, 0] /scale1 * c101s
    zf[:, 0, 1] = zf[:, 0, 0]
    zf[:, 0, 2] = zf[:, 0, 0]
    zf[:, 1, 0] = xp1[:, 1] /scale1 * c101s
    zf[:, 1, 1] = zf[:, 1, 0]
    zf[:, 1, 2] = zf[:, 1, 0]
    zf[:, 2, 0] = xp1[:, 2] /scale1 * c101s
    zf[:, 2, 1] = zf[:, 2, 0]
    zf[:, 2, 2] = zf[:, 2, 0]

    RAB = rho1
    RAB[:, 0, 0] *= rho0[:, 0]
    RAB[:, 1, 1] *= rho0[:, 1]
    RAB[:, 2, 2] *= rho0[:, 2]

    # SUM002 = np.sum(np.multiply(RAB, zf**2), axis=(1, 2))
    SUM002 = np.sum(np.multiply(RAB, zf**2), axis=2)
    SUM002 = np.sum(SUM002, axis=1)
    mu2 = c002 - SUM002

    SUM3 = np.sum(np.multiply(RAB, zf**3), axis=2)
    SUM3 = np.sum(SUM3, axis=1)
    SUM1 = mu2 ** (3 / 2)
    q = (c003 - SUM3) / SUM1

    SUM2 = mu2 ** 2
    SUM4 = np.sum(np.multiply(RAB, zf**4), axis=1)
    SUM4 = np.sum(SUM4, axis=1) + 6 * SUM002 * mu2
    eta = (c004 - SUM4) / SUM2
    
    mu3 = q * SUM1
    mu4 = eta * SUM2

    M5[:, 0] = 1.
    M5[:, 1] = 0.
    M5[:, 2] = mu2
    M5[:, 3] = mu3
    M5[:, 4] = mu4
    hyqmom3(M1, N, rho2, xp2)

    w[:, 0] = xp1[:, 0] * rho1[:, 0, 0] * rho2[:, 0] * mom000
    w[:, 1] = xp1[:, 0] * rho1[:, 0, 0] * rho2[:, 1] * mom000
    w[:, 2] = xp1[:, 0] * rho1[:, 0, 0] * rho2[:, 2] * mom000
    w[:, 3] = xp1[:, 0] * rho1[:, 0, 1] * rho2[:, 0] * mom000
    w[:, 4] = xp1[:, 0] * rho1[:, 0, 1] * rho2[:, 1] * mom000
    w[:, 5] = xp1[:, 0] * rho1[:, 0, 1] * rho2[:, 2] * mom000
    w[:, 6] = xp1[:, 0] * rho1[:, 0, 2] * rho2[:, 0] * mom000
    w[:, 7] = xp1[:, 0] * rho1[:, 0, 2] * rho2[:, 1] * mom000
    w[:, 8] = xp1[:, 0] * rho1[:, 0, 2] * rho2[:, 2] * mom000
    w[:, 9] = xp1[:, 1] * rho1[:, 1, 0] * rho2[:, 0] * mom000
    w[:, 10] = xp1[:, 1] * rho1[:, 1, 0] * rho2[:, 1] * mom000
    w[:, 11] = xp1[:, 1] * rho1[:, 1, 0] * rho2[:, 2] * mom000
    w[:, 12] = xp1[:, 1] * rho1[:, 1, 1] * rho2[:, 0] * mom000
    w[:, 13] = xp1[:, 1] * rho1[:, 1, 1] * rho2[:, 1] * mom000
    w[:, 14] = xp1[:, 1] * rho1[:, 1, 1] * rho2[:, 2] * mom000
    w[:, 15] = xp1[:, 1] * rho1[:, 1, 2] * rho2[:, 0] * mom000
    w[:, 16] = xp1[:, 1] * rho1[:, 1, 2] * rho2[:, 1] * mom000
    w[:, 17] = xp1[:, 1] * rho1[:, 1, 2] * rho2[:, 2] * mom000
    w[:, 18] = xp1[:, 2] * rho1[:, 1, 0] * rho2[:, 0] * mom000
    w[:, 19] = xp1[:, 2] * rho1[:, 2, 0] * rho2[:, 1] * mom000
    w[:, 20] = xp1[:, 2] * rho1[:, 2, 0] * rho2[:, 2] * mom000
    w[:, 21] = xp1[:, 2] * rho1[:, 2, 1] * rho2[:, 0] * mom000
    w[:, 22] = xp1[:, 2] * rho1[:, 2, 1] * rho2[:, 1] * mom000
    w[:, 23] = xp1[:, 2] * rho1[:, 2, 1] * rho2[:, 2] * mom000
    w[:, 24] = xp1[:, 2] * rho1[:, 2, 2] * rho2[:, 0] * mom000
    w[:, 25] = xp1[:, 2] * rho1[:, 2, 2] * rho2[:, 1] * mom000
    w[:, 26] = xp1[:, 2] * rho1[:, 2, 2] * rho2[:, 2] * mom000

    x[:, 0]  = xp1[:, 0] + bx
    x[:, 1]  = xp1[:, 0] + bx
    x[:, 2]  = xp1[:, 0] + bx
    x[:, 3]  = xp1[:, 0] + bx
    x[:, 4]  = xp1[:, 0] + bx
    x[:, 5]  = xp1[:, 0] + bx
    x[:, 6]  = xp1[:, 0] + bx
    x[:, 7]  = xp1[:, 0] + bx
    x[:, 8]  = xp1[:, 0] + bx
    x[:, 9]  = xp1[:, 1] + bx
    x[:, 10] = xp1[:, 1] + bx
    x[:, 11] = xp1[:, 1] + bx
    x[:, 12] = xp1[:, 1] + bx
    x[:, 13] = xp1[:, 1] + bx
    x[:, 14] = xp1[:, 1] + bx
    x[:, 15] = xp1[:, 1] + bx
    x[:, 16] = xp1[:, 1] + bx
    x[:, 17] = xp1[:, 1] + bx
    x[:, 18] = xp1[:, 2] + bx
    x[:, 19] = xp1[:, 2] + bx
    x[:, 20] = xp1[:, 2] + bx
    x[:, 21] = xp1[:, 2] + bx
    x[:, 22] = xp1[:, 2] + bx
    x[:, 23] = xp1[:, 2] + bx
    x[:, 24] = xp1[:, 2] + bx
    x[:, 25] = xp1[:, 2] + bx
    x[:, 26] = xp1[:, 2] + bx

    y[:, 0]  = yf[:, 0] + yp[:, 0, 0] + by 
    y[:, 1]  = yf[:, 0] + yp[:, 0, 0] + by
    y[:, 2]  = yf[:, 0] + yp[:, 0, 0] + by
    y[:, 3]  = yf[:, 0] + yp[:, 0, 1] + by
    y[:, 4]  = yf[:, 0] + yp[:, 0, 1] + by
    y[:, 5]  = yf[:, 0] + yp[:, 0, 1] + by
    y[:, 6]  = yf[:, 0] + yp[:, 0, 2] + by
    y[:, 7]  = yf[:, 0] + yp[:, 0, 2] + by
    y[:, 8]  = yf[:, 0] + yp[:, 0, 2] + by
    y[:, 9]  = yf[:, 1] + yp[:, 1, 0] + by
    y[:, 10] = yf[:, 1] + yp[:, 1, 0] + by
    y[:, 11] = yf[:, 1] + yp[:, 1, 0] + by
    y[:, 12] = yf[:, 1] + yp[:, 1, 1] + by
    y[:, 13] = yf[:, 1] + yp[:, 1, 1] + by
    y[:, 14] = yf[:, 1] + yp[:, 1, 1] + by
    y[:, 15] = yf[:, 1] + yp[:, 1, 2] + by
    y[:, 16] = yf[:, 1] + yp[:, 1, 2] + by
    y[:, 17] = yf[:, 1] + yp[:, 1, 2] + by
    y[:, 18] = yf[:, 2] + yp[:, 2, 0] + by
    y[:, 19] = yf[:, 2] + yp[:, 2, 0] + by
    y[:, 20] = yf[:, 2] + yp[:, 2, 0] + by
    y[:, 21] = yf[:, 2] + yp[:, 2, 1] + by
    y[:, 22] = yf[:, 2] + yp[:, 2, 1] + by
    y[:, 23] = yf[:, 2] + yp[:, 2, 1] + by
    y[:, 24] = yf[:, 2] + yp[:, 2, 2] + by
    y[:, 25] = yf[:, 2] + yp[:, 2, 2] + by
    y[:, 26] = yf[:, 2] + yp[:, 2, 2] + by

    z[:, 0]  = zf[:, 0, 0] + xp2[:,0] + bz 
    z[:, 1]  = zf[:, 0, 1] + xp2[:,1] + bz
    z[:, 2]  = zf[:, 0, 2] + xp2[:,2] + bz
    z[:, 3]  = zf[:, 1, 0] + xp2[:,0] + bz
    z[:, 4]  = zf[:, 1, 1] + xp2[:,1] + bz
    z[:, 5]  = zf[:, 1, 2] + xp2[:,2] + bz
    z[:, 6]  = zf[:, 2, 0] + xp2[:,0] + bz
    z[:, 7]  = zf[:, 2, 1] + xp2[:,1] + bz
    z[:, 8]  = zf[:, 2, 2] + xp2[:,2] + bz
    z[:, 9]  = zf[:, 0, 0] + xp2[:,0] + bz
    z[:, 10] = zf[:, 0, 1] + xp2[:,1] + bz
    z[:, 11] = zf[:, 0, 2] + xp2[:,2] + bz
    z[:, 12] = zf[:, 1, 0] + xp2[:,0] + bz
    z[:, 13] = zf[:, 1, 1] + xp2[:,1] + bz
    z[:, 14] = zf[:, 1, 2] + xp2[:,2] + bz
    z[:, 15] = zf[:, 2, 0] + xp2[:,0] + bz
    z[:, 16] = zf[:, 2, 1] + xp2[:,1] + bz
    z[:, 17] = zf[:, 2, 2] + xp2[:,2] + bz
    z[:, 18] = zf[:, 0, 0] + xp2[:,0] + bz
    z[:, 19] = zf[:, 0, 1] + xp2[:,1] + bz
    z[:, 20] = zf[:, 0, 2] + xp2[:,2] + bz
    z[:, 21] = zf[:, 1, 0] + xp2[:,0] + bz
    z[:, 22] = zf[:, 1, 1] + xp2[:,1] + bz
    z[:, 23] = zf[:, 1, 2] + xp2[:,2] + bz
    z[:, 24] = zf[:, 2, 0] + xp2[:,0] + bz
    z[:, 25] = zf[:, 2, 1] + xp2[:,1] + bz
    z[:, 26] = zf[:, 2, 2] + xp2[:,2] + bz

    return w, x, y, z

    










    