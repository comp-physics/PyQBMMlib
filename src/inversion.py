import numpy as np
import math

from numba import njit

@njit
def sign(q):
    if q > 0:
        return 1
    elif q == 0:
        return 0
    else:
        return -1


def wheeler(moments, adaptive=False):
    """
    This function inverts moments into 1D quadrature weights and abscissas using adaptive Wheeler algorithm.

    :param moments: Statistical moments of the transported PDF
    :type moments: array like
    :return: Abscissas, weights
    :rtype: array like
    """

    n = len(moments) // 2

    # Adaptivity parameters
    rmax = 1e-8
    eabs = 1e-8
    cutoff = 0

    # Check if moments are unrealizable.
    if moments[0] <= 0:
        print("Wheeler: Moments are NOT realizable (moment[0] <= 0.0). Run failed.")
        exit()

    if n == 1 or (adaptive and moments[0] < rmax):
        w = moments[0]
        x = moments[1] / moments[0]
        return x, w

    # Set modified moments equal to input moments.
    nu = moments

    # Construct recurrence matrix
    ind = n
    a = np.zeros(n)
    b = np.zeros(n)
    sigma = np.zeros((2 * ind + 1, 2 * ind + 1))

    for i in range(1, 2 * ind + 1):
        sigma[1, i] = nu[i - 1]

    a[0] = nu[1] / nu[0]
    b[0] = 0

    for k in range(2, ind + 1):
        for l in range(k, 2 * ind - k + 2):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - a[k - 2] * sigma[k - 1, l]
                - b[k - 2] * sigma[k - 2, l]
            )
        a[k - 1] = sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
        b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Find maximum n using diagonal element of sigma
    if adaptive:
        for k in range(ind, 1, -1):
            if sigma[k, k] <= cutoff:
                n = k - 1
                if n == 1:
                    w = moments[0]
                    x = moments[1] / moments[0]
                    return x, w

        # Use maximum n to re-calculate recurrence matrix
        a = np.zeros(n)
        b = np.zeros(n)
        w = np.zeros(n)
        x = np.zeros(n)
        sigma = np.zeros((2 * n + 1, 2 * n + 1))
        sigma[1, 1:] = nu

        a[0] = nu[1] / nu[0]
        b[0] = 0
        for k in range(2, n + 1):
            for l in range(k, 2 * n - k + 2):
                sigma[k, l] = (
                    sigma[k - 1, l + 1]
                    - a[k - 2] * sigma[k - 1, l]
                    - b[k - 2] * sigma[k - 2, l]
                )
            a[k - 1] = (
                sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Check if moments are unrealizable
    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! Program exits.")
        exit()

    # Setup Jacobi matrix for n-point quadrature, adapt n using rmin and eabs
    for n1 in range(n, 0, -1):
        if n1 == 1:
            w = moments[0]
            x = moments[1] / moments[0]
            return x, w

        # Jacobi matrix
        sqrt_b = np.sqrt(b[1:])
        jacobi = np.diag(a) + np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)

        # Compute weights and abscissas
        eigenvalues, eigenvectors = np.linalg.eig(jacobi)
        idx = eigenvalues.argsort()
        x = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        w = moments[0] * eigenvectors[0, :] ** 2

        # Adaptive conditions. When both satisfied, return the results.
        if adaptive:
            dab = np.zeros(n1)
            mab = np.zeros(n1)

            for i in range(n1 - 1, 0, -1):
                dab[i] = min(abs(x[i] - x[0:i]))
                mab[i] = max(abs(x[i] - x[0:i]))

            mindab = min(dab[1:n1])
            maxmab = max(mab[1:n1])
            if n1 == 2:
                maxmab = 1

            if min(w) / max(w) > rmax and mindab / maxmab > eabs:
                return x, w
        else:
            return x, w


def hyperbolic(moments, max_skewness=30, checks=True):
    """
    This is a driver for hyperbolic qmom.
    It calls :func:`hyqmom2` if ``len(moments) = 2``, or :func:`hyqmom3` if ``len(moments) = 3``

    :param moments: Statistical moments of the transported PDF
    :param max_skewness: Maximum skewness (optional, defaults to 30)
    :type moments: array like
    :type max_skewness: float
    :return: Abscissas, weights
    :rtype: array like
    """

    num_moments = len(moments)
    if num_moments == 3:
        return hyqmom2(moments)
    elif num_moments == 5:
        return hyqmom3(moments, max_skewness, checks)
    else:
        print("inversion: hyperbolic: incorrect number of moments(%i)" % num_moments)
        return


@njit
def hyqmom2(moments):
    """
    This function inverts moments into a two-node quadrature rule.

    :param moments: Statistical moments of the transported PDF
    :type moments: array like
    :return: Abscissas, weights
    :rtype: array like
    """

    n = 2
    w = np.zeros(n)
    x = np.zeros(n)

    w[0] = moments[0] / 2.0
    w[1] = w[0]

    bx = moments[1] / moments[0]
    d2 = moments[2] / moments[0]
    c2 = d2 - bx ** 2.0

    if c2 < 10 ** (-12):
        c2 = 10 ** (-12)
    x[0] = bx - math.sqrt(c2)
    x[1] = bx + math.sqrt(c2)

    return x, w


@njit
def hyqmom3(moments, max_skewness=30, checks=True):
    """
    This function inverts moments into a three-node quadrature rule.

    :param moments: Statistical moments of the transported PDF
    :param max_skewness: Maximum skewness
    :type moments: array like
    :type max_skewness: float
    :return: Abscissas, weights
    :rtype: array like
    """

    n = 3
    etasmall = 10 ** (-10)
    verysmall = 10 ** (-14)
    realsmall = 10 ** (-14)

    w = np.zeros(n)
    x = np.zeros(n)
    xp = np.zeros(n)
    xps = np.zeros(n)
    rho = np.zeros(n)

    if moments[0] <= verysmall and checks:
        w[1] = moments[0]
        return x, w

    bx = moments[1] / moments[0]
    d2 = moments[2] / moments[0]
    d3 = moments[3] / moments[0]
    d4 = moments[4] / moments[0]
    c2 = d2 - bx ** 2
    c3 = d3 - 3 * bx * d2 + 2 * bx ** 3
    c4 = d4 - 4 * bx * d3 + 6 * (bx ** 2) * d2 - 3 * bx ** 4

    if checks:
        if c2 < 0:
            if c2 < -verysmall:
                print("Error: c2 negative in three node HYQMOM")
                print(c2)
                return
        else:
            realizable = c2 * c4 - c2 ** 3 - c3 ** 2
            if realizable < 0:
                if c2 >= etasmall:
                    q = c3 / math.sqrt(c2) / c2
                    eta = c4 / c2 / c2
                    if abs(q) > verysmall:
                        slope = (eta - 3) / q
                        det = 8 + slope ** 2
                        qp = 0.5 * (slope + math.sqrt(det))
                        qm = 0.5 * (slope - math.sqrt(det))
                        if q > 0:
                            q = qp
                        else:
                            q = qm
                    else:
                        q = 0

                    eta = q ** 2 + 1
                    c3 = q * math.sqrt(c2) * c2
                    c4 = eta * c2 ** 2
                    if realizable < -(10.0 ** (-6)):
                        print("Error: c4 small in HYQMOM3")
                        return
                else:
                    c3 = 0.0
                    c4 = c2 ** 2.0

    scale = math.sqrt(c2)
    if checks and c2 < etasmall:
        q = 0
        eta = 1
    else:
        q = c3 / math.sqrt(c2) / c2
        eta = c4 / c2 / c2

    if q ** 2 > max_skewness ** 2:
        slope = (eta - 3) / q
        if q > 0:
            q = max_skewness
        else:
            q = -max_skewness
        eta = 3 + slope * q
        if checks:
            realizable = eta - 1 - q ** 2
            if realizable < 0:
                eta = 1 + q ** 2

    xps[0] = (q - math.sqrt(4 * eta - 3 * q ** 2)) / 2.0
    xps[1] = 0.0
    xps[2] = (q + math.sqrt(4 * eta - 3 * q ** 2)) / 2.0

    dem = 1.0 / math.sqrt(4 * eta - 3 * q ** 2)
    prod = -xps[0] * xps[2]
    prod = max(prod, 1 + realsmall)

    rho[0] = -dem / xps[0]
    rho[1] = 1 - 1 / prod
    rho[2] = dem / xps[2]

    srho = np.sum(rho)
    rho = rho / srho
    if min(rho) < 0:
        print("Error: Negative weight in HYQMOM")
        return

    scales = np.sum(rho * xps ** 2) / np.sum(rho)
    xp = xps * scale / math.sqrt(scales)

    w = moments[0] * rho
    x = xp
    x = bx + x

    return x, w



@njit
def conditional_hyperbolic(moments, indices, max_skewness=30, checks=True):
    """
    This function inverts moments into a two-node quadrature rule.

    :param moments: Statistical moments of the transported PDF
    :type moments: array like
    :return: Abscissas, weights
    :rtype: array like
    """

    # num_dim = len(indices)

    if num_dim == 16:
        return chyqmom27(moments, indices, max_skewness, checks)
    if num_dim == 10:
        return chyqmom9(moments, indices, max_skewness, checks)
    # if num_dim == 6:
    #     return chyqmom4(moments, indices)

@njit
def chyqmom4(moments, indices, max_skewness=30):

    # normalidx = indices.tolist()
    # mom00 = moments[normalidx.index([0,0])]
    # mom10 = moments[normalidx.index([1,0])]
    # mom01 = moments[normalidx.index([0,1])]
    # mom20 = moments[normalidx.index([2,0])]
    # mom11 = moments[normalidx.index([1,1])]
    # mom02 = moments[normalidx.index([0,2])]

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

    x = [x, y]
    return x, w


@njit
def chyqmom9(moments, indices, max_skewness=30, checks=True):

    mom00 = moments[0]
    mom10 = moments[1]
    mom01 = moments[2]
    mom20 = moments[3]
    mom11 = moments[4]
    mom02 = moments[5]
    mom30 = moments[6]
    mom03 = moments[7]
    mom40 = moments[8]
    mom04 = moments[9]

    n = 9
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    abscissas = np.zeros((n, 2))

    csmall = 10.0 ** (-10)
    verysmall = 10.0 ** (-14)


    if mom00 < verysmall and checks:
        abscissas[:, 0] = x[:]
        abscissas[:, 1] = y[:]
        w[3] = mom00
        return abscissas, w

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

    M1 = np.array([1, 0, c20, c30, c40])
    xp, rho = hyqmom3(M1, max_skewness, checks)
    if checks and c20 < csmall:
        rho[0] = 0.0
        rho[1] = 1.0
        rho[2] = 0.0
        yf = 0 * xp
        M2 = np.array([1, 0, c02, c03, c04])
        xp2, rho2 = hyqmom3(M2, max_skewness, checks)
        yp21 = xp2[0]
        yp22 = xp2[1]
        yp23 = xp2[2]
        rho21 = rho2[0]
        rho22 = rho2[1]
        rho23 = rho2[2]
    else:
        yf = c11 * xp / c20
        mu2avg = c02 - np.sum(rho * (yf ** 2.0))
        mu2avg = max(mu2avg, 0.0)
        mu2 = mu2avg
        mu3 = 0 * mu2
        mu4 = mu2 ** 2.0
        if mu2 > csmall:
            q = (c03 - np.sum(rho * (yf ** 3.0))) / mu2 ** (3.0 / 2.0)
            eta = (
                c04 - np.sum(rho * (yf ** 4.0)) - 6 * np.sum(rho * (yf ** 2.0)) * mu2
            ) / mu2 ** 2.0
            if eta < (q ** 2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3.0) / q
                    det = 8.0 + slope ** 2.0
                    qp = 0.5 * (slope + math.sqrt(det))
                    qm = 0.5 * (slope - math.sqrt(det))
                    if q > 0:
                        q = qp
                    else:
                        q = qm
                else:
                    q = 0

                eta = q ** 2 + 1

            mu3 = q * mu2 ** (3.0 / 2.0)
            mu4 = eta * mu2 ** 2.0

        M3 = np.array([1, 0, mu2, mu3, mu4])
        xp3, rh3 = hyqmom3(M3, max_skewness, checks)
        yp21 = xp3[0]
        yp22 = xp3[1]
        yp23 = xp3[2]
        rho21 = rh3[0]
        rho22 = rh3[1]
        rho23 = rh3[2]

    w[0] = rho[0] * rho21
    w[1] = rho[0] * rho22
    w[2] = rho[0] * rho23
    w[3] = rho[1] * rho21
    w[4] = rho[1] * rho22
    w[5] = rho[1] * rho23
    w[6] = rho[2] * rho21
    w[7] = rho[2] * rho22
    w[8] = rho[2] * rho23
    w = mom00 * w

    x[0] = xp[0]
    x[1] = xp[0]
    x[2] = xp[0]
    x[3] = xp[1]
    x[4] = xp[1]
    x[5] = xp[1]
    x[6] = xp[2]
    x[7] = xp[2]
    x[8] = xp[2]
    x = bx + x

    y[0] = yf[0] + yp21
    y[1] = yf[0] + yp22
    y[2] = yf[0] + yp23
    y[3] = yf[1] + yp21
    y[4] = yf[1] + yp22
    y[5] = yf[1] + yp23
    y[6] = yf[2] + yp21
    y[7] = yf[2] + yp22
    y[8] = yf[2] + yp23
    y = by + y

    abscissas[:, 0] = x[:]
    abscissas[:, 1] = y[:]
    # SHB Note: If running 0D case then the abscissas output might not work

    return abscissas, w


@njit
def chyqmom27(moments, indices, max_skewness=30, checks=True):

    # Indices used for calling chyqmom9
    RF_idx = np.array(
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4]]
    )

    m000 = moments[0]
    m100 = moments[1]
    m010 = moments[2]
    m001 = moments[3]
    m200 = moments[4]
    m110 = moments[5]
    m101 = moments[6]
    m020 = moments[7]
    m011 = moments[8]
    m002 = moments[9]
    m300 = moments[10]
    m030 = moments[11]
    m003 = moments[12]
    m400 = moments[13]
    m040 = moments[14]
    m004 = moments[15]

    small = 1.0e-10
    isosmall = 1.0e-14
    csmall = 1.0e-10
    wsmall = 1.0e-4
    verysmall = 1.0e-14

    n = 27
    w = np.zeros(n)
    abscissas = np.zeros((n, 3))
    Yf = np.zeros(3)
    Zf = np.zeros((3, 3))
    W = np.zeros(n)

    if m000 <= verysmall and checks:
        W[12] = m000
        return abscissas, W

    bx = m100 / m000
    by = m010 / m000
    bz = m001 / m000

    if checks and m000 <= isosmall:
        d200 = m200 / m000
        d020 = m020 / m000
        d002 = m002 / m000
        d300 = m300 / m000
        d030 = m030 / m000
        d003 = m003 / m000
        d400 = m400 / m000
        d040 = m040 / m000
        d004 = m004 / m000

        c200 = d200 - bx ** 2
        c020 = d020 - by ** 2
        c002 = d002 - bz ** 2
        c300 = d300 - 3 * bx * d200 + 2 * bx ** 3
        c030 = d030 - 3 * by * d020 + 2 * by ** 3
        c003 = d003 - 3 * bz * d002 + 2 * bz ** 3
        c400 = d400 - 4 * bx * d300 + 6 * (bx ** 2) * d200 - 3 * bx ** 4
        c040 = d040 - 4 * by * d030 + 6 * (by ** 2) * d020 - 3 * by ** 4
        c004 = d004 - 4 * bz * d003 + 6 * (bz ** 2) * d002 - 3 * bz ** 4

        c110 = 0
        c101 = 0
        c011 = 0
    else:
        d200 = m200 / m000
        d110 = m110 / m000
        d101 = m101 / m000
        d020 = m020 / m000
        d011 = m011 / m000
        d002 = m002 / m000
        d300 = m300 / m000
        d030 = m030 / m000
        d003 = m003 / m000
        d400 = m400 / m000
        d040 = m040 / m000
        d004 = m004 / m000

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

    if c200 <= 0 and checks:
        c200 = 0
        c300 = 0
        c400 = 0

    if c200 * c400 < (c200 ** 3 + c300 ** 2) and checks:
        q = c300 / c200 ** (3.0 / 2.0)
        eta = c400 / c200 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3.0) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if q > 0:
                q = qp
            else:
                q = qm
        else:
            q = 0

        eta = q ** 2 + 1
        c300 = q * c200 ** (3.0 / 2.0)
        c400 = eta * c200 ** 2.0

    if c020 <= 0 and checks:
        c020 = 0
        c030 = 0
        c040 = 0

    if c200 * c400 < (c200 ** 3 + c300 ** 2) and checks:
        q = c300 / c200 ** (3 / 2)
        eta = c400 / c200 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c300 = q * c200 ** (3 / 2)
        c400 = eta * c200 ** 2

    if c020 <= 0 and checks:
        c020 = 0
        c030 = 0
        c040 = 0

    if c020 * c040 < (c020 ** 3 + c030 ** 2) and checks:
        q = c030 / c020 ** (3 / 2)
        eta = c040 / c020 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c030 = q * c020 ** (3 / 2)
        c040 = eta * c020 ** 2

    if c002 <= 0 and checks:
        c002 = 0
        c003 = 0
        c004 = 0

    if c002 * c004 < (c002 ** 3 + c003 ** 2) and checks:
        q = c003 / c002 ** (3 / 2)
        eta = c004 / c002 ** 2
        if abs(q) > verysmall:
            slope = (eta - 3) / q
            det = 8 + slope ** 2
            qp = 0.5 * (slope + math.sqrt(det))
            qm = 0.5 * (slope - math.sqrt(det))
            if sign(q) == 1:
                q = qp
            else:
                q = qm
        else:
            q = 0
        eta = q ** 2 + 1
        c003 = q * c002 ** (3 / 2)
        c004 = eta * c002 ** 2

    M1 = np.array([1, 0, c200, c300, c400])
    xp, rho = hyqmom3(M1, max_skewness, checks)

    rho11 = 0
    rho12 = 1
    rho13 = 0
    rho21 = 0
    rho23 = 0
    rho31 = 0
    rho32 = 1
    rho33 = 0

    yp11 = 0
    yp12 = 0
    yp13 = 0
    yp21 = 0
    yp22 = 0
    yp23 = 0
    yp31 = 0
    yp32 = 0
    yp33 = 0

    rho111 = 0
    rho112 = 1
    rho113 = 0
    rho121 = 0
    rho122 = 1
    rho123 = 0
    rho131 = 0
    rho132 = 1
    rho133 = 0
    rho211 = 0
    rho212 = 1
    rho213 = 0
    rho221 = 0
    rho222 = 1
    rho223 = 0
    rho231 = 0
    rho232 = 1
    rho233 = 0
    rho311 = 0
    rho312 = 1
    rho313 = 0
    rho321 = 0
    rho322 = 1
    rho323 = 0
    rho331 = 0
    rho332 = 1
    rho333 = 0

    zp111 = 0
    zp112 = 0
    zp113 = 0
    zp121 = 0
    zp122 = 0
    zp123 = 0
    zp131 = 0
    zp132 = 0
    zp133 = 0
    zp211 = 0
    zp212 = 0
    zp213 = 0
    zp221 = 0
    zp222 = 0
    zp223 = 0
    zp231 = 0
    zp232 = 0
    zp233 = 0
    zp311 = 0
    zp312 = 0
    zp313 = 0
    zp321 = 0
    zp322 = 0
    zp323 = 0
    zp331 = 0
    zp332 = 0
    zp333 = 0

    if c200 <= csmall and checks:
        if c020 <= csmall:
            M0 = np.array([1, 0, c002, c003, c004])
            Z0, W0 = hyqmom3(M0, max_skewness, checks)

            rho[0] = 0
            rho[1] = 1
            rho[2] = 0
            rho22 = 1
            rho221 = W0[0]
            rho222 = W0[1]
            rho223 = W0[2]
            xp = 0 * xp
            zp221 = Z0[0]
            zp222 = Z0[1]
            zp223 = Z0[2]
        else:
            M1 = np.array([1, 0, 0, c020, c011, c002, c030, c003, c040, c004])
            Q1, W1 = chyqmom9(M1, RF_idx, max_skewness, checks)
            Y1 = Q1[:,0]
            Z1 = Q1[:,1]

            rho[0] = 0
            rho[1] = 1
            rho[2] = 0
            rho12 = 0
            rho21 = 1
            rho22 = 1
            rho23 = 1
            rho31 = 0
            rho211 = W1[0]
            rho212 = W1[1]
            rho213 = W1[2]
            rho221 = W1[3]
            rho222 = W1[4]
            rho223 = W1[5]
            rho231 = W1[6]
            rho232 = W1[7]
            rho233 = W1[8]

            xp = 0 * xp
            yp21 = Y1[0]
            yp22 = Y1[4]
            yp23 = Y1[8]
            zp211 = Z1[0]
            zp212 = Z1[1]
            zp213 = Z1[2]
            zp221 = Z1[3]
            zp222 = Z1[4]
            zp223 = Z1[5]
            zp231 = Z1[6]
            zp232 = Z1[7]
            zp233 = Z1[8]
    elif c020 <= csmall and checks:
        M2 = np.array([1, 0, 0, c200, c101, c002, c300, c003, c400, c004])
        Q2, W2 = chyqmom9(M2, RF_idx, max_skewness, checks)
        X2 = Q2[:,0]
        Z2 = Q2[:,1]

        rho[0] = 1
        rho[1] = 1
        rho[2] = 1
        rho12 = 1
        rho22 = 1
        rho32 = 1
        rho121 = W2[0]
        rho122 = W2[1]
        rho123 = W2[2]
        rho221 = W2[3]
        rho222 = W2[4]
        rho223 = W2[5]
        rho321 = W2[6]
        rho322 = W2[7]
        rho323 = W2[8]
        xp[0] = X2[0]
        xp[1] = X2[4]
        xp[2] = X2[8]
        zp121 = Z2[0]
        zp122 = Z2[1]
        zp123 = Z2[2]
        zp221 = Z2[3]
        zp222 = Z2[4]
        zp223 = Z2[5]
        zp321 = Z2[6]
        zp322 = Z2[7]
        zp323 = Z2[8]
    elif c002 <= csmall and checks:
        M3 = np.array([1, 0, 0, c200, c110, c020, c300, c030, c400, c040])
        Q3, W3 = chyqmom9(M3, RF_idx, max_skewness, checks)
        X3 = Q3[:,0]
        Y3 = Q3[:,1]

        rho[0] = 1
        rho[1] = 1
        rho[2] = 1
        rho11 = W3[0]
        rho12 = W3[1]
        rho13 = W3[2]
        rho21 = W3[3]
        rho22 = W3[4]
        rho23 = W3[5]
        rho31 = W3[6]
        rho32 = W3[7]
        rho33 = W3[8]
        xp[0] = X3[0]
        xp[1] = X3[4]
        xp[2] = X3[8]
        yp11 = Y3[0]
        yp12 = Y3[1]
        yp13 = Y3[2]
        yp21 = Y3[3]
        yp22 = Y3[4]
        yp23 = Y3[5]
        yp31 = Y3[6]
        yp32 = Y3[7]
        yp33 = Y3[8]
    else:
        M4 = np.array([1, 0, 0, c200, c110, c020, c300, c030, c400, c040])
        Q4, W4 = chyqmom9(M4, RF_idx, max_skewness, checks)
        X4 = Q4[:,0]
        Y4 = Q4[:,1]

        rho11 = W4[0] / (W4[0] + W4[1] + W4[2])
        rho12 = W4[1] / (W4[0] + W4[1] + W4[2])
        rho13 = W4[2] / (W4[0] + W4[1] + W4[2])
        rho21 = W4[3] / (W4[3] + W4[4] + W4[5])
        rho22 = W4[4] / (W4[3] + W4[4] + W4[5])
        rho23 = W4[5] / (W4[3] + W4[4] + W4[5])
        rho31 = W4[6] / (W4[6] + W4[7] + W4[8])
        rho32 = W4[7] / (W4[6] + W4[7] + W4[8])
        rho33 = W4[8] / (W4[6] + W4[7] + W4[8])

        Yf[0] = rho11 * Y4[0] + rho12 * Y4[1] + rho13 * Y4[2]
        Yf[1] = rho21 * Y4[3] + rho22 * Y4[4] + rho23 * Y4[5]
        Yf[2] = rho31 * Y4[6] + rho32 * Y4[7] + rho33 * Y4[8]

        yp11 = Y4[0] - Yf[0]
        yp12 = Y4[1] - Yf[0]
        yp13 = Y4[2] - Yf[0]
        yp21 = Y4[3] - Yf[1]
        yp22 = Y4[4] - Yf[1]
        yp23 = Y4[5] - Yf[1]
        yp31 = Y4[6] - Yf[2]
        yp32 = Y4[7] - Yf[2]
        yp33 = Y4[8] - Yf[2]
        scale1 = math.sqrt(c200)
        scale2 = math.sqrt(c020)
        Rho1 = np.diag(rho)
        Rho2 = np.array(
            [[rho11, rho12, rho13], [rho21, rho22, rho23], [rho31, rho32, rho33]]
        )
        Yp2 = np.array([[yp11, yp12, yp13], [yp21, yp22, yp23], [yp31, yp32, yp33]])
        Yp2s = Yp2 / scale2
        RAB = Rho1 * Rho2
        XAB = np.array(
            [[xp[0], xp[1], xp[2]], [xp[0], xp[1], xp[2]], [xp[0], xp[1], xp[2]]]
        )
        XABs = XAB / scale1
        YAB = Yp2 + np.diag(Yf) * np.ones(3)
        YABs = YAB / scale2
        C01 = np.multiply(RAB, YABs)
        Yc0 = np.ones(3)
        Yc1 = XABs
        Yc2 = Yp2s
        A1 = np.sum(np.multiply(C01, Yc1))
        A2 = np.sum(np.multiply(C01, Yc2))

        c101s = c101 / scale1
        c011s = c011 / scale2
        if c101s ** 2 >= c002 * (1 - small):
            c101s = sign(c101s) * math.sqrt(c002)
        elif c011s ** 2 >= c002 * (1 - small):
            c110s = c110 / scale1 / scale2
            c011s = sign(c011s) * math.sqrt(c002)
            c101s = c110s * c011s

        b0 = 0
        b1 = c101s
        b2 = 0
        if A2 < wsmall:
            b2 = (c011s - A1 * b1) / A2

        Zf = b0 * Yc0 + b1 * Yc1 + b2 * Yc2
        SUM002 = np.sum(np.multiply(RAB, Zf ** 2))
        mu2 = c002 - SUM002
        mu2 = max(0, mu2)
        q = 0
        eta = 1
        if mu2 > csmall:
            SUM1 = mu2 ** (3 / 2)
            SUM3 = np.sum(np.multiply(RAB, Zf ** 3))
            q = (c003 - SUM3) / SUM1
            SUM2 = mu2 ** 2
            SUM4 = np.sum(np.multiply(RAB, Zf ** 4)) + 6 * SUM002 * mu2
            eta = (c004 - SUM4) / SUM2
            if eta < (q ** 2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3) / q
                    det = 8 + slope ** 2
                    qp = 0.5 * (slope + math.sqrt(det))
                    qm = 0.5 * (slope - math.sqrt(det))
                    if sign(q) == 1:
                        q = qp
                    else:
                        q = qm
                else:
                    q = 0
                eta = q ** 2 + 1
        mu3 = q * mu2 ** (3 / 2)
        mu4 = eta * mu2 ** 2
        M5 = np.array([1, 0, mu2, mu3, mu4])
        xp11, rh11 = hyqmom3(M5, max_skewness, checks)

        rho111 = rh11[0]
        rho112 = rh11[1]
        rho113 = rh11[2]

        zp111 = xp11[0]
        zp112 = xp11[1]
        zp113 = xp11[2]

        rh12 = rh11
        xp12 = xp11
        rho121 = rh12[0]
        rho122 = rh12[1]
        rho123 = rh12[2]

        zp121 = xp12[0]
        zp122 = xp12[1]
        zp123 = xp12[2]

        rh13 = rh11
        xp13 = xp11
        rho131 = rh13[0]
        rho132 = rh13[1]
        rho133 = rh13[2]

        zp131 = xp13[0]
        zp132 = xp13[1]
        zp133 = xp13[2]

        rh21 = rh11
        xp21 = xp11
        zp211 = xp21[0]
        zp212 = xp21[1]
        zp213 = xp21[2]

        rho211 = rh21[0]
        rho212 = rh21[1]
        rho213 = rh21[2]

        rh22 = rh11
        xp22 = xp11
        zp221 = xp22[0]
        zp222 = xp22[1]
        zp223 = xp22[2]

        rho221 = rh22[0]
        rho222 = rh22[1]
        rho223 = rh22[2]

        rh23 = rh11
        xp23 = xp11
        zp231 = xp23[0]
        zp232 = xp23[1]
        zp233 = xp23[2]

        rho231 = rh23[0]
        rho232 = rh23[1]
        rho233 = rh23[2]

        rh31 = rh11
        xp31 = xp11
        rho311 = rh31[0]
        rho312 = rh31[1]
        rho313 = rh31[2]

        zp311 = xp31[0]
        zp312 = xp31[1]
        zp313 = xp31[2]

        rh32 = rh11
        xp32 = xp11
        rho321 = rh32[0]
        rho322 = rh32[1]
        rho323 = rh32[2]

        zp321 = xp32[0]
        zp322 = xp32[1]
        zp323 = xp32[2]

        rh33 = rh11
        xp33 = xp11
        rho331 = rh33[0]
        rho332 = rh33[1]
        rho333 = rh33[2]

        zp331 = xp33[0]
        zp332 = xp33[1]
        zp333 = xp33[2]

    W[0] = rho[0] * rho11 * rho111
    W[1] = rho[0] * rho11 * rho112
    W[2] = rho[0] * rho11 * rho113
    W[3] = rho[0] * rho12 * rho121
    W[4] = rho[0] * rho12 * rho122
    W[5] = rho[0] * rho12 * rho123
    W[6] = rho[0] * rho13 * rho131
    W[7] = rho[0] * rho13 * rho132
    W[8] = rho[0] * rho13 * rho133
    W[9] = rho[1] * rho21 * rho211
    W[10] = rho[1] * rho21 * rho212
    W[11] = rho[1] * rho21 * rho213
    W[12] = rho[1] * rho22 * rho221
    W[13] = rho[1] * rho22 * rho222
    W[14] = rho[1] * rho22 * rho223
    W[15] = rho[1] * rho23 * rho231
    W[16] = rho[1] * rho23 * rho232
    W[17] = rho[1] * rho23 * rho233
    W[18] = rho[2] * rho31 * rho311
    W[19] = rho[2] * rho31 * rho312
    W[20] = rho[2] * rho31 * rho313
    W[21] = rho[2] * rho32 * rho321
    W[22] = rho[2] * rho32 * rho322
    W[23] = rho[2] * rho32 * rho323
    W[24] = rho[2] * rho33 * rho331
    W[25] = rho[2] * rho33 * rho332
    W[26] = rho[2] * rho33 * rho333
    W = m000 * W

    abscissas[0, 0] = xp[0]
    abscissas[1, 0] = xp[0]
    abscissas[2, 0] = xp[0]
    abscissas[3, 0] = xp[0]
    abscissas[4, 0] = xp[0]
    abscissas[5, 0] = xp[0]
    abscissas[6, 0] = xp[0]
    abscissas[7, 0] = xp[0]
    abscissas[8, 0] = xp[0]
    abscissas[9, 0] = xp[1]
    abscissas[10, 0] = xp[1]
    abscissas[11, 0] = xp[1]
    abscissas[12, 0] = xp[1]
    abscissas[13, 0] = xp[1]
    abscissas[14, 0] = xp[1]
    abscissas[15, 0] = xp[1]
    abscissas[16, 0] = xp[1]
    abscissas[17, 0] = xp[1]
    abscissas[18, 0] = xp[2]
    abscissas[19, 0] = xp[2]
    abscissas[20, 0] = xp[2]
    abscissas[21, 0] = xp[2]
    abscissas[22, 0] = xp[2]
    abscissas[23, 0] = xp[2]
    abscissas[24, 0] = xp[2]
    abscissas[25, 0] = xp[2]
    abscissas[26, 0] = xp[2]
    abscissas[:, 0] += bx

    abscissas[0, 1] = Yf[0] + yp11
    abscissas[1, 1] = Yf[0] + yp11
    abscissas[2, 1] = Yf[0] + yp11
    abscissas[3, 1] = Yf[0] + yp12
    abscissas[4, 1] = Yf[0] + yp12
    abscissas[5, 1] = Yf[0] + yp12
    abscissas[6, 1] = Yf[0] + yp13
    abscissas[7, 1] = Yf[0] + yp13
    abscissas[8, 1] = Yf[0] + yp13
    abscissas[9, 1] = Yf[1] + yp21
    abscissas[10, 1] = Yf[1] + yp21
    abscissas[11, 1] = Yf[1] + yp21
    abscissas[12, 1] = Yf[1] + yp22
    abscissas[13, 1] = Yf[1] + yp22
    abscissas[14, 1] = Yf[1] + yp22
    abscissas[15, 1] = Yf[1] + yp23
    abscissas[16, 1] = Yf[1] + yp23
    abscissas[17, 1] = Yf[1] + yp23
    abscissas[18, 1] = Yf[2] + yp31
    abscissas[19, 1] = Yf[2] + yp31
    abscissas[20, 1] = Yf[2] + yp31
    abscissas[21, 1] = Yf[2] + yp32
    abscissas[22, 1] = Yf[2] + yp32
    abscissas[23, 1] = Yf[2] + yp32
    abscissas[24, 1] = Yf[2] + yp33
    abscissas[25, 1] = Yf[2] + yp33
    abscissas[26, 1] = Yf[2] + yp33
    abscissas[:, 1] += by

    abscissas[0, 2] = Zf[0, 0] + zp111
    abscissas[1, 2] = Zf[0, 0] + zp112
    abscissas[2, 2] = Zf[0, 0] + zp113
    abscissas[3, 2] = Zf[0, 1] + zp121
    abscissas[4, 2] = Zf[0, 1] + zp122
    abscissas[5, 2] = Zf[0, 1] + zp123
    abscissas[6, 2] = Zf[0, 2] + zp131
    abscissas[7, 2] = Zf[0, 2] + zp132
    abscissas[8, 2] = Zf[0, 2] + zp133
    abscissas[9, 2] = Zf[1, 0] + zp211
    abscissas[10, 2] = Zf[1, 0] + zp212
    abscissas[11, 2] = Zf[1, 0] + zp213
    abscissas[12, 2] = Zf[1, 1] + zp221
    abscissas[13, 2] = Zf[1, 1] + zp222
    abscissas[14, 2] = Zf[1, 1] + zp223
    abscissas[15, 2] = Zf[1, 2] + zp231
    abscissas[16, 2] = Zf[1, 2] + zp232
    abscissas[17, 2] = Zf[1, 2] + zp233
    abscissas[18, 2] = Zf[2, 0] + zp311
    abscissas[19, 2] = Zf[2, 0] + zp312
    abscissas[20, 2] = Zf[2, 0] + zp313
    abscissas[21, 2] = Zf[2, 1] + zp321
    abscissas[22, 2] = Zf[2, 1] + zp322
    abscissas[23, 2] = Zf[2, 1] + zp323
    abscissas[24, 2] = Zf[2, 2] + zp331
    abscissas[25, 2] = Zf[2, 2] + zp332
    abscissas[26, 2] = Zf[2, 2] + zp333
    abscissas[:, 2] += bz

    return abscissas, W
