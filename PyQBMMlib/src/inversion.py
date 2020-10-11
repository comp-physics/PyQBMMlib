import numpy as np

from scipy import sqrt, zeros
from scipy.linalg import eig

import math

def sign(q):
    if q > 0:
        return 1
    elif q == 0:
        return 0
    else:
        return -1

###
### Invesion methods for 1D problems
###
def wheeler(moments, adaptive = False):
    
    # From Bo Kong code in old_python_qmom
    # def adaptive_Wheeler(mom):
    """ Return weights,  nodes, and number of nodes using adaptive Wheeler
    algorithm.
    """
    
    # SHB: need to convert all this stuff to numpy?

    n = len( moments ) // 2

    # SHB let's make adaptive and non-adaptive wheeler one routine with if statemtns
    # if adaptive:
    # Adaptivity parameters
    rmax   = 1e-8
    eabs   = 1e-8
    cutoff = 0
    
    # Check if moments are unrealizable.
    if moments[0] <= 0:
        print("Wheeler: Moments are NOT realizable (moment[0] <= 0.0). Run failed.")
        exit()

    if n == 1 or ( adaptive and moments[0] < rmax ):
        w = moments[0]
        x = moments[1] / moments[0]
        return x, w

    # Set modified moments equal to input moments.
    nu = moments

    # Construct recurrence matrix
    ind = n
    a   = np.zeros( n )
    b   = np.zeros( n )    
    sigma = np.zeros( [ 2*ind+1, 2*ind+1 ] )

    for i in range(1,2*ind+1):
        sigma[1,i] = nu[i-1]

    a[0] = nu[1]/nu[0]
    b[0] = 0

    for k in range(2,ind+1):
        for l in range(k,2*ind-k+2):
            sigma[k, l] = sigma[k-1, l+1]-a[k-2]*sigma[k-1, l]-b[k-2]*sigma[k-2, l]
        a[k-1] = sigma[k, k+1]/sigma[k, k]-sigma[k-1, k]/sigma[k-1, k-1]
        b[k-1] = sigma[k, k]/sigma[k-1, k-1]

    # Find maximum n using diagonal element of sigma
    if adaptive:
        for k in range(ind,1,-1):
            if sigma[k,k] <= cutoff:
                n = k-1
                if n == 1:
                    w = moments[0]
                    x = moments[1]/moments[0]
                    return x, w

        # Use maximum n to re-calculate recurrence matrix
        a = np.zeros( n )
        b = np.zeros( n )
        w = np.zeros( n ) 
        x = np.zeros( n )
        sigma = np.zeros( [ 2*n+1, 2*n+1 ] )
        sigma[1,1:] = nu
        
        a[0] = nu[1] / nu[0]
        b[0] = 0
        for k in range(2, n+1):
            for l in range(k, 2*n-k+2):
                sigma[k, l] = sigma[k-1, l+1]-a[k-2]*sigma[k-1, l]-b[k-2]*sigma[k-2, l]
            a[k-1] = sigma[k, k+1]/sigma[k, k]-sigma[k-1, k]/sigma[k-1, k-1]
            b[k-1] = sigma[k, k]/sigma[k-1, k-1]

    # Check if moments are unrealizable
    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! Program exits.")
        exit()

    # Setup Jacobi matrix for n-point quadrature, adapt n using rmin and eabs
    for n1 in range(n,0,-1):
        if n1 == 1:
            w = moments[0]
            x = moments[1]/moments[0]
            return x, w

        # Jacobi matrix
        sqrt_b = np.sqrt( b[1:] ) 
        jacobi = np.diag( a ) + np.diag( sqrt_b, -1 ) + np.diag( sqrt_b, 1 )
        
        # Compute weights and abscissas
        eigenvalues, eigenvectors = np.linalg.eig( jacobi )
        idx = eigenvalues.argsort()
        x   = eigenvalues[idx].real
        eigenvectors = eigenvectors[:,idx].real
        w = moments[0]*eigenvectors[0,:]**2

        # SHB: Let's combine adaptive and non-adaptive into one routine using an if-statement here

        # Adaptive conditions. When both satisfied, return the results.
        if adaptive:
            dab = zeros(n1)
            mab = zeros(n1)

            for i in range(n1-1,0,-1):
                dab[i] = min(abs(x[i]-x[0:i]))
                mab[i] = max(abs(x[i]-x[0:i]))

            mindab = min(dab[1:n1])
            maxmab = max(mab[1:n1])
            if n1 == 2:
                maxmab = 1

            if min(w)/max(w) > rmax and mindab/maxmab > eabs:
                return x, w
        else:
            return x, w

    # weights   = np.array([])
    # abscissas = np.array([])
    # return weights, abscissas

def hyperbolic(moments, max_skewness = 30):
    
    num_moments = len( moments )
    if num_moments == 3:
        return hyperbolic_two_nodes( moments )
    elif num_moments == 5: 
        return hyperbolic_three_nodes( moments, max_skewness )
    else:
        print('inversion: hyperbolic: incorrect number of moments(%i)' % num_moments)
        return [],[]

def hyperbolic_two_nodes(moments):

    n = 2
    w = np.zeros(n)
    x = np.zeros(n)
    
    w[0] = moments[0]/2.
    w[1] = w[0]

    bx = moments[1]/moments[0]
    d2 = moments[2]/moments[0]
    c2 = d2 - bx**2.

    if c2 < 10**(-12):
        c2 = 10**(-12)
    x[0] = bx - math.sqrt(c2)
    x[1] = bx + math.sqrt(c2)

    return x,w

def hyperbolic3(momenst, max_skewness):
    # needs to be ported to python
    # refer to HYQMOM3 in QBMMlib Mathematica
    n = 3
    etasmall  = 10**(-10) 
    verysmall = 10**(-14) 
    realsmall = 10**(-14) 

    w   = np.zeros(n)
    x   = np.zeros(n) 
    xp  = np.zeros(n)
    xps = np.zeros(n)
    rho = np.zeros(n)

    if moments[0] <= verysmall:
        w[1] = moments[0]
        return x,w

    bx = moments[1]/moments[0] 
    d2 = moments[2]/moments[0] 
    d3 = moments[3]/moments[0] 
    d4 = moments[4]/moments[0] 
    c2 = d2 - bx**2 
    c3 = d3 - 3*bx*d2 + 2*bx**3 
    c4 = d4 - 4*bx*d3 + 6*(bx**2)*d2 - 3*bx**4
    realizable = c2*c4 - c2**3 - c3**2 
    if c2 < 0: 
        if c2 < -verysmall:
            print("Error: c2 negative in three node HYQMOM")
            exit()
        c2 = 0
        c3 = 0 
        c4 = 0
    else:
        if realizable < 0:
            if c2 >= etasmall:
                q = c3/math.sqrt(c2)/c2 
                eta = c4/c2/c2 
                if abs(q) > verysmall:
                    slope = (eta - 3)/q 
                    det = 8 + slope**2
                    qp = 0.5*(slope+math.sqrt(det)) 
                    qm = 0.5*(slope-math.sqrt(det)) 
                    if q > 0: 
                        q = qp
                    else:
                        q = qm
                else:
                    q = 0

                eta = q**2 + 1 
                c3 = q*math.sqrt(c2)*c2 
                c4 = eta*c2**2 
                if realizable < -10.**(-6):
                    print("Error: c4 small in HYQMOM3")
                    exit()
            else: 
                c3 = 0.
                c4 = c2**2.

    scale = math.sqrt(c2)
    if c2 >= etasmall:
        q = c3/math.sqrt(c2)/c2 
        eta = c4/c2/c2
    else: 
        q = 0 
        eta = 1

    if q**2 > max_skewness**2:
        slope = (eta - 3)/q 
        if q > 0:
            q = max_skewness
        else:
            q = -max_skewness
        eta = 3 + slope*q 
        realizable = eta - 1 - q**2 
        if realizable < 0:
            eta = 1 + q**2

    xps[0] = (q - math.sqrt(4*eta-3*q**2))/2.
    xps[1] = 0. 
    xps[2] = (q + math.sqrt(4*eta-3*q**2))/2.

    dem = 1./math.sqrt(4*eta - 3*q**2)
    prod = -xps[0]*xps[2] 
    prod = max(prod,1+realsmall)

    rho[0] = -dem/xps[0] 
    rho[1] = 1 - 1/prod 
    rho[2] = dem/xps[2] 

    srho = sum(rho)
    rho = rho/srho
    if min(rho) < 0 :
        print("Error: Negative weight in HYQMOM")
        exit()
    scales = sum(rho*xps**2)/sum(rho)
    for i in range(n):
        xp[i] = xps[i]*scale/math.sqrt(scales)

    w = moments[0]*rho
    x = xp
    x = bx + x 

    return x,w


###
### Conditional inversion methods for 2+D problems
###
def conditional(moments, indices, permutation = 12):

    print('inversion: Warning: Conditional QMOM not implemented. Returning empty arrays')

    num_dim = indices.shape[1] #len( indices )

    if permutation == 12:
        weights, abscissas = conditional_12( moments, indices )
    elif permutation == 21:
        weights, abscissas = conditional_21( moments, indices )    
    
    return weights, abscissas

def conditional_hyperbolic(moments, indices, max_skewness = 30):

    num_dim = len(indices)

    if num_dim == 6:
        return chyqmom4( moments, indices, max_skewness )
    elif num_dim == 10:
        return chyqmom9( moments, indices )

def chyqmom4(moments, idx, max_skewness = 30):

    # SHB: HACK so that I can get the required elements from the numpy array
    normalidx = idx.tolist()

    mom00 = moments[normalidx.index([0,0])]
    mom10 = moments[normalidx.index([1,0])]
    mom01 = moments[normalidx.index([0,1])]
    mom20 = moments[normalidx.index([2,0])]
    mom11 = moments[normalidx.index([1,1])]
    mom02 = moments[normalidx.index([0,2])]

    n = 4
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    bx  = mom10/mom00
    by  = mom01/mom00
    d20 = mom20/mom00
    d11 = mom11/mom00
    d02 = mom02/mom00

    c20 = d20 - bx**2.
    c11 = d11 - bx*by
    c02 = d02 - by**2.

    M1 = [1, 0, c20]
    xp, rho = hyperbolic_two_nodes(M1) 
    yf = c11*xp/c20 
    mu2avg = c02 - sum(rho*yf**2)
    mu2avg = max(mu2avg, 0)
    mu2 = mu2avg 
    M3 = [ 1, 0, mu2 ] 
    xp3, rh3 = hyperbolic_two_nodes(M3)
    yp21  = xp3[0] 
    yp22  = xp3[1] 
    rho21 = rh3[0] 
    rho22 = rh3[1] 

    w[0] = rho[0]*rho21 
    w[1] = rho[0]*rho22 
    w[2] = rho[1]*rho21 
    w[3] = rho[1]*rho22 
    w = mom00*w 

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
    
    x = [ x, y ]
    return x,w

def chyqmom9(moments, indices, max_skewness = 30):

    mom00 = moments[idx.index([0,0])]
    mom10 = moments[idx.index([1,0])]
    mom01 = moments[idx.index([0,1])]
    mom20 = moments[idx.index([2,0])]
    mom11 = moments[idx.index([1,1])]
    mom02 = moments[idx.index([0,2])]
    mom30 = moments[idx.index([3,0])]
    mom03 = moments[idx.index([0,3])]
    mom40 = moments[idx.index([4,0])]
    mom04 = moments[idx.index([0,4])]

    n = 9
    w = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)

    csmall    = 10.**(-10) 
    verysmall = 10.**(-14) 

    bx  = mom10/mom00 
    by  = mom01/mom00 
    d20 = mom20/mom00 
    d11 = mom11/mom00 
    d02 = mom02/mom00 
    d30 = mom30/mom00 
    d03 = mom03/mom00 
    d40 = mom40/mom00 
    d04 = mom04/mom00 

    c20 = d20 - bx**2. 
    c11 = d11 - bx*by
    c02 = d02 - by**2.
    c30 = d30 - 3.*bx*d20 + 2.*bx**3.
    c03 = d03 - 3.*by*d02 + 2.*by**3.
    c40 = d40 - 4.*bx*d30 + 6*(bx**2.)*d20 - 3.*bx**(4)
    c04 = d04 - 4.*by*d03 + 6*(by**2.)*d02 - 3.*by**(4)

    M1 = [1, 0, c20, c30, c40] 
    xp, rho = hyperbolic_three_nodes(M1)
    if c20 < csmall:
        rho[0] = 0.
        rho[1] = 1.
        rho[2] = 0.
        yf = 0*xp 
        M2 = [1, 0, c02, c03, c04]
        xp2, rho2 = hyperbolic_three_nodes(M2, max_skewness)
        yp21 = xp2[0] 
        yp22 = xp2[1] 
        yp23 = xp2[2] 
        rho21 = rho2[0] 
        rho22 = rho2[1] 
        rho23 = rho2[2]
    else:
        yf = c11*xp/c20 
        mu2avg = c02 - sum(rho*(yf**2.))
        mu2avg = max(mu2avg, 0.)
        mu2 = mu2avg 
        mu3 = 0*mu2 
        mu4 = mu2**2. 
        if mu2 > csmall:
            q   = (c03 - sum(rho*(yf**3.)))/mu2**(3./2.) 
            eta = (c04 - sum(rho*(yf**4.)) - 6*sum(rho*(yf**2.))*mu2)/mu2**2.
            if eta < (q**2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3.)/q 
                    det = 8. + slope**2. 
                    qp = 0.5*(slope + math.sqrt(det)) 
                    qm = 0.5*(slope - math.sqrt(det)) 
                    if q > 0:
                        q = qp
                    else:
                        q = qm
                else: 
                    q = 0

                eta = q**2 + 1

            mu3 = q*mu2**(3./2.) 
            mu4 = eta*mu2**2.

        M3 = [1, 0, mu2, mu3, mu4]
        xp3, rh3 = hyperbolic_three_nodes(M3,max_skewness)
        yp21 = xp3[0] 
        yp22 = xp3[1] 
        yp23 = xp3[2] 
        rho21 = rh3[0] 
        rho22 = rh3[1] 
        rho23 = rh3[2]

    w[0] = rho[0]*rho21 
    w[1] = rho[0]*rho22 
    w[2] = rho[0]*rho23 
    w[3] = rho[1]*rho21 
    w[4] = rho[1]*rho22 
    w[5] = rho[1]*rho23 
    w[6] = rho[2]*rho21 
    w[7] = rho[2]*rho22 
    w[8] = rho[2]*rho23 
    w = mom00*w

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

    x = [ x, y ]
    return x,w

def chyqmom27(moments, indices, max_skewness = 30):

    mom000 = moments[idx.index([0,0,0])]
    mom100 = moments[idx.index([1,0,0])]
    mom010 = moments[idx.index([0,1,0])]
    mom001 = moments[idx.index([0,0,1])]
    mom200 = moments[idx.index([2,0,0])]
    mom110 = moments[idx.index([1,1,0])]
    mom101 = moments[idx.index([1,0,1])]
    mom020 = moments[idx.index([0,2,0])]
    mom011 = moments[idx.index([0,1,1])]
    mom002 = moments[idx.index([0,0,2])]
    mom300 = moments[idx.index([3,0,0])]
    mom030 = moments[idx.index([0,3,0])]
    mom003 = moments[idx.index([0,0,3])]
    mom400 = moments[idx.index([4,0,0])]
    mom040 = moments[idx.index([0,4,0])]
    mom004 = moments[idx.index([0,0,4])]

    small = 1.e-10 
    isosmall = 1.e-14
    csmall = 1.e-10
    wsmall = 1.e-4      
    verysmall = 1.e-14

    n = 27
    u = np.zeros(n)
    v = np.zeros(n)
    w = np.zeros(n)
    z = np.zeros(n)

    if m000 <= verysmall:
        n[13] = m000
        return

    bu  = mom100/mom000 
    bv  = mom010/mom000 
    bw  = mom001/mom000 

    if m000 <= isosmall: 
        d200 = m200/m000
        d020 = m020/m000
        d002 = m002/m000
        d300 = m300/m000
        d030 = m030/m000
        d003 = m003/m000
        d400 = m400/m000
        d040 = m040/m000
        d004 = m004/m000

        c200 = d200 - bu**2
        c020 = d020 - bv**2
        c002 = d002 - bw**2
        c300 = d300 - 3*bu*d200 + 2*bu**3
        c030 = d030 - 3*bv*d020 + 2*bv**3
        c003 = d003 - 3*bw*d002 + 2*bw**3
        c400 = d400 - 4*bu*d300 + 6*(bu**2)*d200 - 3*bu**4
        c040 = d040 - 4*bv*d030 + 6*(bv**2)*d020 - 3*bv**4
        c004 = d004 - 4*bw*d003 + 6*(bw**2)*d002 - 3*bw**4

        c110 = 0
        c101 = 0 
        c011 = 0
    else:
        d200 = m200/m000
        d110 = m110/m000
        d101 = m101/m000
        d020 = m020/m000
        d011 = m011/m000
        d002 = m002/m000
        d300 = m300/m000
        d030 = m030/m000
        d003 = m003/m000
        d400 = m400/m000
        d040 = m040/m000
        d004 = m004/m000

        c200 = d200 - bu**2
        c110 = d110 - bu*bv
        c101 = d101 - bu*bw
        c020 = d020 - bv**2
        c011 = d011 - bv*bw
        c002 = d002 - bw**2
        c300 = d300 - 3*bu*d200 + 2*bu**3
        c030 = d030 - 3*bv*d020 + 2*bv**3
        c003 = d003 - 3*bw*d002 + 2*bw**3
        c400 = d400 - 4*bu*d300 + 6*bu**2*d200 - 3*bu**4
        c040 = d040 - 4*bv*d030 + 6*bv**2*d020 - 3*bv**4
        c004 = d004 - 4*bw*d003 + 6*bw**2*d002 - 3*bw**4

    if c200 <= 0:
        c200 = 0
        c300 = 0
        c400 = 0

    if c200*c400 < (c200**3 + c300**2):
        q = c300/c200**(3./2.)
        eta = c400/c200**2
        if abs(q) > verysmall:
            slope = (eta - 3.)/q
            det   = 8 + slope**2
            qp    = 0.5*( slope + math.sqrt(det) )
            qm    = 0.5*( slope - math.sqrt(det) )
            if q > 0:
                q = qp
            else:
                q = qm
        else:
            q = 0

        eta  = q**2 + 1
        c300 = q*c200**(3./2.)
        c400 = eta*c200**2.

    if c020 <= 0:
        c020 = 0
        c030 = 0
        c040 = 0

    if c200*c400 < (c200**3 + c300**2):
        q = c300/c200**(3/2) 
        eta = c400/c200**2 
        if abs(q) > verysmall:
            slope = (eta - 3)/q 
            det   = 8 + slope**2 
            qp    = 0.5*( slope + math.sqrt(det) ) 
            qm    = 0.5*( slope - math.sqrt(det) ) 
            if sign(q) == 1:
                q = qp 
            else:
                q = qm 
        else:
            q = 0 
        eta  = q**2 + 1 
        c300 = q*c200**(3/2) 
        c400 = eta*c200**2 

    if c020 <= 0:
        c020 = 0 
        c030 = 0 
        c040 = 0 

    if c020*c040 < (c020**3 + c030**2):
        q   = c030/c020**(3/2) 
        eta = c040/c020**2 
        if abs(q) > verysmall:
            slope = (eta - 3)/q 
            det   = 8 + slope**2 
            qp    = 0.5*( slope + math.sqrt(det) ) 
            qm    = 0.5*( slope - math.sqrt(det) ) 
            if sign(q) == 1:
                q = qp 
            else:
                q = qm 
        else:
            q = 0 
        eta  = q**2 + 1 
        c030 = q*c020**(3/2) 
        c040 = eta*c020**2 
    if c002 <= 0:
        c002 = 0 
        c003 = 0 
        c004 = 0 
    if c002*c004 < (c002**3 + c003**2):
        q   = c003/c002**(3/2) 
        eta = c004/c002**2 
        if abs(q) > verysmall:
            slope = (eta - 3)/q 
            det   = 8 + slope**2 
            qp    = 0.5*( slope + math.sqrt(det) ) 
            qm    = 0.5*( slope - math.sqrt(det) ) 
            if sign(q) == 1:
                q = qp 
            else:
                q = qm 
        else:
            q = 0 
        eta = q**2 + 1 
        c003 = q*c002**(3/2) 
        c004 = eta*c002**2 
    M1 = [ 1, 0, c200, c300, c400 ] 
    [rho, up ] = three_node_hyqmom( M1 ) 

    rho11 = 0 
    rho12 = 1 
    rho13 = 0 
    rho21 = 0 
    rho23 = 0 
    rho31 = 0 
    rho32 = 1 
    rho33 = 0 
    vp11 = 0 
    vp12 = 0 
    vp13 = 0 
    vp21 = 0 
    vp22 = 0 
    vp23 = 0 
    vp31 = 0 
    vp32 = 0 
    vp33 = 0 

    Vf = np.zeros(3) 

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
    wp111 = 0 
    wp112 = 0 
    wp113 = 0 
    wp121 = 0 
    wp122 = 0 
    wp123 = 0 
    wp131 = 0 
    wp132 = 0 
    wp133 = 0 
    wp211 = 0 
    wp212 = 0 
    wp213 = 0 
    wp221 = 0 
    wp222 = 0 
    wp223 = 0 
    wp231 = 0 
    wp232 = 0 
    wp233 = 0 
    wp311 = 0 
    wp312 = 0 
    wp313 = 0 
    wp321 = 0 
    wp322 = 0 
    wp323 = 0 
    wp331 = 0 
    wp332 = 0 
    wp333 = 0 
    Wf = np.zeros(3,3) 

    if c200 <= csmall:
        if c020 <= csmall: 
            M0 = [ 1, 0, c002, c003, c004] 
            [ N0, W0 ] = three_node_hyqmom( M0 ) 

            rho[1] = 0 
            rho[2] = 1 
            rho[3] = 0 
            rho22 = 1 
            rho221 = N0[1] 
            rho222 = N0[2] 
            rho223 = N0[3] 
            up = 0*up 
            wp221 = W0[1] 
            wp222 = W0[2] 
            wp223 = W0[3] 
        else:
            M1 = [ 1, 0, 0, c020, c011, c002, c030, c003, c040, c004] 
            [ N1, V1, W1 ] = nine_node_10mom_hycqmom_2D( M1 ) 
            rho[1] = 0 
            rho[2] = 1 
            rho[3] = 0 
            rho12 = 0 
            rho21 = 1 
            rho22 = 1 
            rho23 = 1 
            rho31 = 0 
            rho211 = N1[1] 
            rho212 = N1[2] 
            rho213 = N1[3] 
            rho221 = N1[4] 
            rho222 = N1[5] 
            rho223 = N1[6] 
            rho231 = N1[7] 
            rho232 = N1[8] 
            rho233 = N1[9] 
            up = 0*up 
            vp21 = V1[1] 
            vp22 = V1[5] 
            vp23 = V1[9] 
            wp211 = W1[1] 
            wp212 = W1[2] 
            wp213 = W1[3] 
            wp221 = W1[4] 
            wp222 = W1[5] 
            wp223 = W1[6] 
            wp231 = W1[7] 
            wp232 = W1[8] 
            wp233 = W1[9] 
    elif c020 <= csmall :
        M2 = [ 1, 0, 0, c200, c101, c002, c300, c003, c400, c004] 
        [ N2, U2, W2 ] = nine_node_10mom_hycqmom_2D( M2 ) 

        rho[1] = 1 
        rho[2] = 1 
        rho[3] = 1 
        rho12 = 1 
        rho22 = 1 
        rho32 = 1 
        rho121 = N2[1] 
        rho122 = N2[2] 
        rho123 = N2[3] 
        rho221 = N2[4] 
        rho222 = N2[5] 
        rho223 = N2[6] 
        rho321 = N2[7] 
        rho322 = N2[8] 
        rho323 = N2[9] 
        up[1] = U2[1] 
        up[2] = U2[5] 
        up[3] = U2[9] 
        wp121 = W2[1] 
        wp122 = W2[2] 
        wp123 = W2[3] 
        wp221 = W2[4] 
        wp222 = W2[5] 
        wp223 = W2[6] 
        wp321 = W2[7] 
        wp322 = W2[8] 
        wp323 = W2[9] 
    elif c002 <= csmall :
        M3 = [ 1, 0, 0, c200, c110, c020, c300, c030, c400, c040] 
        [ N3, U3, V3 ] = nine_node_10mom_hycqmom_2D( M3 )
        rho[1]= 1 
        rho[2]= 1 
        rho[3]= 1 
        rho11 = N3[1] 
        rho12 = N3[2] 
        rho13 = N3[3] 
        rho21 = N3[4] 
        rho22 = N3[5] 
        rho23 = N3[6] 
        rho31 = N3[7] 
        rho32 = N3[8] 
        rho33 = N3[9] 
        up[1]= U3[1] 
        up[2]= U3[5] 
        up[3]= U3[9] 
        vp11 = V3[1] 
        vp12 = V3[2] 
        vp13 = V3[3] 
        vp21 = V3[4] 
        vp22 = V3[5] 
        vp23 = V3[6] 
        vp31 = V3[7] 
        vp32 = V3[8] 
        vp33 = V3[9] 
    else:
        M4 = [ 1, 0, 0, c200, c110, c020, c300, c030, c400, c040] 
        U4 = np.zeros(4)
        [ N4, U4 , V4 ] = nine_node_10mom_hycqmom_2D( M4 ) 

        rho11 = N4[1]/[N4[1]+N4[2]+N4[3]] 
        rho12 = N4[2]/[N4[1]+N4[2]+N4[3]] 
        rho13 = N4[3]/[N4[1]+N4[2]+N4[3]] 
        rho21 = N4[4]/[N4[4]+N4[5]+N4[6]] 
        rho22 = N4[5]/[N4[4]+N4[5]+N4[6]] 
        rho23 = N4[6]/[N4[4]+N4[5]+N4[6]] 
        rho31 = N4[7]/[N4[7]+N4[8]+N4[9]] 
        rho32 = N4[8]/[N4[7]+N4[8]+N4[9]] 
        rho33 = N4[9]/[N4[7]+N4[8]+N4[9]] 
        Vf[1] = rho11*V4[1]+rho12*V4[2]+rho13*V4[3] 
        Vf[2] = rho21*V4[4]+rho22*V4[5]+rho23*V4[6] 
        Vf[3] = rho31*V4[7]+rho32*V4[8]+rho33*V4[9] 
        vp11 = V4[1]-Vf[1] 
        vp12 = V4[2]-Vf[1] 
        vp13 = V4[3]-Vf[1] 
        vp21 = V4[4]-Vf[2] 
        vp22 = V4[5]-Vf[2] 
        vp23 = V4[6]-Vf[2] 
        vp31 = V4[7]-Vf[3] 
        vp32 = V4[8]-Vf[3] 
        vp33 = V4[9]-Vf[3] 
        scale1 = math.sqrt(c200) 
        scale2 = math.sqrt(c020) 
        Rho1 = diag(rho) 
        Rho2 = [ [ rho11, rho12, rho13], [rho21, rho22, rho23], [rho31, rho32, rho33]] 
        Vp2 = [ [vp11, vp12, vp13], [vp21, vp22, vp23], [vp31, vp32, vp33] ] 
        Vp2s = Vp2/scale2 
        RAB = Rho1*Rho2 
        UAB = [up, up, up] 
        UABs = UAB/scale1 
        VAB = Vp2 + diag(Vf)*ones(3) 
        VABs = VAB/scale2 
        C01 = np.multiply(RAB,VABs)
        Vc0 = ones(3) 
        Vc1 = UABs 
        Vc2 = Vp2s 
        A1 = sum(sum(np.multiply(C01,Vc1)))  
        A2 = sum(sum(np.multiply(C01,Vc2)))  

        c101s = c101/scale1  
        c011s = c011/scale2 
        if c101s**2 >= c002*(1 - small) :
            c101s = sign(c101s)*math.sqrt(c002) 
        elif c011s**2 >= c002*(1 - small) :
            c110s = c110/scale1/scale2 
            c011s = sign(c011s)*math.sqrt(c002) 
            c101s = c110s*c011s 

        b0 = 0
        b1 = c101s
        b2 = 0
        if A2 > wsmall:
            b2 = ( c011s - A1*b1 )/A2 
        Wf = b0*Vc0 + b1*Vc1 + b2*Vc2  
        SUM002 = sum(sum(np.multiply(RAB,Wf**2))) 
        mu2 = c002 - SUM002 
        mu2 = max(0,mu2) 
        q = 0 ; eta = 1 
        if mu2 > csmall:
            SUM1 = mu2**(3/2) 
            SUM3 = sum(sum(np.multiply(RAB,Wf**3))) 
            q = ( c003 - SUM3 )/SUM1 
            SUM2 = mu2**2 
            SUM4 = sum(sum(np.multiple(RAB,Wf**4))) + 6*SUM002*mu2 
            eta = ( c004 - SUM4 )/SUM2 
            if eta < (q**2 + 1):
                if abs(q) > verysmall:
                    slope = (eta - 3)/q 
                    det = 8 + slope**2 
                    qp = 0.5*( slope + math.sqrt(det) ) 
                    qm = 0.5*( slope - math.sqrt(det) ) 
                    if sign(q) == 1:
                        q = qp 
                    else:
                        q = qm 
                else:
                    q = 0 
                eta = q**2 + 1 
        mu3 = q*mu2**(3/2) 
        mu4 = eta*mu2**2 
        M5 = [1, 0, mu2, mu3, mu4] 
        [ rh11, up11 ] = three_node_hyqmom( M5 ) 
        rho111 = rh11[1] 
        rho112 = rh11[2] 
        rho113 = rh11[3] 
        wp111 = up11[1] 
        wp112 = up11[2] 
        wp113 = up11[3] 
        rh12 = rh11 
        up12 = up11 
        rho121 = rh12[1] 
        rho122 = rh12[2] 
        rho123 = rh12[3] 
        wp121 = up12[1] 
        wp122 = up12[2] 
        wp123 = up12[3] 
        rh13 = rh11 
        up13 = up11 
        rho131 = rh13[1] 
        rho132 = rh13[2] 
        rho133 = rh13[3] 
        wp131 = up13[1] 
        wp132 = up13[2] 
        wp133 = up13[3] 
        rh21 = rh11 
        up21 = up11 
        wp211 = up21[1] 
        wp212 = up21[2] 
        wp213 = up21[3] 
        rho211 = rh21[1] 
        rho212 = rh21[2] 
        rho213 = rh21[3] 
        rh22 = rh11 
        up22 = up11 
        wp221 = up22[1] 
        wp222 = up22[2] 
        wp223 = up22[3] 
        rho221 = rh22[1] 
        rho222 = rh22[2] 
        rho223 = rh22[3] 
        rh23 = rh11 
        up23 = up11 
        wp231 = up23[1] 
        wp232 = up23[2] 
        wp233 = up23[3] 
        rho231 = rh23[1] 
        rho232 = rh23[2] 
        rho233 = rh23[3] 
        rh31 = rh11 
        up31 = up11 
        rho311 = rh31[1] 
        rho312 = rh31[2] 
        rho313 = rh31[3] 
        wp311 = up31[1] 
        wp312 = up31[2] 
        wp313 = up31[3] 
        rh32 = rh11 
        up32 = up11 
        rho321 = rh32[1] 
        rho322 = rh32[2] 
        rho323 = rh32[3] 
        wp321 = up32[1] 
        wp322 = up32[2] 
        wp323 = up32[3] 
        rh33 = rh11 
        up33 = up11 
        rho331 = rh33[1] 
        rho332 = rh33[2] 
        rho333 = rh33[3] 
        wp331 = up33[1] 
        wp332 = up33[2] 
        wp333 = up33[3] 

    N[1] = rho[1]*rho11*rho111
    N[2] = rho[1]*rho11*rho112
    N[3] = rho[1]*rho11*rho113
    N[4] = rho[1]*rho12*rho121
    N[5] = rho[1]*rho12*rho122
    N[6] = rho[1]*rho12*rho123
    N[7] = rho[1]*rho13*rho131
    N[8] = rho[1]*rho13*rho132
    N[9] = rho[1]*rho13*rho133
    N[10]= rho[2]*rho21*rho211
    N[11]= rho[2]*rho21*rho212
    N[12]= rho[2]*rho21*rho213
    N[13]= rho[2]*rho22*rho221
    N[14]= rho[2]*rho22*rho222 
    N[15]= rho[2]*rho22*rho223
    N[16]= rho[2]*rho23*rho231
    N[17]= rho[2]*rho23*rho232
    N[18]= rho[2]*rho23*rho233
    N[19]= rho[3]*rho31*rho311
    N[20]= rho[3]*rho31*rho312
    N[21]= rho[3]*rho31*rho313
    N[22]= rho[3]*rho32*rho321
    N[23]= rho[3]*rho32*rho322
    N[24]= rho[3]*rho32*rho323
    N[25]= rho[3]*rho33*rho331
    N[26]= rho[3]*rho33*rho332
    N[27]= rho[3]*rho33*rho333
    N = m000*N 

    U[1] = up[1]
    U[2] = up[1]
    U[3] = up[1]
    U[4] = up[1]
    U[5] = up[1]
    U[6] = up[1]
    U[7] = up[1]
    U[8] = up[1]
    U[9] = up[1]
    U[10]= up[2] 
    U[11]= up[2] 
    U[12]= up[2] 
    U[13]= up[2]
    U[14]= up[2]
    U[15]= up[2] 
    U[16]= up[2] 
    U[17]= up[2] 
    U[18]= up[2] 
    U[19]= up[3]
    U[20]= up[3]
    U[21]= up[3]
    U[22]= up[3]
    U[23]= up[3]
    U[24]= up[3]
    U[25]= up[3]
    U[26]= up[3]
    U[27]= up[3]
    U = bu + U 

    V[1] = Vf[1]+vp11
    V[2] = Vf[1]+vp11
    V[3] = Vf[1]+vp11
    V[4] = Vf[1]+vp12
    V[5] = Vf[1]+vp12
    V[6] = Vf[1]+vp12
    V[7] = Vf[1]+vp13
    V[8] = Vf[1]+vp13
    V[9] = Vf[1]+vp13
    V[10]= Vf[2]+vp21
    V[11]= Vf[2]+vp21
    V[12]= Vf[2]+vp21
    V[13]= Vf[2]+vp22 
    V[14]= Vf[2]+vp22 
    V[15]= Vf[2]+vp22
    V[16]= Vf[2]+vp23 
    V[17]= Vf[2]+vp23
    V[18]= Vf[2]+vp23
    V[19]= Vf[3]+vp31
    V[20]= Vf[3]+vp31
    V[21]= Vf[3]+vp31
    V[22]= Vf[3]+vp32
    V[23]= Vf[3]+vp32
    V[24]= Vf[3]+vp32
    V[25]= Vf[3]+vp33
    V[26]= Vf[3]+vp33
    V[27]= Vf[3]+vp33
    V = bv + V 

    W[1] = Wf[1,1]+wp111
    W[2] = Wf[1,1]+wp112
    W[3] = Wf[1,1]+wp113
    W[4] = Wf[1,2]+wp121
    W[5] = Wf[1,2]+wp122
    W[6] = Wf[1,2]+wp123
    W[7] = Wf[1,3]+wp131
    W[8] = Wf[1,3]+wp132
    W[9] = Wf[1,3]+wp133
    W[10]= Wf[2,1]+wp211 
    W[11]= Wf[2,1]+wp212
    W[12]= Wf[2,1]+wp213
    W[13]= Wf[2,2]+wp221 
    W[14]= Wf[2,2]+wp222 
    W[15]= Wf[2,2]+wp223
    W[16]= Wf[2,3]+wp231
    W[17]= Wf[2,3]+wp232
    W[18]= Wf[2,3]+wp233
    W[19]= Wf[3,1]+wp311
    W[20]= Wf[3,1]+wp312
    W[21]= Wf[3,1]+wp313
    W[22]= Wf[3,2]+wp321
    W[23]= Wf[3,2]+wp322
    W[24]= Wf[3,2]+wp323
    W[25]= Wf[3,3]+wp331
    W[26]= Wf[3,3]+wp332
    W[27]= Wf[3,3]+wp333
    W = bw + W 


def conditional_12(moments, indices):

    num_nodes_1 = ( moments[:,0].max() + 1 ) // 2
    num_nodes_2 = ( moments[:,1].max() + 1 ) // 2

    wts_1, xi_1 = wheeler( moments, indices )

    print(wts_1)
    
    return weights, abscissas

def conditional_21(moments, indices):

    weights   = np.array([])
    abscissas = np.array([])    
    return weights, abscissas
