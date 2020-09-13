import numpy as np

from scipy import sqrt, zeros
from scipy.linalg import eig

import math

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

def hyperbolic(moms, config):
    
    max_skewness = config['max_skewness']
    if len(moms) == 3:
        n = 2
    elif len(moms) == 5:
        n = 3
    else:
        print('HyQMOM inversion. Fatal error, n != 2 or 3. Aborting')
        exit()

    if n == 2:
        return hyperbolic2(moms)
    elif n == 3:
        return hyperbolic3(moms,max_skewness)

def hyperbolic2(moms):
    n = 2
    w = zeros(n)
    x = zeros(n)

    w[0] = moms[0]/2.
    w[1] = w[0]

    bx = moms[1]/moms[0]
    d2 = moms[2]/moms[0]
    c2 = d2 - bx**2.

    if c2 < 10**(-12):
        c2 = 10**(-12)
    x[0] = bx - math.sqrt(c2)
    x[1] = bx + math.sqrt(c2)

    return x,w

def hyperbolic3(moms,qmax):
    # needs to be ported to python
    # refer to HYQMOM3 in QBMMlib Mathematica
    n = 3
    etasmall  = 10**(-10) 
    verysmall = 10**(-14) 
    realsmall = 10**(-14) 

    w   = zeros(n)
    x   = zeros(n) 
    xp  = zeros(n)
    xps = zeros(n)
    rho = zeros(n)

    if moms[0] <= verysmall:
        w[1] = moms[0]
        return x,w

    bx = moms[1]/moms[0] 
    d2 = moms[2]/moms[0] 
    d3 = moms[3]/moms[0] 
    d4 = moms[4]/moms[0] 
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

    if q**2 > qmax**2:
        slope = (eta - 3)/q 
        if q > 0:
            q = qmax
        else:
            q = -qmax
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

    w = moms[0]*rho
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

def conditional_hyperbolic(moments, config, max_skewness = 30):

    idx = config['indices']
    num_dim = len(idx)

    if num_dim == 6:
        return chyqmom4( moments, idx )
    elif num_dim == 10:
        return chyqmom9( moments, idx )

    # print('inversion: Warning: Conditional Hyperbolic QMOM not implemented. Returning empty arrays')
    # weights   = np.array([])
    # abscissas = np.array([])    
    # return weights, abscissas

def chyqmom4(moments, idx):

    mom00 = moments[idx.index([0,0])]
    mom10 = moments[idx.index([1,0])]
    mom01 = moments[idx.index([0,1])]
    mom20 = moments[idx.index([2,0])]
    mom11 = moments[idx.index([1,1])]
    mom02 = moments[idx.index([0,2])]

    n = 4
    w = zeros(n)
    x = zeros(n)
    y = zeros(n)

    bx  = mom10/mom00
    by  = mom01/mom00
    d20 = mom20/mom00
    d11 = mom11/mom00
    d02 = mom02/mom00

    c20 = d20 - bx**2.
    c11 = d11 - bx*by
    c02 = d02 - by**2.

    M1 = [1, 0, c20]
    xp, rho = hyperbolic2(M1) 
    yf = c11*xp/c20 
    mu2avg = c02 - sum(rho*yf**2)
    mu2avg = max(mu2avg, 0)
    mu2 = mu2avg 
    M3 = [ 1, 0, mu2 ] 
    xp3, rh3 = hyperbolic2(M3)
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

def chyqmom9(moments, indices):

    return x,w


def conditional_12(moments, indices):

    num_nodes_1 = ( moments[:,0].max() + 1 ) // 2
    num_nodes_2 = ( moments[:,1].max() + 1 ) // 2

    wts_1, xi_1 = wheeler( moments, indices )

    print wts_1
    
    return weights, abscissas

def conditional_21(moments, indices):

    weights   = np.array([])
    abscissas = np.array([])    
    return weights, abscissas
