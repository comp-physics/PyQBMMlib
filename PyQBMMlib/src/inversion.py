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
    for k in range(2,n+1):
        for l in range(k,2*n-k+2):
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

    weights   = np.array([])
    abscissas = np.array([])    
    return weights, abscissas

def conditional_hyperbolic(moments, indices, max_skewness = 30):

    print('inversion: Warning: Conditional Hyperbolic QMOM not implemented. Returning empty arrays')

    weights   = np.array([])
    abscissas = np.array([])    
    return weights, abscissas
