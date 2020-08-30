import numpy as np

from scipy import sqrt, zeros
from scipy.linalg import eig

###
### Invesion methods for 1D problems
###
def wheeler(mom,adaptive):

    adaptive = False
    # print('inversion: Warning: Adaptive Wheeler not implemented. Returning empty arrays')

    # From Bo Kong code in old_python_qmom
    # def adaptive_Wheeler(mom):
    """ Return weights,  nodes, and number of nodes using adaptive Wheeler
    algorithm.
    """

    # SHB: need to convert all this stuff to numpy?

    n = len(mom)/2

    # SHB let's make adaptive and non-adaptive wheeler one routine with if statemtns
    # if adaptive:
    # Adaptivity parameters
    rmax = 1e-8
    eabs = 1e-8
    cutoff = 0
    
    # Check if moments are unrealizable.
    if mom[0] <= 0:
        print("Moments are NOT realizable, moment[0] <= 0.0. Program exits.")
        exit()

    if n == 1 or (adaptive and mom[0] < rmax):
        w = mom[0]
        x = mom[1]/mom[0]
        return w, x

    # Set modified moments equal to input moments.
    nu = mom

    # Construct recurrence matrix
    ind = n
    a = zeros(ind)
    b = zeros(ind)
    sig = zeros((2*ind+1, 2*ind+1))

    for i in range(1, 2*ind+1):
        sig[1,i] = nu[i-1]

    a[0] = nu[1]/nu[0]
    b[0] = 0

    for k in range(2, ind+1):
        for l in range(k, 2*ind-k+2):
            sig[k, l] = sig[k-1, l+1]-a[k-2]*sig[k-1, l]-b[k-2]*sig[k-2, l]
        a[k-1] = sig[k, k+1]/sig[k, k]-sig[k-1, k]/sig[k-1, k-1]
        b[k-1] = sig[k, k]/sig[k-1, k-1]

    # Find maximum n using diagonal element of sig
    if adaptive:
        for k in range(ind,1,-1):
            if sig[k,k] <= cutoff:
                n = k-1
                if n == 1:
                    w = mom[0]
                    x = mom[1]/mom[0]
                    return w, x

    # Use maximum n to re-calculate recurrence matrix
    a = zeros(n)
    b = zeros(n)
    w = zeros(n)
    x = zeros(n)
    sig = zeros((2*n+1,2*n+1))
    for i in range(1,2*n+1):
        sig[1,i] = nu[i-1]

    a[0] = nu[1] / nu[0]
    b[0] = 0
    for k in range(2, n+1):
        for l in range(k, 2*n-k+2):
            sig[k, l] = sig[k-1, l+1]-a[k-2]*sig[k-1, l]-b[k-2]*sig[k-2, l]
        a[k-1] = sig[k, k+1]/sig[k, k]-sig[k-1, k]/sig[k-1, k-1]
        b[k-1] = sig[k, k]/sig[k-1, k-1]

    # Check if moments are unrealizable
    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! Program exits.")
        exit()

    # Setup Jacobi matrix for n-point quadrature, adapt n using rmin and eabs
    for n1 in range(n,0,-1):
        if n1 == 1:
            w = mom[0]
            x = mom[1]/mom[0]
            return w, x
        z = zeros((n1, n1))
        for i in range(n1-1):
            z[i, i] = a[i]
            z[i, i+1] = sqrt(b[i+1])
            z[i+1, i] = z[i, i+1]
        z[n1-1, n1-1] = a[n1-1]

        # Compute weights and abscissas
        eigenvalues, eigenvectors = eig(z)
        idx = eigenvalues.argsort()
        x = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        w = mom[0]*eigenvectors[0, :]**2

        # SHB: Let's combine adaptive and non-adaptive into one routine using an if-statement here

        # Adaptive conditions. When both satisfied, return the results.
        if adaptive:
            dab = zeros(n1)
            mab = zeros(n1)

            for i in range(n1-1, 0, -1):
                dab[i] = min(abs(x[i]-x[0:i]))
                mab[i] = max(abs(x[i]-x[0:i]))

            mindab = min(dab[1:n1])
            maxmab = max(mab[1:n1])
            if n1 == 2:
                maxmab = 1

            if min(w)/max(w) > rmax and mindab/maxmab > eabs:
                return w, x
        else:
            return w, x

    # weights   = np.array([])
    # abscissas = np.array([])
    # return weights, abscissas

def hyperbolic(moms, max_skewness = 30):



    n = len(moms)/2

    if n=2:
        w = zeros(n)
        x = w
        bx = moms[2]/moms[1]
        d2 = moms[3]/moms[1]
        c2 = d2 - bx**2.
        w[1] = moms[1]/2.
        w[2] = w[1]
        if c2 < 10**(-12):
            c2 = 10**(-12)
        x[1] = bx - math.sqrt(c2)
        x[2] = bx + math.sqrt(c2)
        return w, x
   
    # needs to be ported to python
    # refer to HYQMOM3 in QBMMlib Mathematica
    if n=3:
        etasmall = 10**(-10); 
        verysmall = 10**(-14); 
        realsmall = 10**(-14); 
        n = Table[0, {i, 3}]; 
        u = n; 
        If[moms[1] <= verysmall, 
            n[2] = moms[1]; 
            Return[{n, u}, Module];
        ]; 
        bu = moms[2]/moms[1]; 
        d2 = moms[3]/moms[1]; 
        d3 = moms[4]/moms[1]; 
        d4 = moms[5]/moms[1]; 
        c2 = d2 - bu**2; 
        c3 = d3 - 3*bu d2 + 2 bu**3; 
        c4 = d4 - 4*bu d3 + 6 bu**2 d2 - 3 bu**4;
        realizable = c2 c4 - c2**3 - c3**2; 
        if c2 < 0: 
            if c2 < -verysmall:
                Print["Error: c2 negative in three node HYQMOM"];Abort[];]; 
            c2 = 0
            c3 = 0 
            c4 = 0
        else:
            if realizable < 0:
                if c2 >= etasmall:
                        q = c3/Sqrt[c2]/c2; 
                        eta = c4/c2/c2; 
                        if Abs[q] > verysmall :
                            slope = (eta - 3)/q; 
                            det = 8 + slope**2;
                            qp = 0.5 (slope + Sqrt[det]); 
                            qm = 0.5 (slope - Sqrt[det]); 
                            if Sign[q] = 1: 
                                q = qp
                            else:
                                q = qm
                        else:
                            q = 0;
                        ]; 
                        eta = q**2 + 1; 
                        c3 = q Sqrt[c2] c2; c4 = eta c2**2; 
                        if realizable < -10**(-6):
                            Print["Error: c4 small in HYQMOM3"];
                            Abort[];
                    else: 
                        c3 = 0; 
                        c4 = c2**2;
        scale = Sqrt[c2]; 
        If[c2 >= etasmall, 
            q = c3/Sqrt[c2]/c2; 
            eta = c4/c2/c2;
        , 
            q = 0; 
            eta = 1;
        ]; 
        If[
            q**2 > qmax**2, 
            slope = (eta - 3)/q; 
            q = qmax Sign[q]; 
            eta = 3 + slope q; 
            realizable = eta - 1 - q**2; 
            If[realizable < 0, eta = 1 + q**2];
        ]; 
        ups[1] = (q - Sqrt[4 eta - 3 q**2])/2; 
        ups[2] = 0; 
        ups[3] = (q + Sqrt[4 eta - 3 q**2])/2; 
        dem = 1/Sqrt[4 eta - 3 q**2]; 
        prod = -ups[1] ups[3]; 
        prod = Max[prod, 1 + realsmall]; 
        rho[1] = -dem/ups[1]; 
        rho[2] = 1 - 1/prod; 
        rho[3] = dem/ups[3]; 
        srho = Sum[rho[i], {i, 3}]; 
        Do[rho[i] = rho[i]/srho, {i, 3}]; 
        scales = Sum[rho[i] ups[i]**2, {i, 3}]/Sum[rho[i], {i, 3}]; 
        Do[up[i] = ups[i] scale/Sqrt[scales], {i, 3}]; 
        If[
            Min[Table[rho[i], {i, 3}] < 0, 
            Print["Error: Negative weight in HYQMOM"];Abort[];
        ]; 
        n[1] = rho[1]; 
        n[2] = rho[2]; 
        n[3] = rho[3]; 
        n = moms[1] n; 

        u[1] = up[1]; 
        u[2] = up[2]; 
        u[3] = up[3]; 
        u = bu + u; 

        Return[{n, u}, Module];
    ];


    print('inversion: Warning: n is not 2 or 3 aborting')

    weights   = np.array([])
    abscissas = np.array([])    
    return weights, abscissas

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
