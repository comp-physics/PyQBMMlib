import numpy as np

###
### Invesion methods for 1D problems
###
def classic_wheeler(moments):

    print('inversion: Warning: Classic Wheeler not implemented. Returning empty arrays')
    
    weights   = np.array([])
    abscissas = np.array([])
    return weights, abscissas

def adaptive_wheeler(moments):

    print('inversion: Warning: Adaptive Wheeler not implemented. Returning empty arrays')

    weights   = np.array([])
    abscissas = np.array([])
    return weights, abscissas

def hyperbolic(moments, max_skewness = 30):

    print('inversion: Warning: Hyperbolic QMOM not implemented. Returning empty arrays')

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
