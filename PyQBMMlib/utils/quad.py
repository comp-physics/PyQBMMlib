import math
import numpy as np
from numba  import jit


# @jit
# @jit(nopython=True)
# def myquad(w,x0,x1,i0,i1):
#     return w*(x0**i0)*(x1**i1)

@jit(nopython=True)
def shb_quadrature(weights, abscissas, moment_index, num_quadrature_nodes, num_internal_coords):
    q = 0.
    for i in range( num_quadrature_nodes ):
        # q += myquad(weights[i],abscissas[0][i],abscissas[1][i],moment_index[0],moment_index[1])
        q += weights[i]*(abscissas[0][i]**moment_index[0]) * \
               (abscissas[1][i]**moment_index[1])
    return q

@jit(nopython=True)
def shb_project(weights, abscissas, indices, num_quadrature_nodes, num_internal_coords, num_indices):
    moments = np.zeros( num_indices )
    for i_index in range( num_indices ):
        moments[i_index] = shb_quadrature(weights, abscissas, indices[i_index], num_quadrature_nodes, num_internal_coords)
    return moments
