import math
import numpy as np
from numba import jit

@jit(nopython=True)
def shb_quadrature_1d(weights, abscissas, moment_index):
    xi_to_idx = abscissas ** moment_index
    q = np.dot( weights, xi_to_idx )
    return q

@jit(nopython=True)
def shb_quadrature_2d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.
    for i in range( num_quadrature_nodes ):
        q += weights[i]*(abscissas[0][i]**moment_index[0]) * \
               (abscissas[1][i]**moment_index[1])
    return q

@jit(nopython=True)
def shb_project_1d(weights, abscissas, indices):
    moments = np.zeros( len(indices) )
    for i_index in range( len(indices) ):
        moments[i_index] = shb_quadrature_1d(weights, abscissas, indices[i_index])
    return moments

@jit(nopython=True)
def shb_project_2d(weights, abscissas, indices, num_quadrature_nodes):
    moments = np.zeros( len(indices) )
    for i_index in range( len(indices) ):
        moments[i_index] = shb_quadrature_2d(weights, abscissas, indices[i_index], num_quadrature_nodes)
    return moments

# @jit(nopython=True)
# def shb_project(weights, abscissas, indices, num_quadrature_nodes, num_internal_coords):
#     moments = np.zeros( len(indices) )
#     for i_index in range( len(indices) ):
#         if num_internal_coords == 1:
#             moments[i_index] = shb_quadrature_1d(weights, abscissas, indices[i_index])
#         else:
#             moments[i_index] = shb_quadrature_2d(weights, abscissas, indices[i_index], num_quadrature_nodes)
#     return moments
