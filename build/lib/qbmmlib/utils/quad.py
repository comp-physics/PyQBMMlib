import math
import numpy as np


def quadrature_1d(weights, abscissas, moment_index):
    xi_to_idx = abscissas ** moment_index
    q = np.dot(weights, xi_to_idx)
    return q


def quadrature_2d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * (abscissas[0][i] ** moment_index[0])
            * (abscissas[1][i] ** moment_index[1])
        )
    return q


def quadrature_3d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * abscissas[0, i] ** moment_index[0]
            * abscissas[1, i] ** moment_index[1]
            * abscissas[2, i] ** moment_index[2]
        )
    return q
