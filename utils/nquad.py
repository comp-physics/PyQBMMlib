import math
import numpy as np
from numba import jit
from itertools import product


@jit(nopython=True)
def quadrature_1d(weights, abscissas, moment_index):
    """
    This function computes quadrature in 1D
    Inputs:
    - weights: quadrature weights
    Return:
    """
    xi_to_idx = abscissas ** moment_index
    q = np.dot(weights, xi_to_idx)
    return q


@jit(nopython=True)
def quadrature_2d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * (abscissas[0][i] ** moment_index[0])
            * (abscissas[1][i] ** moment_index[1])
        )
    return q

@jit(nopython=True)
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

@jit(nopython=True)
def flux_quadrature(wts_left, xi_left, wts_right, xi_right, indices, num_moments, num_nodes):
    flux = np.zeros(num_moments)
    for m in range(num_moments):
        for n in range(num_nodes):
            # compute local fluxes
            flux_left = (
                wts_left[n]
                * xi_left[0, n]**indices[m, 0]
                * xi_left[1, n]**indices[m, 1]
                * xi_left[2, n]**indices[m, 2]
            )
            flux_right = (
                wts_right[n]
                * xi_right[0, n]**indices[m, 0]
                * xi_right[1, n]**indices[m, 1]
                * xi_right[2, n]**indices[m, 2]
            )
            # limiter
            flux_left = flux_left * max(xi_left[0, n], 0)
            flux_right = flux_right * min(xi_right[0, n], 0)
            # quadrature
            flux[m] += flux_left + flux_right
        
    return flux
        

@jit(nopython=True)
def compute_fluxes(weights, abscissas, indices, num_moments, num_nodes, flux):

    for i_point in range(1, num_points-1):

        # Compute left flux
        wts_left = weights[i_point-1]
        wts_right = weights[i_point]
        xi_left = abscissas[i_point-1]
        xi_right = abscissas[i_point]
        f_left = flux_quadrature(wts_left, xi_left, wts_right, xi_right, 
                                 indices, num_moments, num_nodes)
        
            # Compute right flux
        wts_left = wts_right
        xi_left = xi_right
        wts_right = weights[i_point+1]
        xi_right = abscissas[i_point+1]
        f_right = flux_quadrature(wts_left, xi_left, wts_right, xi_right, 
                                  indices, num_moments, num_nodes)
        
        # Reconstruct flux
        self.flux[i_point] = f_left - f_right
