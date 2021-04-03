import math
import numpy as np
from inversion import *
from numba import njit
from itertools import product


@njit
def projection(
        weights, abscissas, indices,
        num_coords, num_nodes):
    moments = np.zeros(indices.shape[0])
    ni = len(indices)
    for i in range(ni):
        if num_coords == 3:
            moments[i] = quadrature_3d(
                weights, abscissas, indices[i], num_nodes
            )
        if num_coords == 2:
            moments[i] = quadrature_2d(
                weights, abscissas, indices[i], num_nodes
            )
        # if num_coords == 1:
        #     moments[i] = quadrature_1d(weights, abscissas, indices[i])
    return moments


@njit
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


@njit
def quadrature_2d(weights, abscissas, moment_index, num_quadrature_nodes):
    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * (abscissas[0][i] ** moment_index[0])
            * (abscissas[1][i] ** moment_index[1])
        )
    return q


@njit
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


@njit
def flux_quadrature(wts_left, xi_left, wts_right, xi_right, indices, num_moments, num_nodes):
    # Input: Local weights/abscissas at left/right
    # Output: Local fluxes at left/right

    flux = np.zeros(num_moments)
    for m in range(num_moments):
        for n in range(num_nodes):
            # compute local fluxes
            flux_left = (
                wts_left[n]
                * xi_left[0, n]**indices[m, 0]
                * xi_left[1, n]**indices[m, 1]

            )
            flux_right = (
                wts_right[n]
                * xi_right[0, n]**indices[m, 0]
                * xi_right[1, n]**indices[m, 1]

            )

            if len(indices[0]) == 3:
                flux_left *= (
                    xi_left[2, n]**indices[m, 2]
                    )
                flux_right *= (
                    xi_right[2, n]**indices[m, 2]
                    )

            # limiter
            flux_left = flux_left * max(xi_left[0, n], 0)
            flux_right = flux_right * min(xi_right[0, n], 0)
            # quadrature
            flux[m] += flux_left + flux_right
        
    return flux
        

@njit
def compute_fluxes(weights, abscissas, indices, num_points, num_moments, num_nodes, flux):
    # Input: Global weights, abscissas 
    # Compute: Global flux quadrature
    # Output: All domain fluxes

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
        f_right = flux_quadrature(wts_left, xi_left, wts_right, xi_right, indices, num_moments, num_nodes)
        
        # Reconstruct flux
        flux[i_point] = f_left - f_right

@njit
def domain_project(state, indices, weights, abscissas, num_points, num_coords, num_nodes):
    # Input: Weights, abscissas, indices, num_points, num_coords, num_nodes
    # Output: State

    for i_point in range(1, num_points-1):
        state[i_point] = projection(
                weights[i_point], 
                abscissas[i_point], 
                indices, 
                num_coords, 
                num_nodes)

    # Boundary conditions
    state[0] = projection(weights[-2], abscissas[-2], indices, num_coords, num_nodes)
    state[-1] = projection(weights[1], abscissas[1], indices, num_coords, num_nodes)

@njit
def domain_invert_3d(state, indices, weights, abscissas, num_points, num_coords, num_nodes):
    # Input: State, indices, num_points, num_coords, num_nodes
    # Output: Weights, abscissas

    for i_point in range(1, num_points-1):
        # Invert
        xi, wts = chyqmom27(state[i_point], indices)
        abscissas[i_point] = xi.T
        weights[i_point] = wts

    # Boundary conditions
    xi, wts = chyqmom27(state[0], indices)
    abscissas[0] = xi.T
    weights[0] = wts

    xi, wts = chyqmom27(state[-1], indices)
    abscissas[-1] = xi.T
    weights[-1] = wts

    return weights, abscissas

@njit
def domain_invert_2d(state, indices, weights, abscissas, num_points, num_coords, num_nodes):

    for i_point in range(1, num_points-1):
        # print(' ipt', i_point)
        # print('State before', state[i_point])
        # Invert
        xi, wts = chyqmom9(state[i_point], indices)
        abscissas[i_point] = xi.T
        weights[i_point] = wts
        # Project
        state[i_point] = projection(wts, xi.T, indices, num_coords, num_nodes)
        # print('State after', state[i_point])
        if np.isnan(abscissas[i_point]).any() or \
            np.isnan(weights[i_point]).any():
            ii = np.where(np.isnan(abscissas[i_point]))[0]
            # print('state',  iii, 'is nan:', state[iii])
            print('abs:', abscissas[i_point])
            print('w:',weights[i_point])
            print('found nan in wts or absc')
            raise Exception()

    # Boundary conditions
    state[0] = projection(weights[-2], abscissas[-2], indices, num_coords, num_nodes)
    state[-1] = projection(weights[1], abscissas[1], indices, num_coords, num_nodes)

    # print('get pt0')
    # print(state[0])
    xi, wts = chyqmom9(state[0], indices)
    abscissas[0] = xi.T
    weights[0] = wts
    # print('get pt-1')
    # print(state[-1])
    xi, wts = chyqmom9(state[-1], indices)
    abscissas[-1] = xi.T
    weights[-1] = wts
    # print(' finish update quad 2d')

    print('max = ', np.max(state[:,:]))
    if np.isnan(state).any():
        print('found nan')
        ii = np.where(np.isnan(state))[0]
        iii = ii[0]
        print('state',  iii, 'is nan:', state[iii])
        raise Exception()
