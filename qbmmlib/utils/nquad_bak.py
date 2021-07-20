import numpy as np
from inversion import chyqmom9, chyqmom27
from numba import njit

@njit
def projection(
        weights, abscissas, indices,
        num_coords, num_nodes):

    moments = np.zeros(indices.shape[0])
    ni = len(indices)
    for i in range(ni):
        moments[i] = quadrature(
            weights, abscissas, indices[i], num_coords
            )
    return moments


@njit
def quadrature(weights, abscissas, moment_index, num_coords):
    if num_coords == 3: 
        return np.sum(
                weights[:]
                * abscissas[0, :] ** moment_index[0]
                * abscissas[1, :] ** moment_index[1]
                * abscissas[2, :] ** moment_index[2]
                )
    elif num_coords == 2:
        return np.sum(
                weights[:]
                * abscissas[0, :] ** moment_index[0]
                * abscissas[1, :] ** moment_index[1]
                )
    elif num_coords == 1:
        return np.sum(
                weights[:]
                * abscissas[0, :] ** moment_index[0]
                )

@njit
def quadrature_limited(weights, abscissas, moment_index, num_coords, num_nodes):
    flux_min = np.zeros(num_nodes)
    flux_max = np.zeros(num_nodes)
    for n in range(num_nodes):
        q = weights[n] * np.prod(moment_index[:,n]**moment_index[:])
        flux_min[n] = q * min(abscissas[0, n], 0.)
        flux_max[n] = q * max(abscissas[0, n], 0.)
    return np.sum(flux_min), np.sum(flux_max)

@njit
def flux_quadrature(weights, abscissas, indices, num_moments, num_nodes, num_points):
    flux_min = np.zeros((num_points,num_moments))
    flux_max = np.zeros((num_points,num_moments))
    num_coords = len(indices[0])
    for i in range(num_points):
        for m in range(num_moments):
            flux_min[i,m], flux_max[i,m] = quadrature_limited(
                weights[i],abscissas[i],indices[m],num_coords,num_nodes)
    return flux_min, flux_max

@njit
def domain_get_fluxes(weights, abscissas, indices, num_points, num_moments, num_nodes, flux):

    print("entering domain_get_flux")
    f_min, f_max = flux_quadrature(
                    weights, abscissas, 
                    indices, num_moments, 
                    num_nodes, num_points)

    f_sum = np.zeros_like(flux)
    f_sum[1:-1] = f_max[0:-2] + f_min[1:-1]
    flux[1:-2] = f_sum[1:-2] - f_sum[2:-1]


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
