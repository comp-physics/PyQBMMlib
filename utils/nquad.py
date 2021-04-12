import numpy as np
from inversion import hyqmom3, chyqmom9, chyqmom27
from numba import njit

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
        if num_coords == 1:
            moments[i] = quadrature_1d(
                weights, abscissas, indices[i], num_nodes
            )
    return moments


@njit
def quadrature_1d(weights, abscissas, moment_index, num_quadrature_nodes):
    # xi_to_idx = abscissas ** moment_index
    # q = np.dot(weights, xi_to_idx)
    # return q

    q = 0.0
    for i in range(num_quadrature_nodes):
        q += (
            weights[i]
            * (abscissas[0][i] ** moment_index[0])
        )
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
def flux_quadrature(wts, xi, indices, num_moments, num_nodes, num_points):
    flux_min = np.zeros((num_points,num_moments,num_nodes))
    flux_max = np.zeros((num_points,num_moments,num_nodes))
    for i in range(num_points):
        for m in range(num_moments):
            for n in range(num_nodes):
                flux = (
                    wts[i,n]
                    * xi[i, 0, n]**indices[m, 0]
                )

                if len(indices[0]) == 2:
                    flux *= (
                        xi[i, 1, n]**indices[m, 1]
                        )

                if len(indices[0]) == 3:
                    flux *= (
                        xi[i, 1, n]**indices[m, 1] *
                        xi[i, 2, n]**indices[m, 2]
                        )

                flux_min[i,m,n] = flux * min(xi[i, 0, n], 0.)
                flux_max[i,m,n] = flux * max(xi[i, 0, n], 0.)
        
    return flux_min, flux_max


@njit
def domain_get_fluxes(weights, abscissas, indices, num_points, num_moments, num_nodes, flux):

    f_min, f_max = flux_quadrature(
                    weights, abscissas, 
                    indices, num_moments, 
                    num_nodes, num_points)

    f_sum = np.zeros_like(flux)
    f_sum[1:-1,:] = np.sum(f_max[0:-2,:] + f_min[1:-1,:],axis=2)
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

# @njit
def domain_invert_2d(state, indices, weights, abscissas, num_points, num_coords, num_nodes):
    # Input: State, indices, num_points, num_coords, num_nodes
    # Output: Weights, abscissas

    for i_point in range(1, num_points-1):
        # Invert
        xi, wts = chyqmom9(state[i_point], indices)
        abscissas[i_point] = xi.T
        weights[i_point] = wts

    # Boundary conditions
    xi, wts = chyqmom9(state[0], indices)
    abscissas[0] = xi.T
    weights[0] = wts

    xi, wts = chyqmom9(state[-1], indices)
    abscissas[-1] = xi.T
    weights[-1] = wts

    return weights, abscissas

# @njit
def domain_invert_1d(state, indices, weights, abscissas, num_points, num_coords, num_nodes):
    # Input: State, indices, num_points, num_coords, num_nodes
    # Output: Weights, abscissas

    for i_point in range(1, num_points-1):
        # Invert
        xi, wts = hyqmom3(state[i_point])
        abscissas[i_point] = xi.T
        weights[i_point] = wts

    # Boundary conditions
    xi, wts = hyqmom3(state[0])
    abscissas[0] = xi.T
    weights[0] = wts

    xi, wts = hyqmom3(state[-1])
    abscissas[-1] = xi.T
    weights[-1] = wts

    return weights, abscissas

# @njit
# def domain_invert_2d(state, indices, weights, abscissas, num_points, num_coords, num_nodes):

#     for i_point in range(1, num_points-1):
#         # print(' ipt', i_point)
#         # print('State before', state[i_point])
#         # Invert
#         xi, wts = chyqmom9(state[i_point], indices)
#         abscissas[i_point] = xi.T
#         weights[i_point] = wts
#         # Project
#         state[i_point] = projection(wts, xi.T, indices, num_coords, num_nodes)
#         # print('State after', state[i_point])
#         if np.isnan(abscissas[i_point]).any() or \
#             np.isnan(weights[i_point]).any():
#             ii = np.where(np.isnan(abscissas[i_point]))[0]
#             # print('state',  iii, 'is nan:', state[iii])
#             print('abs:', abscissas[i_point])
#             print('w:',weights[i_point])
#             print('found nan in wts or absc')
#             raise Exception()

#     # Boundary conditions
#     state[0] = projection(weights[-2], abscissas[-2], indices, num_coords, num_nodes)
#     state[-1] = projection(weights[1], abscissas[1], indices, num_coords, num_nodes)

#     # print('get pt0')
#     # print(state[0])
#     xi, wts = chyqmom9(state[0], indices)
#     abscissas[0] = xi.T
#     weights[0] = wts
#     # print('get pt-1')
#     # print(state[-1])
#     xi, wts = chyqmom9(state[-1], indices)
#     abscissas[-1] = xi.T
#     weights[-1] = wts
#     # print(' finish update quad 2d')

#     print('max = ', np.max(state[:,:]))
#     if np.isnan(state).any():
#         print('found nan')
#         ii = np.where(np.isnan(state))[0]
#         iii = ii[0]
#         print('state',  iii, 'is nan:', state[iii])
#         raise Exception()
