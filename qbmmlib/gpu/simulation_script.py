
from os import stat
import sys
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from nquad_cuda import QUAD

sys.path.append('../utils/')
from jets_util import jet_initialize_moments

indices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
    ]
)

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
    return moments

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

if __name__ == "__main__":

    # initialize jet 
    num_coords = 3
    num_nodes = 27
    num_points = 10
    num_moments = 16

    

    # states 
    ## Note: It is important to keep the parallelizable index (largest)
    ## on the most inner dimension
    state = np.zeros([num_moments, num_points])
    rhs = np.zeros([num_moments, num_points])

    # Initialize weights and abscissas
    weights = np.zeros([num_nodes, num_points])
    abscissas = np.zeros([num_nodes, num_coords, num_points])

    wts_left, wts_right, xi_left, xi_right = jet_initialize_moments(num_coords, num_nodes)

    disc_loc = 0.125
    n_pt = num_points - 2
    disc_idx = int(n_pt * disc_loc) - 2
    print('Dislocation index is ', disc_idx, ' out of ', n_pt, ' points')
    print("abscissas left: ", xi_left[0,:])
    print("abscissas right: ", xi_right[0,:])

    # Populate weights
    weights[:, :disc_idx] = np.asarray([wts_left]).T
    weights[:, -disc_idx:] = np.asarray([wts_right]).T
    # Populate abscissas
    abscissas[:, :, :disc_idx] = np.asarray([xi_left]).T
    abscissas[:, :, -disc_idx:] = np.asarray([xi_right]).T 

    # Populate state
    moments_left = projection(wts_left, xi_left, indices,
            num_coords, num_nodes)
    moments_right = projection(wts_right, xi_right, indices,
            num_coords, num_nodes)

    state[:, :disc_idx] = np.asarray([moments_left]).T
    state[:, -disc_idx:] = np.asarray([moments_right]).T

    state[:, 0] = np.asarray([moments_right])
    state[:, -1] = np.asarray([moments_left])

    #compute_rhs 
    grid_inversion(state)
    domain_get_fluxes(weights, abscissas, qbmm_mgr.indices,
                    num_points, qbmm_mgr.num_moments,
                    qbmm_mgr.num_nodes, flux)
    rhs = flux / grid_spacing

