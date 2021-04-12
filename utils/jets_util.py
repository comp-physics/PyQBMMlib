import sys

sys.path.append("../src/")
from qbmm_manager import *
import numpy as np


def jet_initialize_moments(num_coords,num_nodes):
    print('Initializing jet with', num_coords, 'coordinates')
    if num_coords == 3:
        return init_3d_jet(num_nodes)
    elif num_coords == 2:
        return init_2d_jet(num_nodes)

def init_3d_jet(num_nodes):
    wts_left = np.zeros(num_nodes)
    wts_right = np.zeros(num_nodes)
    xi_left = np.zeros([3, num_nodes])
    xi_right = np.zeros([3, num_nodes])

    w_left = 1
    w_right = 0.5

    u_left = 0.01
    u_right = 1

    wts_left[0] = 0.125 * w_left
    wts_left[2] = 0.125 * w_left
    wts_left[6] = 0.125 * w_left
    wts_left[8] = 0.125 * w_left
    wts_left[18] = 0.125 * w_left
    wts_left[20] = 0.125 * w_left
    wts_left[24] = 0.125 * w_left
    wts_left[26] = 0.125 * w_left

    wts_right[0] = 0.125 * w_right
    wts_right[2] = 0.125 * w_right
    wts_right[6] = 0.125 * w_right
    wts_right[8] = 0.125 * w_right
    wts_right[18] = 0.125 * w_right
    wts_right[20] = 0.125 * w_right
    wts_right[24] = 0.125 * w_right
    wts_right[26] = 0.125 * w_right

    xi_left[0, 0] = u_left
    xi_left[0, 2] = u_left
    xi_left[0, 6] = u_left
    xi_left[0, 8] = u_left
    xi_left[0, 18] = u_left
    xi_left[0, 20] = u_left
    xi_left[0, 24] = u_left
    xi_left[0, 26] = u_left

    xi_right[0, 0] = u_right
    xi_right[0, 2] = u_right
    xi_right[0, 6] = u_right
    xi_right[0, 8] = u_right
    xi_right[0, 18] = u_right
    xi_right[0, 20] = u_right
    xi_right[0, 24] = u_right
    xi_right[0, 26] = u_right

    xi_left[1, 0] = u_left
    xi_left[1, 2] = u_left
    xi_left[1, 6] = -1
    xi_left[1, 8] = -1
    xi_left[1, 18] = 1
    xi_left[1, 20] = 1
    xi_left[1, 24] = -1
    xi_left[1, 26] = -1

    xi_right[1, 0] = 1
    xi_right[1, 2] = 1
    xi_right[1, 6] = -1
    xi_right[1, 8] = -1
    xi_right[1, 18] = 1
    xi_right[1, 20] = 1
    xi_right[1, 24] = -1
    xi_right[1, 26] = -1

    xi_left[2, 0] = -1
    xi_left[2, 2] = 1
    xi_left[2, 6] = -1
    xi_left[2, 8] = 1
    xi_left[2, 18] = -1
    xi_left[2, 20] = 1
    xi_left[2, 24] = -1
    xi_left[2, 26] = 1
    xi_right[2, :] = xi_left[2, :]

    xi_left[1, :] += 0.5 * xi_left[0, :]
    xi_right[1, :] += 0.5 * xi_right[0, :]

    xi_left[2, :] = 0.0
    xi_right[2, :] = 0.0

    return wts_left, wts_right, xi_left, xi_right


def init_2d_jet(num_nodes):
    wts_left = np.zeros(num_nodes)
    wts_right = np.zeros(num_nodes)
    xi_left = np.zeros([2, num_nodes])
    xi_right = np.zeros([2, num_nodes])

    w_left = 1
    w_right = 0.5

    u_left = 0.01
    u_right = 1

    wts_left[0] = w_left / 4.
    wts_left[2] = w_left / 4.
    wts_left[6] = w_left / 4.
    wts_left[8] = w_left / 4.

    wts_right[0] = w_right / 4.
    wts_right[2] = w_right / 4.
    wts_right[6] = w_right / 4.
    wts_right[8] = w_right / 4.

    xi_left[0, 0] = u_left
    xi_left[0, 2] = u_left
    xi_left[0, 6] = u_left
    xi_left[0, 8] = u_left

    xi_right[0, 0] = u_right
    xi_right[0, 2] = u_right
    xi_right[0, 6] = u_right
    xi_right[0, 8] = u_right

    # Taking from y-direction in 3D jet 
    xi_left[1, 0] = u_left
    xi_left[1, 2] = u_left
    # xi_left[1, 0] = 1
    # xi_left[1, 2] = 1
    xi_left[1, 6] = -1
    xi_left[1, 8] = -1

    # Try taking from z-direction instead
    # xi_left[1, 0] = -1
    # xi_left[1, 2] = 1
    # xi_left[1, 6] = -1
    # xi_left[1, 8] = 1

    xi_right[1, :] = xi_left[1, :]

    xi_left[1, :] += 0.5 * xi_left[0, :]
    xi_right[1, :] += 0.5 * xi_right[0, :]

    # xi_left[1, :] = 0.
    # xi_right[1, :] = 0.

    # xi_right[:, :] += np.random.rand()*1e-7
    # xi_left[:, :] += np.random.rand()*1e-7
    # wts_left[:] += np.random.rand()*1e-7
    # wts_right[ :] += np.random.rand()*1e-7

    return wts_left, wts_right, xi_left, xi_right
