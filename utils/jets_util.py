import sys

sys.path.append("../src/")
from qbmm_manager import *
import numpy as np


def jet_initialize_moments(qbmm_mgr):

    num_nodes = qbmm_mgr.num_quadrature_nodes

    wts_left = np.zeros(num_nodes)
    wts_right = np.zeros(num_nodes)
    xi_left = np.zeros([3, num_nodes])
    xi_right = np.zeros([3, num_nodes])

    num_left = 1.0
    num_right = 0.5

    u_left = 0.01
    u_right = 1.0

    wts_left[0] = 0.125 * num_left
    wts_left[2] = 0.125 * num_left
    wts_left[6] = 0.125 * num_left
    wts_left[8] = 0.125 * num_left
    wts_left[18] = 0.125 * num_left
    wts_left[20] = 0.125 * num_left
    wts_left[24] = 0.125 * num_left
    wts_left[26] = 0.125 * num_left

    wts_right[0] = 0.125 * num_right
    wts_right[2] = 0.125 * num_right
    wts_right[6] = 0.125 * num_right
    wts_right[8] = 0.125 * num_right
    wts_right[18] = 0.125 * num_right
    wts_right[20] = 0.125 * num_right
    wts_right[24] = 0.125 * num_right
    wts_right[26] = 0.125 * num_right

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
    xi_left[1, 6] = -1.0
    xi_left[1, 8] = -1.0
    xi_left[1, 18] = 1.0
    xi_left[1, 20] = 1.0
    xi_left[1, 24] = -1.0
    xi_left[1, 26] = -1.0

    xi_right[1, 0] = 1.0
    xi_right[1, 2] = 1.0
    xi_right[1, 6] = -1.0
    xi_right[1, 8] = -1.0
    xi_right[1, 18] = 1.0
    xi_right[1, 20] = 1.0
    xi_right[1, 24] = -1.0
    xi_right[1, 26] = -1.0

    xi_left[2, 0] = -1.0
    xi_left[2, 2] = 1.0
    xi_left[2, 6] = -1.0
    xi_left[2, 8] = 1.0
    xi_left[2, 18] = -1.0
    xi_left[2, 20] = 1.0
    xi_left[2, 24] = -1.0
    xi_left[2, 26] = 1.0
    xi_right[2, :] = xi_left[2, :]

    xi_left[1, :] += 0.5 * xi_left[0, :]
    xi_right[1, :] += 0.5 * xi_right[0, :]

    xi_left[2, :] = 0.0
    xi_right[2, :] = 0.0

    print(qbmm_mgr.indices)

    moments_left = qbmm_mgr.projection(wts_left, xi_left, qbmm_mgr.indices)
    moments_right = qbmm_mgr.projection(wts_right, xi_right, qbmm_mgr.indices)

    return moments_left, moments_right
