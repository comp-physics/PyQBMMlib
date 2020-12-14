import numpy as np


def local_flux(weight, abscissa, moment):

    flux = 1.0
    flux *= abscissa[0] ** moment[0]
    flux *= abscissa[1] ** moment[1]
    flux *= abscissa[2] ** moment[2]
    # for i_coord in range( 3, num_coords ): flux *= abscissa[i_cord] ** moment[i_coord]
    flux *= weight
    return flux


def moment_fluxes(indices, wts_left, wts_right, xi_left, xi_right):
    """
    Computes moment fluxes

    inputs:
    -------
    num_nodes:  number of quadrature nodes, depends on inversion algorithm
    indices:    moment indices, size [ num_moments, num_internal_coords ]
    wts_left:   weights on the left side,  size [ num_nodes ]
    wts_right:  weights on the right side, size [ num_nodes ]
    xi_left:    abscissas on the left side,  size [ num_internal_coords, num_nodes ]
    xi_right:   abscissas on the right side, size [ num_internal_corods, num_nodes ]
    """

    num_moments = len(indices)
    num_coords, num_nodes = xi_left.shape

    flux = np.zeros(num_moments)

    for i_moment in range(num_moments):
        for i_node in range(num_nodes):

            # compute local fluxes
            flux_left = local_flux(
                wts_left[i_node], xi_left[:, i_node], indices[i_moment, :]
            )
            flux_right = local_flux(
                wts_right[i_node], xi_right[:, i_node], indices[i_moment, :]
            )

            # limiter (?)
            flux_left = flux_left * max(xi_left[0, i_node], 0.0)
            flux_right = flux_right * min(xi_right[0, i_node], 0.0)

            # quadrature
            flux[i_moment] += flux_left + flux_right

    return flux
