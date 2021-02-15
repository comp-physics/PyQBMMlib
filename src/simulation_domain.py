import sys
sys.path.append('../utils/')

from qbmm_manager import *
from stats_util import *
from jets_util import *
from itertools import product

try:
    import numba
    from nquad import *
except:
    print("Did not find numba! Install it for significant speedups.")
    from quad import *

class simulation_domain():
    """
    This class handles grid and state for flow problems
    """
    def __init__(self, config):
        """
        Constructor
        
        :param config: Configuration
        :type config: dict
        """
        # Create a qbmm manager
        self.qbmm_mgr = qbmm_manager(config)

        # Get config
        self.num_dim = config["domain"]["num_dim"]
        self.num_points = config["domain"]["num_points"]
        self.extents = config["domain"]["grid_extents"]

        self.grid_spacing = (self.extents[1] - self.extents[0]) / (self.num_points - 2)

        if "flow" in config["domain"]:
            self.flow = config["domain"]["flow"]
            self.flux = np.zeros([self.num_points, self.qbmm_mgr.num_moments])
        else:
            self.flow = False
        
        print("domain: init: Configuration options ready:")
        print("\t num_dim      = %i" % self.num_dim)
        print("\t num_points   = %i" % self.num_points)
        print("\t extents      = [%.4E, %.4E]" % (self.extents[0], self.extents[1]))
        print("\t grid spacing = %.4E" % self.grid_spacing)
        
        # Initialize grid state & rhs
        self.state = np.zeros([self.num_points, self.qbmm_mgr.num_moments])
        self.rhs = np.zeros([self.num_points, self.qbmm_mgr.num_moments])

        # Initialize weights and abscissas
        self.weights = np.zeros([self.num_points, self.qbmm_mgr.num_nodes])
        self.abscissas = np.zeros([self.num_points, self.qbmm_mgr.num_coords, self.qbmm_mgr.num_nodes])
        
        return

    
    def initialize_state_uniform(self, mu, sigma):
        """
        Initialize grid state
        """

        raw_moments = raw_gaussian_moments_univar(self.qbmm_mgr.num_moments, mu, sigma)
        for i_point in range(self.num_points):
            self.state[i_point] = raw_moments

        print(self.state)
        return

    
    def initialize_state_jets(self, state):
        """
        Initialize grid state to 1D crossing-jets

        :param state: domain moments
        :type state: array like
        """
        wts_left, wts_right, xi_left, xi_right = jet_initialize_moments(self.qbmm_mgr.num_nodes)
        # Populate weights
        self.weights[:48] = wts_left
        self.weights[-48:] = wts_right
        # Populate abscissas
        self.abscissas[:48] = xi_left
        self.abscissas[-48:] = xi_right 
        # Populate state
        moments_left = self.qbmm_mgr.projection(wts_left, xi_left, self.qbmm_mgr.indices)
        moments_right = self.qbmm_mgr.projection(wts_right, xi_right, self.qbmm_mgr.indices)
        state[:48] = moments_left
        state[-48:] = moments_right
        state[0] = moments_right
        state[-1] = moments_left

        print("Domain: state: ", state[0,:])
        print("Domain: weights: ", self.weights[0,:])
        print("Domain: abscissas: ", self.abscissas[0,0,:])
        return


    def max_abscissa(self):
        """
        Return the maximum value of the x-abscissa in the simulation domain
        """
        return max(0.01,abs(self.abscissas[:,0,:]).max())


    def update_quadrature(self, state, project = True):
        """
        This function updates the quadrature weights and abscissas for a given state.

        :param state: domain moments
        :type state: array like
        """    
        # Loop over interior points
        for i_point in range(1, self.num_points-1):
            # Invert
            xi, wts = self.qbmm_mgr.moment_invert(state[i_point], self.qbmm_mgr.indices)
            self.abscissas[i_point] = xi.T
            self.weights[i_point] = wts
            # Project
            state[i_point] = self.qbmm_mgr.projection(wts, xi.T, self.qbmm_mgr.indices)

        # Boundary conditions
        state[0] = self.qbmm_mgr.projection(self.weights[-2], self.abscissas[-2],
                                            self.qbmm_mgr.indices)
        state[-1] = self.qbmm_mgr.projection(self.weights[1], self.abscissas[1],
                                             self.qbmm_mgr.indices)
        xi, wts = self.qbmm_mgr.moment_invert(state[0], self.qbmm_mgr.indices)
        self.abscissas[0] = xi.T
        self.weights[0] = wts
        xi, wts = self.qbmm_mgr.moment_invert(state[-1], self.qbmm_mgr.indices)
        self.abscissas[-1] = xi.T
        self.weights[-1] = wts
        return


    def local_flux(self, weight, abscissa, index):
        """
        Compute local moment flux for given quadrature weight and abscissa

        :param weight: quadrature weight
        :param abscissa: quadrature abscissa
        :param index: moment index
        :type weight: float
        :type abscissa: array like
        :type index: array like
        """
        return weight*(abscissa[0]**index[0])*(abscissa[1]**index[1])*(abscissa[2]**index[2])


    def moment_fluxes(self, indices, wts_left, wts_right, xi_left, xi_right):
        """
        Computes moment fluxes
        :param indices: domain moment indices
        :param wts_left: quadrature weights on the left face of a grid cell
        :param wts_right: quadrature weights on the right face of a grid cell
        :param xi_left: quadrature abscissas on the left face of a grid cell
        :param xi_right: quadrature abscissas on the right face of a grid cell
        :type indices: array like
        :type wts_left: array like
        :type wts_right: array like
        :type xi_left: array like
        :type xi_right: array like
        """
        return flux_quadrature(wts_left, xi_left, wts_right, xi_right, indices,
                               self.qbmm_mgr.num_moments, self.qbmm_mgr.num_nodes)

        # flux = np.zeros(self.qbmm_mgr.num_moments)        
        # for m, n in product(range(self.qbmm_mgr.num_moments), range(self.qbmm_mgr.num_nodes)):
        #     # compute local fluxes
        #     flux_left = self.local_flux(wts_left[n], xi_left[:, n], indices[m, :])
        #     flux_right = self.local_flux(wts_right[n], xi_right[:, n], indices[m, :])
        #     # limiter
        #     flux_left = flux_left * max(xi_left[0, n], 0)
        #     flux_right = flux_right * min(xi_right[0, n], 0)
            
        #     # quadrature
        #     flux[m] += flux_left + flux_right

        # return flux      
    


    def compute_fluxes(self, state):
        """
        Compute moment fluxes

        :param state: domain moments
        :type state: array like
        """
        for i_point in range(1, self.num_points-1):

            # Compute left flux
            wts_left = self.weights[i_point-1]
            wts_right = self.weights[i_point]
            xi_left = self.abscissas[i_point-1]
            xi_right = self.abscissas[i_point]
            f_left = self.moment_fluxes(self.qbmm_mgr.indices, wts_left, wts_right,
                                        xi_left, xi_right)
            
            # Compute right flux
            wts_left = wts_right
            xi_left = xi_right
            wts_right = self.weights[i_point+1]
            xi_right = self.abscissas[i_point+1]
            f_right = self.moment_fluxes(self.qbmm_mgr.indices, wts_left, wts_right,
                                         xi_left, xi_right)

            # Reconstruct flux
            self.flux[i_point] = f_left - f_right
            
            
        return

    
    def compute_rhs(self, state):
        """
        Compute moment transport fluxes, source terms.

        :param state: domain moments
        :type state: array like
        """
        if self.flow:
            self.compute_fluxes(state)
            self.rhs = self.flux / self.grid_spacing
        elif self.qbmm_mgr.internal_dynamics:
            internal_rhs = np.zeros([self.num_points, self.qbmm_mgr.num_moments])
            self.qbmm_mgr.compute_rhs(state, internal_rhs)
            self.rhs = internal_rhs
        else:
            return self.zeros(self.rhs.shape)

        return self.rhs


    def project(self, state):
        """
        Project moments

        :param state: domain moments
        :type state: array like
        """
        self.update_quadrature(state)        
        
        
