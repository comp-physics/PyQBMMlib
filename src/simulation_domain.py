import sys
sys.path.append('../utils/')

from qbmm_manager import *
from stats_util import *
from jets_util import *
from itertools import product

import numba
from nquad import *

# try:
#     import numba
#     from nquad import *
# except:
#     print("simulation_domain: Did not find numba! Install it for significant speedups.")
#     from quad import *

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
        # print('shape num coords :',self.qbmm_mgr.num_coords)
        wts_left, wts_right, xi_left, xi_right = jet_initialize_moments(
                self.qbmm_mgr.num_coords,
                self.qbmm_mgr.num_nodes)

        disc_loc = 0.125
        n_pt = len(self.weights) - 2
        disc_idx = int(n_pt * disc_loc) - 2

        # Populate weights
        self.weights[:disc_idx] = wts_left
        self.weights[-disc_idx:] = wts_right
        # Populate abscissas
        self.abscissas[:disc_idx] = xi_left
        self.abscissas[-disc_idx:] = xi_right 

        # Populate state
        moments_left = projection(wts_left, xi_left, self.qbmm_mgr.indices,
                self.qbmm_mgr.num_coords, self.qbmm_mgr.num_nodes)
        moments_right = projection(wts_right, xi_right, self.qbmm_mgr.indices,
                self.qbmm_mgr.num_coords, self.qbmm_mgr.num_nodes)

        state[:disc_idx] = moments_left
        state[-disc_idx:] = moments_right
        state[0] = moments_right
        state[-1] = moments_left

        # print("Domain: state: ", state[0,:])
        # print("Domain: weights: ", self.weights[0,:])
        # print("Domain: abscissas: ", self.abscissas[0,0,:])
        return


    def max_abscissa(self):
        """
        Return the maximum value of the x-abscissa in the simulation domain
        """
        return max(0.01,abs(self.abscissas[:,0,:]).max())

    def compute_rhs(self, state):
        """
        Compute moment transport fluxes, source terms.

        :param state: domain moments
        :type state: array like
        """
        if self.flow:
            self.grid_inversion(state)
            domain_get_fluxes(self.weights, self.abscissas, self.qbmm_mgr.indices,
                           self.num_points, self.qbmm_mgr.num_moments,
                           self.qbmm_mgr.num_nodes, self.flux)
            self.rhs = self.flux / self.grid_spacing
        elif self.qbmm_mgr.internal_dynamics:
            internal_rhs = np.zeros([self.num_points, self.qbmm_mgr.num_moments])
            self.qbmm_mgr.compute_rhs(state, internal_rhs)
            self.rhs = internal_rhs
        else:
            return self.zeros(self.rhs.shape)

        return self.rhs


    def grid_inversion(self,state):
        """
        Invert moments to weights/abscissas over whole grid

        :param state: domain moments
        :type state: array like
        """
        if self.qbmm_mgr.num_coords == 3:
            domain_invert_3d(state, self.qbmm_mgr.indices, 
                            self.weights, self.abscissas,
                            self.num_points, self.qbmm_mgr.num_coords,
                            self.qbmm_mgr.num_nodes)        
        elif self.qbmm_mgr.num_coords == 2:
            domain_invert_2d(state, self.qbmm_mgr.indices, 
                            self.weights, self.abscissas,
                            self.num_points, self.qbmm_mgr.num_coords,
                            self.qbmm_mgr.num_nodes)
        else:
            raise NotImplementedError

    def project(self, state):
        # if self.qbmm_mgr.num_coords == 3:
        domain_project(state, self.qbmm_mgr.indices, 
                    self.weights, self.abscissas,
                    self.num_points, self.qbmm_mgr.num_coords,
                    self.qbmm_mgr.num_nodes)        
        # else:
        #     raise NotImplementedError('No projection for 2D yet')

