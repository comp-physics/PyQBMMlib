import sys
sys.path.append('../utils/')

from qbmm_manager import *
from stats_util import *

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

        self.num_dim = config["domain"]["num_dim"]
        self.num_points = config["domain"]["num_points"]
        self.extents = config["domain"]["grid_extents"]

        self.grid_spacing = (self.extents[1] - self.extents[0]) / (self.num_points - 1)

        print('domain: init: Configuration options ready:')
        print('\t num_dim = %i' % self.num_dim)
        print('\t num_points = %i' % self.num_points)
        print('\t extents = [%.4E, %.4E]' % (self.extents[0], self.extents[1]))
        print('\t grid spacing = %.4E' % self.grid_spacing)

        # Create a qbmm manager
        self.qbmm_mgr = qbmm_manager(config)
        
        # Initialize grid state
        self.state = np.zeros([self.num_points,self.qbmm_mgr.num_moments])
        
        return

    def create_grid(self):
        """
        Creates grid
        """
        self.X = np.linspace(self.extents[0], self.extents[1], self.num_points)
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
        Initialize grid state to R.O. Fox's jet conditions
        """
        moments_left, moments_right = jet_initialize_momemts(self.qbmm_mgr)
        state[0:self.num_points/2] = moments_left
        state[1+self.num_points/2:self.num_points] = moments_right
        return
    
    def compute_fluxes(self, state):
        """
        Compute moment fluxes
        """
        for i_point in range(1,self.num_points-1):

            # Compute left flux
            xi_left, wts_left = qbmm_mgr.moment_invert(state[i_point-1], self.qbmm_mgr.indices)
            xi_right, wts_right = qbmm_mgr.moment_invert(state[i_point], self.qbmm_mgr.indices)
            f_left = moment_fluxes(self.qbmm_mgr.indices, wts_left, wts_right, xi_left, x_right)

            # Compute right flux
            xi_left = xi_right
            wts_left = wts_right
            xi_right, wts_right = qbmm_mgr.moment_invert(self.state[i_point+1], self.qbmm_mgr.indices)            
            f_right = moment_fluxes(self.qbmm_mgr.indices, wts_left, wts_right, xi_left)

            # Reconstruct flux
            flux = f_left - f_right
            
        return

    def compute_rhs(self, state):

        flux = self.compute_flux(state)
        rhs = flux / self.grid_spacing
        return rhs

