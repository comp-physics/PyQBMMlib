"""

.. module:: devel_driver
   :platform: MacOS, Unix
   :synopsis: A useful module indeed.

.. moduleauthor:: SHB <spencer@caltech.edu> and Esteban Cisneros

These are example drivers that you may use as templates for your application.

"""

from advancer import *
from config_manager import *
from simulation_domain import *
import sys

sys.path.append("../utils/")
from stats_util import *
from jets_util import *
from pretty_print_util import *
import cProfile

import warnings
warnings.filterwarnings('error')

def flow_example_3d():
    """
    This driver solves a flow-coupled problem.
    Currently, it only computes moment fluxes, but work is underway to solve the compressible flow equations.
    """
    # In development

    cfl = 0.4
    N = 50
    dx = 1./N
    U_max = 1
    
    config = {}
    config['qbmm'] = {}
    config["advancer"] = {}
    config["domain"] = {}

    config["qbmm"]["internal_dynamics"] = ""
    config["qbmm"]["num_coords"] = 3
    config["qbmm"]["num_nodes"] = 27
    config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30

    config["domain"]["flow"] = True
    config["domain"]["num_dim"] = 1
    config["domain"]["num_points"] = N + 2
    config["domain"]["grid_extents"] = [0, 1]
    
    # config["advancer"]["method"] = "Euler"
    config["advancer"]["method"] = "RK2"
    config["advancer"]["time_step"] = cfl * dx / U_max
    config["advancer"]["cfl"] = cfl
    # config["advancer"]["project"] = False
    config["advancer"]["final_time"] = 0.1
    config["advancer"]["num_steps"] = 5
    config["advancer"]["num_steps_print"] = 1 #1000
    config["advancer"]["num_steps_write"] = 5 #1000
    config["advancer"]["output_dir"] = "output/"
    config["advancer"]["output_id"] = "example_flow_compiled"
    config["advancer"]["write_to"] = "txt"

    advancer = time_advancer(config)
    advancer.initialize_state_jets()
    
    advancer.run()    

def flow_example_2d():
    """
    This driver solves a flow-coupled problem.
    Currently, it only computes moment fluxes, but work is underway to solve the compressible flow equations.
    """
    # In development

    cfl = 0.4
    N = 400
    dx = 1./N
    U_max = 1
    
    config = {}
    config['qbmm'] = {}
    config["advancer"] = {}
    config["domain"] = {}

    config["qbmm"]["internal_dynamics"] = ""
    config["qbmm"]["num_coords"] = 2
    config["qbmm"]["num_nodes"] = 9
    config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30

    config["domain"]["flow"] = True
    config["domain"]["num_dim"] = 1
    config["domain"]["num_points"] = N + 2
    config["domain"]["grid_extents"] = [0, 1]
    
    # config["advancer"]["method"] = "Euler"    
    config["advancer"]["method"] = "RK2"
    config["advancer"]["time_step"] = cfl * dx / U_max
    config["advancer"]["cfl"] = cfl
    # config["advancer"]["project"] = False
    config["advancer"]["final_time"] = 0.6
    config["advancer"]["num_steps"] = 10000
    config["advancer"]["num_steps_print"] = 1 #1000
    config["advancer"]["num_steps_write"] = 10 #1000
    config["advancer"]["output_dir"] = "output/"
    config["advancer"]["output_id"] = "example_flow_compiled"
    config["advancer"]["write_to"] = "txt"

    advancer = time_advancer(config)
    advancer.initialize_state_jets()
    advancer.run()    

def flow_example_1d():
    """
    This driver solves a flow-coupled problem.
    Currently, it only computes moment fluxes, but work is underway to solve the compressible flow equations.
    """
    # In development

    cfl = 0.4
    N = 40
    dx = 1./N
    U_max = 1
    
    config = {}
    config['qbmm'] = {}
    config["advancer"] = {}
    config["domain"] = {}

    config["qbmm"]["internal_dynamics"] = ""
    config["qbmm"]["num_coords"] = 1
    config["qbmm"]["num_nodes"] = 3
    # config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["method"] = "hyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30

    config["domain"]["flow"] = True
    config["domain"]["num_dim"] = 1
    config["domain"]["num_points"] = N + 2
    config["domain"]["grid_extents"] = [0, 1]
    
    config["advancer"]["method"] = "Euler"    
    # config["advancer"]["method"] = "RK2"
    config["advancer"]["time_step"] = cfl * dx / U_max
    config["advancer"]["cfl"] = cfl
    # config["advancer"]["project"] = False
    config["advancer"]["final_time"] = 0.1
    config["advancer"]["num_steps"] = 10000
    config["advancer"]["num_steps_print"] = 1 #1000
    config["advancer"]["num_steps_write"] = 10 #1000
    config["advancer"]["output_dir"] = "output/"
    config["advancer"]["output_id"] = "example_flow_compiled"
    config["advancer"]["write_to"] = "txt"

    advancer = time_advancer(config)
    advancer.initialize_state_jets()
    advancer.run()    
    

def advance_example(config):
    """
    This driver solves the moments transport equations.
    It is independent of governing dynamics, which are specified in the ``config`` dictionary.
    It is constrained to problems with 1 and 2 internal coordinates by the initial condition.
    The initial condition is Gaussian.

    :param config: Configuration dictionary
    :type config: dict
    """

    advancer = time_advancer(config)

    # Initialize condition
    num_dim = config["qbmm"]["num_coords"]
    mu = config["init_condition"]["mu"]
    sigma = config["init_condition"]["sigma"]

    if num_dim == 1:
        advancer.initialize_state_gaussian_univar(mu, sigma)
    elif num_dim == 2:
        mu1 = mu[0]
        mu2 = mu[1]
        sigma1 = sigma[0]
        sigma2 = sigma[1]
        advancer.initialize_state_gaussian_bivar(mu1, mu2, sigma1, sigma2)

    # Run
    advancer.run()


def advance_example2dp1():
    # In development!
    config = {}
    config["qbmm"] = {}
    config["advancer"] = {}

    config["qbmm"]["flow"] = False
    config["qbmm"]["internal_dynamics"] = " - x - xdot - r0"
    config["qbmm"]["num_coords"] = 2
    config["qbmm"]["num_nodes"] = 4
    config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30
    config["qbmm"]["polydisperse"] = True
    config["qbmm"]["num_poly_nodes"] = 5
    config["qbmm"]["poly_symbol"] = "r0"

    config["advancer"]["method"] = "RK23"
    config["advancer"]["time_step"] = 1.0e-5
    config["advancer"]["final_time"] = 30.0
    config["advancer"]["error_tol"] = 1.0e-5
    config["advancer"]["num_steps"] = 20000
    config["advancer"]["num_steps_print"] = 1000
    config["advancer"]["num_steps_write"] = 1000
    config["advancer"]["output_dir"] = "D/"
    config["advancer"]["output_id"] = "example_2D"
    config["advancer"]["write_to"] = "txt"

    advancer = time_advancer(config)

    # Initial condition
    mu1 = 1.0
    mu2 = 0.0
    mu3 = 0.1
    sigma1 = 0.1
    sigma2 = 0.2

    advancer.initialize_state_gaussian_bivar(mu1, mu2, sigma1, sigma2)

    advancer.run()


if __name__ == "__main__":

    # np.set_printoptions(formatter={"float": "{: 0.4E}".format})

    # nargs = len(sys.argv)
    # if nargs == 2:
    #     config_file = sys.argv[1]
    #     config_mgr = config_manager(config_file)
    #     config = config_mgr.get_config()domain_invert_3d
    #     advance_example(config)
        ### Complete workflow for devel drwrite_to_h5iver:
        ### 1. Check whether input file exists
        ### 2. If yes, run advance_example, then stop
        ### 3. If no, compare argv[1] to case name
        ### 4. If argv matches case, run, then stop
        ### 5. If argv does not match case, then exit
    # else:
    #     flow_example_3d()
        # flow_example_2d()
        # flow_example_1d()
        #print('devel_driver: no config file supplied')


    flow_example_3d()52