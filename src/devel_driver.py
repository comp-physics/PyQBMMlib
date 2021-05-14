"""

.. module:: devel_driver
   :platform: MacOS, Unix
   :synopsis: A useful module indeed.

.. moduleauthor:: SHB <spencer@caltech.edu> and Esteban Cisneros

These are example drivers that you may use as templates for your application.

"""

from advancer import *
from config_manager import *
import sys

sys.path.append("../utils/")
from stats_util import *
from euler_util import *
from jets_util import *
from pretty_print_util import *
import cProfile
from mc import mc
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('error')


def monte_carlo():
    config = {}
    config["advancer"] = {}
    config["mc"] = {}
    config["model"] = {}
    config["pop"] = {}
    config["wave"] = {}
    config["qbmm"] = {}

    # config["advancer"]["method"] = "Euler"
    config["advancer"]["method"] = "RK3"
    config["advancer"]["time_step"] = 1.0e-6
    config["advancer"]["final_time"] = 200.0
    config["advancer"]["error_tol"] = 1.0e-7
    config["advancer"]["num_steps"] = 200000
    config["advancer"]["num_steps_print"] = 100
    config["advancer"]["num_steps_write"] = 100
    #config["advancer"]["output_dir"] = "../data/Constant_Forcing/"
    #config["advancer"]["output_dir"] = "../data/Sinusoidal_Forcing/"
    #config["advancer"]["output_id"] = "example_2D"
    config["advancer"]["write_to"] = "txt"

    # Acoustic
    # config["wave"]["amplitude"] = 3
    
    #config["wave"]["form"] = "constant"
    #config["wave"]["amplitude"] = 1/0.20
    #config["advancer"]["output_dir"] = "../data/Constant_Forcing/"
    #config["advancer"]["output_id"] = "Constant_Pressure"+str(int(100.0/config["wave"]["amplitude"]))
    
    #config["wave"]["form"] = "sine"
    #config["wave"]["amplitude"] = 0.85
    #config["wave"]["period"] = 3.0
    #config["advancer"]["output_dir"] = "../data/Sinusoidal_Forcing/"
    #config["advancer"]["output_id"] = "Sinusoidal_Pressure"+str(int(100.0*config["wave"]["amplitude"]))+"_Period"+str(int(config["wave"]["period"]))
    
    config["wave"]["form"] = "random"
    config["wave"]["cycles"] = 100.0
    config["wave"]["period"] = [5,7,9]
    config["wave"]["amplitude"] = np.random.uniform(0.15,0.20,np.size(config["wave"]["period"]))
    #config["wave"]["amplitude"] = [0.1723, 0.1768, 0.1973]
    config["wave"]["phase"]  = np.random.uniform(0.0,2.0*np.pi,np.size(config["wave"]["period"]))
    #config["wave"]["phase"]  = [1.3987, 1.3668, 3.5368]
    config["advancer"]["output_dir"] = "../data/Random_Forcing/"
    config["advancer"]["output_id"] = "Random_Pressure_Realization"+str(int(1))
    
    
    #config["wave"]["amplitude"] = 0.85
    #config["advancer"]["output_id"] = "Constant_Pressure"+str(int(100.0/config["wave"]["amplitude"]))
    #config["wave"]["form"] = "sine"
    #config["wave"]["form"] = "constant"
    #config["wave"]["period"] = 4.0
    # config["wave"]["cycles"] = 2.0
    #config["advancer"]["output_id"] = "Sinusoidal_Pressure"+str(int(100.0*config["wave"]["amplitude"]))+"_Period"+str(int(config["wave"]["period"]))

    config["mc"]["Nsamples"] = 1000
    config["mc"]["Ntimes"] = 2001

    # in R and Rdot directions
    config["pop"]["shape"] = ["normal", "normal"]
    config["pop"]["mu"] = [1.0+1.E-12,0.0+1.E-12]
    config["pop"]["sig"] = [0.05,0.05]
    # config["pop"]["moments"] = [[1, 0], [0, 1], [1, 1]]
    #config["pop"]["moments"] = [ [1,0], [0,1], [2,0], [1,1], [0,2], [3, 2], [2, 1], [3, 0], [ 3*(1-1.4), 0 ] ]
    config["pop"]["moments"] = [ [0,0], 
                                 [1,0], [0,1], 
                                 [2,0], [1,1], [0,2], 
                                 [3,0], [2,1], [1,2], [0,3], 
                                 [4,0], [3,1], [2,2], [1,3], [0,4], 
                                 [5,0], [4,1], [3,2], [2,3], [1,4], [0,5], 
                                 [3*(1-1.4), 0], 
                                 [-1,2], [-2,1], [-4,0], [-1,0], [-1,1], [-3,0], [-1,3], [-2,2], [-4,1] ]

    # Bubble properties
    config["model"]["model"] = "RPE"
    # config["model"]["model"] = "KM"
    # config["model"]["model"] = "Linear"
    # config["model"]["R"] = 1.0
    # config["model"]["V"] = 0.0
    config["model"]["gamma"] = 1.0
    # config["model"]["c"] = 1000.
    # config["model"]["Ca"] = 0.5
    config["model"]["Re_inv"] = 1/1000.
    # config["model"]["Web"] = 13.9

    
    config["qbmm"]["governing_dynamics"] = " - 1.5*xdot*xdot/x + 1./(x**4) - 1./(x*p)  - 4/1000*xdot/x/x"
    # config["qbmm"]["governing_dynamics"] = " - 1.5*xdot*xdot/x + 1./(x**4) - 1./(x*0.9) "
    # == Rddot

    # config["qbmm"]["governing_dynamics"] = " - x - xdot"
    config["qbmm"]["num_internal_coords"] = 2
    config["qbmm"]["num_quadrature_nodes"] = 4
    # config["qbmm"]["num_quadrature_nodes"] = 9
    config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30

    # Initialize condition
    advancer = time_advancer(config)
    num_dim = config["qbmm"]["num_internal_coords"]
    mu = config["pop"]["mu"]
    sigma = config["pop"]["sig"]

    if num_dim == 1:
        advancer.initialize_state_gaussian_univar(mu, sigma)
    elif num_dim == 2:
        advancer.initialize_state_gaussian_bivar(mu[0], mu[1], sigma[0], sigma[1])

    advancer.initialize_wave(wave_config=config["wave"])

    advancer.run()


    # Monte Carlo
    mymc = mc(config)
    mymc.run()

    # plt.tight_layout()
    plt.show()


    # config["pop"]["moments"] = [[1, 0], [0, 1], [1, 1]]
    # config["pop"]["moments"] = advancer.qbmm_mgr.indices[0:3]
    # config["pop"]["moments"] = [ [3, 2], [2, 1], [3, 0], [ 3*(1-1.4), 0, 3*1.4 ] ]


def flow_example():
    """
    This driver solves a flow-coupled problem.
    Currently, it only computes moment fluxes, but work is underway to solve the compressible flow equations.
    """
    # In development
    config = {}
    config["qbmm"] = {}
    config["advancer"] = {}

    config["qbmm"]["flow"] = True
    config["qbmm"]["governing_dynamics"] = ""
    config["qbmm"]["num_internal_coords"] = 3
    config["qbmm"]["num_quadrature_nodes"] = 27
    config["qbmm"]["method"] = "chyqmom"
    config["qbmm"]["adaptive"] = False
    config["qbmm"]["max_skewness"] = 30

    qbmm_mgr = qbmm_manager(config)
    indices = qbmm_mgr.indices

    # Initial condition
    # mu1    = 1.0
    # mu2    = 1.0
    # mu3    = 1.0
    # sigma1 = 0.1
    # sigma2 = 0.1
    # sigma3 = 0.1
    # moments = raw_gaussian_moments_trivar( indices, mu1, mu2, mu3,
    #                                        sigma1, sigma2, sigma3 )

    moments_left, moments_right = jet_initialize_moments(qbmm_mgr)

    xi_left, wts_left = qbmm_mgr.moment_invert(moments_left, indices)
    xi_right, wts_right = qbmm_mgr.moment_invert(moments_right, indices)

    print(wts_left)
    print(wts_right)

    print(xi_left)
    print(xi_right)

    flux = moment_fluxes(indices, wts_left, wts_right, xi_left, xi_right)

    print(flux)

    return


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
    num_dim = config["qbmm"]["num_internal_coords"]
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

    return


def advance_example2dp1():
    # In development!
    config = {}
    config["qbmm"] = {}
    config["advancer"] = {}

    config["qbmm"]["flow"] = False
    config["qbmm"]["governing_dynamics"] = " - x - xdot - r0"
    config["qbmm"]["num_internal_coords"] = 2
    config["qbmm"]["num_quadrature_nodes"] = 4
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

    return


if __name__ == "__main__":

    np.set_printoptions(formatter={"float": "{: 0.4E}".format})

    nargs = len(sys.argv)
    if nargs == 2:
        config_file = sys.argv[1]
        config_mgr = config_manager(config_file)
        config = config_mgr.get_config()
        advance_example(config)
        ### Complete workflow for devel driver:
        ### 1. Check whether input file exists
        ### 2. If yes, run advance_example, then stop
        ### 3. If no, compare argv[1] to case name
        ### 4. If argv matches case, run, then stop
        ### 5. If argv does not match case, then exit
    else:
        monte_carlo()
        # print("devel_driver: no config file supplied")

    exit()
