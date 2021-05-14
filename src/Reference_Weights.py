#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 14:39:33 2021

@author: alexis
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

import warnings
warnings.filterwarnings('error')

import scipy.io as sio
from scipy import interpolate


def monte_carlo():
    config = {}
    config["advancer"] = {}
    config["mc"] = {}
    config["model"] = {}
    config["pop"] = {}
    config["wave"] = {}
    config["qbmm"] = {}

    config["advancer"]["method"] = "Euler"
    #config["advancer"]["method"] = "RK3"
    config["advancer"]["time_step"] = 1.0e-6
    config["advancer"]["final_time"] = 50.0
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
    #config["wave"]["amplitude"] = 1/0.90
    #config["wave"]["amplitude"] = 0.85
    #config["advancer"]["output_id"] = "Constant_Pressure"+str(int(100.0/config["wave"]["amplitude"]))
    #config["wave"]["form"] = "sine"
    #config["wave"]["form"] = "constant"
    #config["wave"]["period"] = 4.0
    # config["wave"]["cycles"] = 2.0
    #config["advancer"]["output_id"] = "Sinusoidal_Pressure"+str(int(100.0*config["wave"]["amplitude"]))+"_Period"+str(int(config["wave"]["period"]))
    
    #config["wave"]["form"] = "constant"
    #config["wave"]["amplitude"] = 1/0.20
    #config["advancer"]["output_dir"] = "../data/Constant_Forcing/"
    #config["advancer"]["output_id"] = "Constant_Pressure"+str(int(100.0/config["wave"]["amplitude"]))
    
    config["wave"]["form"] = "random"
    config["wave"]["cycles"] = 100.0
    config["wave"]["period"] = [5,7,9]
    #config["wave"]["amplitude"] = np.random.uniform(0.15,0.20,np.size(config["wave"]["period"]))
    config["wave"]["amplitude"] = [0.1723, 0.1768, 0.1973]
    #config["wave"]["phase"]  = np.random.uniform(0.0,2.0*np.pi,np.size(config["wave"]["period"]))
    config["wave"]["phase"]  = [1.3987, 1.3668, 3.5368]
    config["advancer"]["output_dir"] = "../data/Random_Forcing/"
    config["advancer"]["output_id"] = "Random_Pressure_Realization"+str(int(32))
    
    

    config["mc"]["Nsamples"] = 1000
    config["mc"]["Ntimes"] = 5001

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
    config["qbmm"]["num_quadrature_nodes"] = 9
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
    
    
    #folder_name = '../data/Constant_Forcing/';
    #mc_file_format = 'MC_HM_Constant_Pressure';
    #qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
    #MC_file_names = []
    #MC_file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
    
    folder_name = "../data/Random_Forcing/"
    mc_file_format = 'MC_HM_Random_Pressure_Realization';
    qbmm_HM_file_format = 'QBMM_HM_Random_Pressure_Realization';
    MC_file_names = []
    kstart = 32
    kend   = 32
    MC_file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(kstart,kend+1,1)]
    
    if (config["qbmm"]["num_quadrature_nodes"] == 4):
        total_cases = len(MC_file_names)
        for kk in range(0,total_cases):
            mat_data = sio.loadmat(MC_file_names[kk])
            moments  = mat_data['moments']
            total_times = moments[0,:].size
            abscissas_hist = np.zeros((total_times,2,4))
            weights_hist   = np.zeros((total_times,4))
        
            for tt in range(0,total_times):
                advancer.stage_state[0][0:6] = moments[0:6,tt]
                abscissas, weights = advancer.qbmm_mgr.moment_invert(advancer.stage_state[0], advancer.stage_k[0])
                abscissas_hist[tt,0,0:4] = abscissas[0][0:4]
                abscissas_hist[tt,1,0:4] = abscissas[1][0:4]
                weights_hist[tt,0:4] = weights[0:4]
            #sio.savemat(folder_name+"MC_Weights"+str(ii_flag)+".mat" ,{"abscissas_hist":abscissas_hist,"weights_hist":weights_hist})
            sio.savemat(folder_name+"MC_Weights_4points_Realization"+str(kstart+kk)+".mat" ,{"abscissas_hist":abscissas_hist,"weights_hist":weights_hist})
        
    elif (config["qbmm"]["num_quadrature_nodes"] == 9):
        total_cases = len(MC_file_names)
        for kk in range(0,total_cases):
            mat_data = sio.loadmat(MC_file_names[kk])
            moments  = mat_data['moments']
            total_times = moments[0,:].size
            abscissas_hist = np.zeros((total_times,2,9))
            weights_hist   = np.zeros((total_times,9))
        
            for tt in range(0,total_times):
                advancer.stage_state[0][0:6] = moments[0:6,tt]
                advancer.stage_state[0][6] = moments[6,tt]
                advancer.stage_state[0][7] = moments[9,tt]
                advancer.stage_state[0][8] = moments[10,tt]
                advancer.stage_state[0][9] = moments[14,tt]
                abscissas, weights = advancer.qbmm_mgr.moment_invert(advancer.stage_state[0], advancer.stage_k[0])
                abscissas_hist[tt,0,0:9] = abscissas[0][0:9]
                abscissas_hist[tt,1,0:9] = abscissas[1][0:9]
                weights_hist[tt,0:9] = weights[0:9]
            #sio.savemat(folder_name+"MC_Weights"+str(ii_flag)+".mat" ,{"abscissas_hist":abscissas_hist,"weights_hist":weights_hist})
            sio.savemat(folder_name+"MC_Weights_9points_Realization"+str(kstart+kk)+".mat" ,{"abscissas_hist":abscissas_hist,"weights_hist":weights_hist})
        
    #abscissas, weights = self.moment_invert(moments, self.indices)
    
    #advancer.stage_state[0][0] = 1.0
    #print(advancer.stage_state[0])
    #print(advancer.stage_k[0])
    #print(abscissas[0][0:4])
    

    #advancer.run()


    # Monte Carlo
    #mymc = mc(config)
    #mymc.run()

    # plt.tight_layout()
    #plt.show()


    # config["pop"]["moments"] = [[1, 0], [0, 1], [1, 1]]
    # config["pop"]["moments"] = advancer.qbmm_mgr.indices[0:3]
    # config["pop"]["moments"] = [ [3, 2], [2, 1], [3, 0], [ 3*(1-1.4), 0, 3*1.4 ] ]
    
    return abscissas_hist, weights_hist;



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
        abscissas, weights = monte_carlo()
        # print("devel_driver: no config file supplied")

    #exit()

