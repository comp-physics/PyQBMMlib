#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:34:07 2021

@author: alexis
"""


print('Importing scipy... ', end='', flush=True)
import scipy.io as sio
from scipy import interpolate
print('Done!')
print('Importing numpy... ', end='', flush=True)
import numpy as np
from numpy import linspace
from numpy import fft
print('Done!')



###### 1. Load Matlab data from Gaussian closure of nonlinear RP - START. #####
def Load_Data_MC():
    
    print('Importing Monte-Carlo data... ', end='', flush=True)

    folder_name = '../data/Constant_Forcing/';
    mc_file_format = 'MC_HM_Constant_Pressure';
    
    file_names = []
    file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(25,100,5)]

    total_cases = len(file_names)
    
    for kk in range(0,total_cases):
        mat_data = sio.loadmat(file_names[kk])
        moments  = mat_data['moments']
        if (kk == 0):
            max_out = np.zeros((total_cases,4))
            total_times = moments[0,:].size
            output_data = np.zeros((total_cases,4,total_times))
            T = mat_data['T']
            
        output_data[kk,:,:] = moments[:,:]
        for ii in range(0,4):
            max_out[kk,ii] = np.max(output_data[kk,ii,:])
        
    del mat_data

        
    print('Done!')
    return output_data, max_out, total_cases, total_times, T;
    #return MC_moments, total_times, TIME, dt, pratios;
##### 1. Load Matlab data from Gaussian closure of nonlinear RP - END. ########


def Load_Data_QBMM(T,mc_times):
    
    print('Importing QBMM data... ', end='', flush=True)
    
    folder_name = '../data/Constant_Forcing/';
    qbmm_file_format = 'qbmm_state_Constant_Pressure';
    
    file_names = []
    file_names = [folder_name+qbmm_file_format+str(ii)+'.dat' for ii in range(25,100,5)]
    
    total_cases = len(file_names)
    max_in = np.zeros((total_cases,5))
    qbmm_moments = np.zeros((total_cases,5,mc_times),dtype=float)
    for kk in range(0,total_cases):
        data = np.genfromtxt(file_names[kk],dtype=float,delimiter='')
        for ii in range(0,5):
            #f = interpolate.interp1d(data[:,0],data[:,2+ii])
            f = interpolate.InterpolatedUnivariateSpline(data[:,0],data[:,2+ii],k=2)
            qbmm_moments[kk,ii,:] = f(T)
            max_in[kk,ii] = np.max(qbmm_moments[kk,ii,:])
        



    print('Done!')
    return qbmm_moments, max_in;



