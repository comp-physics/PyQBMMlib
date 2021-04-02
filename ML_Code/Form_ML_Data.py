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
def Load_Data_Output(cases):
    
    print('Importing Monte-Carlo data... ', end='', flush=True)

    if (cases == "constant"):
        folder_name = '../data/Constant_Forcing/';
        mc_file_format = 'MC_HM_Constant_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
        MC_file_names = []
        MC_file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(15,100,5)]
    
        QBMM_file_names = []
        QBMM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(15,100,5)]
        
    elif (cases == "sine"):
        folder_name = '../data/Sinusoidal_Forcing/';
        mc_file_format = 'MC_HM_Sinusoidal_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Sinusoidal_Pressure';
        MC_file_names = []
        MC_file_names[0:17] = [folder_name+mc_file_format+str(ii)+'_Period3'+'.mat' for ii in range(5,90,5)]
        #MC_file_names[17:34] = [folder_name+mc_file_format+str(ii)+'_Period5'+'.mat' for ii in range(5,90,5)]
    
        QBMM_file_names = []
        QBMM_file_names[0:17] = [folder_name+qbmm_HM_file_format+str(ii)+'_Period3'+'.mat' for ii in range(5,90,5)]
        #QBMM_file_names[17:34] = [folder_name+qbmm_HM_file_format+str(ii)+'_Period5'+'.mat' for ii in range(5,90,5)]
        

    total_cases = len(MC_file_names)
    for kk in range(0,total_cases):
        mat_data = sio.loadmat(MC_file_names[kk])
        moments  = mat_data['moments']
        if (kk == 0):
            max_out = np.zeros((total_cases,4))
            total_times = moments[0,:].size
            output_data = np.zeros((total_cases,4,total_times))
            qbmm_HM = np.zeros((total_cases,4,total_times),dtype=float)
            T = mat_data['T']
            
        output_data[kk,:,:] = moments[:,:]
        
        mat_data = sio.loadmat(QBMM_file_names[kk])
        T_qbmm = mat_data['T']
        moments  = mat_data['moments']
        for ii in range(0,4):
            f = interpolate.InterpolatedUnivariateSpline(T_qbmm,moments[:,ii],k=2)
            qbmm_HM[kk,ii,:] = f(T)
        
        output_data[kk,:,:] = output_data[kk,:,:] -qbmm_HM[kk,:,:]
        
        for ii in range(0,4):
            max_out[kk,ii] = np.max(output_data[kk,ii,:])
        
    del mat_data

        
    print('Done!')
    return output_data, qbmm_HM, max_out, total_cases, total_times, T;
    #return MC_moments, total_times, TIME, dt, pratios;
##### 1. Load Matlab data from Gaussian closure of nonlinear RP - END. ########


def Load_Data_QBMM(cases,T,mc_times):
    
    print('Importing QBMM data... ', end='', flush=True)
    
    if (cases == "constant"):
        folder_name = '../data/Constant_Forcing/';
        qbmm_LM_file_format = 'qbmm_state_Constant_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
        LM_file_names = []
        LM_file_names = [folder_name+qbmm_LM_file_format+str(ii)+'.dat' for ii in range(15,100,5)]
        
    elif (cases == "sine"):
        folder_name = '../data/Sinusoidal_Forcing/';
        qbmm_LM_file_format = 'qbmm_state_Sinusoidal_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Sinusoidal_Pressure';
        LM_file_names = []
        LM_file_names[0:17] = [folder_name+qbmm_LM_file_format+str(ii)+'_Period3'+'.dat' for ii in range(5,90,5)]
        #LM_file_names[17:34] = [folder_name+qbmm_LM_file_format+str(ii)+'_Period5'+'.dat' for ii in range(5,90,5)]
    
#    HM_file_names = []
#    HM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(15,100,5)]
    
    
    
    folder_name = '../data/Cons'
    
    
    total_cases = len(LM_file_names)
    max_in = np.zeros((total_cases,5))
    qbmm_moments = np.zeros((total_cases,5,mc_times),dtype=float)
    #qbmm_HM = np.zeros((total_cases,4,mc_times),dtype=float)
    for kk in range(0,total_cases):
        data = np.genfromtxt(LM_file_names[kk],dtype=float,delimiter='')
        for ii in range(0,5):
            #f = interpolate.interp1d(data[:,0],data[:,2+ii])
            f = interpolate.InterpolatedUnivariateSpline(data[:,0],data[:,2+ii],k=2)
            qbmm_moments[kk,ii,:] = f(T)
            max_in[kk,ii] = np.max(qbmm_moments[kk,ii,:])
            
#        mat_data = sio.loadmat(HM_file_names[kk])
#        T_qbmm = mat_data['T']
#        moments  = mat_data['moments']
#        for ii in range(0,4):
#            f = interpolate.InterpolatedUnivariateSpline(T_qbmm,moments[:,ii],k=2)
#            qbmm_HM[kk,ii,:] = f(T)



    print('Done!')
    return qbmm_moments, max_in;



