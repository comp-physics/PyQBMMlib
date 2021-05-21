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
from loss_params import loss_params



###### 1. Load Matlab data from Gaussian closure of nonlinear RP - START. #####
def Load_HM_Data_Output(cases):
    
    print('Importing Monte-Carlo data... ', end='', flush=True)

    if (cases == "constant"):
        folder_name = '../data/Constant_Forcing/';
        mc_file_format = 'MC_HM_Constant_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
        MC_file_names = []
        MC_file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
    
        QBMM_file_names = []
        QBMM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
        
    elif (cases == "sine"):
        folder_name = '../data/Sinusoidal_Forcing/';
        mc_file_format = 'MC_HM_Sinusoidal_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Sinusoidal_Pressure';
        MC_file_names = []
        QBMM_file_names = []
        for jj in range(3,10):
            MC_file_names[17*(jj-3):17*(jj-2)]   = [folder_name+mc_file_format+str(ii)+'_Period'+str(jj)+'.mat' for ii in range(5,90,5)]
            QBMM_file_names[17*(jj-3):17*(jj-2)] = [folder_name+qbmm_HM_file_format+str(ii)+'_Period'+str(jj)+'.mat' for ii in range(5,90,5)]
            
    elif (cases == "random"):
        folder_name = '../data/Random_Forcing/';
        mc_file_format = 'MC_HM_Random_Pressure_Realization';
        qbmm_HM_file_format = 'QBMM_HM_Random_Pressure_Realization';
        MC_file_names = []
        QBMM_file_names = []
        MC_file_names   = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(1,31,1)]
        QBMM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(1,31,1)]
            
        

    input_vals  = [1,2,3,4,5,6,7,17,21] #mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu32, mu3(1-gamma)0
    output_vals = [6,7,17,21] #mu30, mu21, mu32, mu3(1-gamma)0 
    input_size = np.size(input_vals)+1
    total_cases = len(MC_file_names)
    for kk in range(0,total_cases):
        mat_data = sio.loadmat(MC_file_names[kk])
        moments  = mat_data['moments']
        pressure = mat_data['pressure']
        if (kk == 0):
            max_out = np.zeros((total_cases,4))
            max_in  = np.zeros((total_cases,input_size))
            total_times = moments[0,:].size
            total_times = 2001
            output_data = np.zeros((total_cases,4,total_times))
            qbmm_HM = np.zeros((total_cases,4,total_times),dtype=float)
            input_data = np.zeros((total_cases,input_size,total_times),dtype=float)
            qbmm_moments = np.zeros((total_cases,input_size,total_times),dtype=float)
            T_old = mat_data['T']
            T = [ 0.10*ii for ii in range(0,total_times) ]
        
        for ii in range(0,4):
            output_data[kk,ii,:] = moments[output_vals[ii],0:2001:1]
        
        
        mat_data = sio.loadmat(QBMM_file_names[kk])
        T_qbmm = mat_data['T']
        qbmm_moments  = mat_data['moments']
        for ii in range(0,input_size-1):
            f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,input_vals[ii]],k=2)
            input_data[kk,ii,:] = f(T)
        f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
        input_data[kk,input_size-1,:] = f(T)
            
        for ii in range(0,4):
            f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,output_vals[ii]],k=2)
            qbmm_HM[kk,ii,:] = f(T)
        
        for ii in range(0,4):
            output_data[kk,ii,:] = output_data[kk,ii,:] -1.0*qbmm_HM[kk,ii,:]
        
        
        for ii in range(0,input_size):
            max_in[kk,ii] = np.max(input_data[kk,ii,:])
            
        for ii in range(0,4):
            #max_out[kk,ii] = np.max(output_data[kk,ii,:])
            max_out[kk,ii] = np.max(input_data[kk,ii+5,:])
    
    for ii in range(0,input_size):        
        max_in[:,ii] = np.max(max_in[:,ii])
        
    for ii in range(0,4):
        max_out[:,ii] = np.max(max_out[:,ii])
    
    del mat_data

        
    print('Done!')
    return output_data, qbmm_HM, max_out, total_cases, total_times, T, input_data, max_in;
    #return MC_moments, total_times, TIME, dt, pratios;
##### 1. Load Matlab data from Gaussian closure of nonlinear RP - END. ########




###### 1. Load Matlab data from Gaussian closure of nonlinear RP - START. #####
def Load_LM_Data_Output(cases,approach):
    
    print('Importing Monte-Carlo data... ', end='', flush=True)

    if (cases == "constant"):
        folder_name = '../data/Constant_Forcing/';
        mc_file_format = 'MC_HM_Constant_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
        MC_file_names = []
        MC_file_names = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
    
        QBMM_file_names = []
        QBMM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
        
    elif (cases == "sine"):
        folder_name = '../data/Sinusoidal_Forcing/';
        mc_file_format = 'MC_HM_Sinusoidal_Pressure';
        qbmm_HM_file_format = 'QBMM_HM_Sinusoidal_Pressure';
        MC_file_names = []
        QBMM_file_names = []
        for jj in range(3,10):
            MC_file_names[17*(jj-3):17*(jj-2)]   = [folder_name+mc_file_format+str(ii)+'_Period'+str(jj)+'.mat' for ii in range(5,90,5)]
            QBMM_file_names[17*(jj-3):17*(jj-2)] = [folder_name+qbmm_HM_file_format+str(ii)+'_Period'+str(jj)+'.mat' for ii in range(5,90,5)]
            
    elif (cases == "random"):
        folder_name = '../data/Random_Forcing/';
        mc_file_format = 'MC_HM_Random_Pressure_Realization';
        qbmm_HM_file_format = 'QBMM_HM_Random_Pressure_Realization';
        MC_file_names = []
        QBMM_file_names = []
        MC_file_names   = [folder_name+mc_file_format+str(ii)+'.mat' for ii in range(1,11,1)]
        QBMM_file_names = [folder_name+qbmm_HM_file_format+str(ii)+'.mat' for ii in range(1,11,1)]

    if (approach == "1"):
        input_vals  = [1,2,3,4,5] #mu10, mu01, mu20, mu11, mu02
        output_vals = [1,2,3,4,5] #mu10, mu01, mu20, mu11, mu02 
        input_size = np.size(input_vals)+1
        output_dim = np.size(output_vals)
        ml_dim     = output_dim
    elif (approach == "2"):
        input_vals  = []
        output_vals = [0,
                       1,2,
                       3,4,5,
                       6,7,8,9,
                       10,11,12,13,14,
                       15,16,17,18,19,20,
                       21,
                       22,23,24,25,26,27,28,29,30] #mu00, mu10, mu01
        input_size = np.size(input_vals)+1
        output_dim = np.size(output_vals)
        ml_dim     = int(6)
    elif (approach == "3"):
        input_vals  = [1,2,3,4,5] #mu10, mu01, mu20, mu11, mu02
        output_vals = [6,8,9,13] #mu30, mu03, mu40, mu04
        input_size = np.size(input_vals)+1
        output_dim = np.size(output_vals)
        ml_dim     = output_dim
    elif (approach == "4"):
        input_vals  = [1,2,3,4,5] #mu10, mu01, mu20, mu11, mu02
        output_vals = [0,
                       1,2,
                       3,4,5,
                       6,7,8,9,
                       10,11,12,13,14,
                       15,16,17,18,19,20,
                       21,
                       22,23,24,25,26,27,28,29,30] #mu00, mu10, mu01
#        output_vals = [0,
#                       1,2,
#                       3,4,5,
#                       6,7,8,9,
#                       10,11,12,13,14,
#                       15,16,17,18,19,20] #mu00, mu10, mu01
        input_size = np.size(input_vals)+1
        output_dim = np.size(output_vals)+1
        ml_dim     = int(12)
        xi_max = 0.0
        xid_max = 0.0
        
    used_features = input_size
    total_cases = len(MC_file_names)
    for kk in range(0,total_cases):
        mat_data = sio.loadmat(MC_file_names[kk])
        moments  = mat_data['moments']
        pressure = mat_data['pressure']
        if (kk == 0):
            max_out = np.zeros((total_cases,output_dim))
            max_in  = np.zeros((total_cases,input_size))
            total_times = moments[0,:].size
            total_times = 4001
            lm_times = 4001
            LM_QBMM = np.zeros((total_cases,30,lm_times))
            LM_MC   = np.zeros((total_cases,30,lm_times))
            LM_pressure = np.zeros((total_cases,lm_times))
            output_data = np.zeros((total_cases,output_dim,total_times))
            qbmm_LM = np.zeros((total_cases,output_dim,total_times),dtype=float)
            input_data = np.zeros((total_cases,input_size,total_times),dtype=float)
            qbmm_moments = np.zeros((total_cases,input_size,total_times),dtype=float)
            T_old = mat_data['T']
            T = [ 0.05*ii for ii in range(0,total_times) ]
            T_LM = [ 0.05*ii for ii in range(0,lm_times) ]
        
        for ii in range(0,output_dim-1):
            output_data[kk,ii,:] = moments[output_vals[ii],0:4001:1]
        
        
        mat_data = sio.loadmat(QBMM_file_names[kk])
        T_qbmm = mat_data['T']
        qbmm_moments  = mat_data['moments']
        if (approach == "1"):
            for ii in range(0,input_size-1):
                f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,input_vals[ii]],k=2)
                input_data[kk,ii,:] = f(T)
            f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
            input_data[kk,input_size-1,:] = f(T)
        elif (approach == "2"):
            for ii in range(0,input_size-1):
                f = interpolate.InterpolatedUnivariateSpline(T_old,moments[input_vals[ii],:],k=2)
                input_data[kk,ii,:] = f(T)
            f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
            input_data[kk,input_size-1,:] = f(T)
        elif (approach == "4"):
            for ii in range(0,input_size-1):
                #f = interpolate.InterpolatedUnivariateSpline(T_old,moments[input_vals[ii],:],k=2)
                f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,input_vals[ii]],k=2)
                input_data[kk,ii,:] = f(T)
            f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
            input_data[kk,input_size-1,:] = f(T)
            output_data[kk,output_dim-1,:] = f(T)
        
        for ii in range(0,30):
            f = interpolate.InterpolatedUnivariateSpline(T,moments[1+ii,:],k=2)
            LM_MC[kk,ii,:] = f(T_LM)
            f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,1+ii],k=2)
            LM_QBMM[kk,ii,:] = f(T_LM)
        f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
        LM_pressure[kk,:] = f(T_LM)
        
        
        if (approach == "1"):    
            for ii in range(0,output_dim-1):
                f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,output_vals[ii]],k=2)
                qbmm_LM[kk,ii,:] = f(T)
        
        for ii in range(0,output_dim-1):
            output_data[kk,ii,:] = output_data[kk,ii,:] -1.0*qbmm_LM[kk,ii,:]
        
        if (approach == "1"):
            for ii in range(0,input_size):
                max_in[kk,ii] = np.max(input_data[kk,ii,:])
            for ii in range(0,output_dim):
                #max_out[kk,ii] = np.max(output_data[kk,ii,:])
                max_out[kk,ii] = np.max(input_data[kk,ii,:])
        elif (approach == "2"):
            for ii in range(0,input_size):
                max_in[kk,ii] = 1.0
            for ii in range(0,output_dim):
                #max_out[kk,ii] = np.max(output_data[kk,ii,:])
                max_out[kk,ii] = 1.0
        elif (approach == "4"):
            for ii in range(0,input_size):
                max_in[kk,ii] = 1.0
            for ii in range(0,output_dim):
                #max_out[kk,ii] = np.max(output_data[kk,ii,:])
                max_out[kk,ii] = 1.0


            
    for ii in range(0,input_size):        
        max_in[:,ii] = np.max(max_in[:,ii])
        
    for ii in range(0,output_dim):
        max_out[:,ii] = np.max(max_out[:,ii])
    
    
    
    
    del mat_data

        
    print('Done!')
    return output_data, qbmm_LM, max_out, total_cases, total_times, T, input_data, max_in, used_features, output_dim, ml_dim, LM_MC, LM_QBMM, LM_pressure, lm_times, input_size;
    #return MC_moments, total_times, TIME, dt, pratios;
##### 1. Load Matlab data from Gaussian closure of nonlinear RP - END. ########





















































