#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:59:46 2021

@author: alexis
"""

# Helper libraries
print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
from tensorflow import keras 
print('Done!')

print('Importing numpy... ', end='', flush=True)
import numpy as np
from numpy import linspace
from numpy import fft
print('Done!')
print('Importing scipy... ', end='', flush=True)
import scipy.io as sio
from scipy import interpolate
print('Done!')
print('Importing matplotlib.pyplot... ', end='', flush=True)
#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
print('Done!')
print('Importing math... ', end='', flush=True)
import math
print('Done!')
import tkinter
from scipy.signal import savgol_filter


import tensorflow.keras.backend as kb
import tensorflow.keras.losses

from loss_params import loss_params


#def custom_loss(y_actual,y_pred):
#    custom_loss=kb.square(y_actual[:,:,0]-y_pred[:,:,0])+kb.square(y_actual[:,:,0]*y_actual[:,:,1]-y_pred[:,:,1])+kb.square(y_actual[:,:,0]*y_actual[:,:,2]-y_pred[:,:,2])
#    return custom_loss
#keras.losses.custom_loss = custom_loss

def param_loss(alpha):

    def custom_loss(y_actual,y_pred):
    
        npoints = alpha.npoints
        ids = alpha.ids
        Re = alpha.Re
        
        
        
        #xi_con = 1.0
        #xid_con = 0.0
        xi_con  = y_actual[:,:,1]
        xid_con = y_actual[:,:,2] 

        # Constraint the sum of weights yo be equal to 1
        ymom = y_actual[:,:,32+3*0]+y_pred[:,:,0]
        for ii in range(1,npoints):
            ymom = ymom +y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii]
        yloss = 100.0*kb.square(ymom-y_actual[:,:,0])   
        
        # All weights should be positive
#        for ii in range(0,npoints):
#            yloss = yloss +100.0*kb.relu(-(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii]))   
            
        # All xi-points should be positive
#        for ii in range(0,npoints):
#            yloss = yloss +100.0*kb.relu(-(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1]))
            

        
        # Demand M_{0,2} > M_{0,1}^2, (M_{2,0} > M_{1,0}^2)
        
#        ymom01 = (y_actual[:,:,32+3*0]+y_pred[:,:,0])*(y_actual[:,:,32+3*0+2]+y_pred[:,:,2])
#        ymom02 = (y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],2)
#        ymom11 = (y_actual[:,:,32+3*0]+y_pred[:,:,0])*(y_actual[:,:,32+3*0+1]+y_pred[:,:,1])*(y_actual[:,:,32+3*0+2]+y_pred[:,:,2])
#        ymom10 = (y_actual[:,:,32+3*0]+y_pred[:,:,0])*(y_actual[:,:,32+3*0+1]+y_pred[:,:,1])
#        ymom20 = (y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],2)
#        for ii in range(1,npoints):
#            ymom01 = ymom01+(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2])
#            ymom02 = ymom02+(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],2)
#            ymom11 = ymom11+(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1])*(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2])
#            ymom10 = ymom10+(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1])
#            ymom20 = ymom20+(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],2)
#            
#        yloss = yloss +kb.square( (y_actual[:,:,3] -y_actual[:,:,1]*y_actual[:,:,1]) -(ymom20 -ymom10*ymom10)  )
#        yloss = yloss +kb.square( (y_actual[:,:,5] -y_actual[:,:,2]*y_actual[:,:,2]) -(ymom02 -ymom01*ymom01)  )
#        yloss = yloss +kb.square( (y_actual[:,:,4] -y_actual[:,:,1]*y_actual[:,:,2]) -(ymom11 -ymom10*ymom01)  )
        
        
        
        # Constraint to match particular moments
        loss_mom = [1,2,   # first-order moments
                    3,4,5,
                     6, 7,
                    17,
                    21] # second-order moments 
#        loss_coef = [1.0,5.0,
#                     1.0,5.0,20.0,
#                     1.0,7.0,
#                     25.0,
#                     1.0]
        loss_coef = alpha.mom_scale_coeffs
        
        
        counter = int(-1)
        for kflag in loss_mom:
            counter = counter+1
            ymom = (y_actual[:,:,32]+y_pred[:,:,0])*pow(y_actual[:,:,33]+y_pred[:,:,1],ids[0,kflag])*pow(y_actual[:,:,34]+y_pred[:,:,2],ids[1,kflag])
            for ii in range(1,npoints):
                ymom = ymom +(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,kflag])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,kflag])
            yloss = yloss +kb.square(ymom-y_actual[:,:,kflag])/pow(loss_coef[0,kflag-1],2)
        
        
        rhs_coef = alpha.rhs_scale_coeffs
        # RHS for evolution of moment M_{0,1}
        y_RHS = -1.5*y_actual[:,:,22] -(4.0/Re)*y_actual[:,:,23] +y_actual[:,:,24] -y_actual[:,:,31]*y_actual[:,:,25]
        ymom = -1.5*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,22])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,22])
        ymom = ymom -(4.0/Re)*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,23])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,23])
        ymom = ymom +(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,24])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,24])
        ymom = ymom -y_actual[:,:,31]*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,25])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,25])
        for ii in range(1,npoints):
            ymom = ymom -1.5*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,22])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,22])
            ymom = ymom -(4.0/Re)*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,23])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,23])
            ymom = ymom +(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,24])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,24])
            ymom = ymom -y_actual[:,:,31]*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,25])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,25])
        yloss = yloss +kb.square(ymom -y_RHS)/pow(rhs_coef[0,1],2)
        
        
        # RHS for evolution of moment M_{1,1}
        y_RHS = -0.5*y_actual[:,:,5] -(4.0/Re)*y_actual[:,:,26] +y_actual[:,:,27]
        ymom = -0.5*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,5])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,5])
        ymom = ymom -(4.0/Re)*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,26])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,26])
        ymom = ymom +(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,27])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,27])
        for ii in range(1,npoints):
            ymom = ymom -0.5*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,5])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,5])
            ymom = ymom -(4.0/Re)*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,26])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,26])
            ymom = ymom +(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,27])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,27])
        yloss = yloss +kb.square(ymom -y_RHS)/pow(rhs_coef[0,3],2)
        
        
        
        # RHS for evolution of moment M_{0,2}
        y_RHS = -3.0*y_actual[:,:,28] -(8.0/Re)*y_actual[:,:,29] +2.0*y_actual[:,:,30] -2.0*y_actual[:,:,31]*y_actual[:,:,26]
        ymom = -3.0*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,28])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,28])
        ymom = ymom -(8.0/Re)*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,29])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,29])
        ymom = ymom +2.0*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,30])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,30])
        ymom = ymom -2.0*y_actual[:,:,31]*(y_actual[:,:,32+3*0]+y_pred[:,:,0])*pow(y_actual[:,:,32+3*0+1]+y_pred[:,:,1],ids[0,26])*pow(y_actual[:,:,32+3*0+2]+y_pred[:,:,2],ids[1,26])
        for ii in range(1,npoints):
            ymom = ymom -3.0*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,28])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,28])
            ymom = ymom -(8.0/Re)*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,29])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,29])
            ymom = ymom +2.0*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,30])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,30])
            ymom = ymom -2.0*y_actual[:,:,31]*(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,26])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,26])
        yloss = yloss +kb.square(ymom -y_RHS)/pow(rhs_coef[0,4],2)
        
        return yloss
    return custom_loss
    


# Define it as sum w_i x-dot_i/x_dot
def param_loss2(alpha):

    def custom_loss(y_actual,y_pred):
    
        npoints = alpha.npoints
        ids = alpha.ids
        Re = alpha.Re

        # Constraint the sum of weights yo be equal to 1
        ymom = y_pred[:,:,0]
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]
        yloss = 10.0*kb.square(ymom-y_actual[:,:,0])   
        
        # All weights should be positive
        for ii in range(0,npoints):
            yloss = yloss +10.0*kb.relu(-y_pred[:,:,3*ii])   
            
        # Shouldn't make xi values too small
        for ii in range(0,npoints):
            yloss = yloss +10.0*kb.relu(-y_pred[:,:,3*ii+1]+0.20)
            
        # Constraint to match M_{1,0}
        kflag = 1
        yval = y_pred[:,:,1]
        for ii in range(1,npoints):
            yval = yval*y_pred[:,:,3*ii+1]
        ymom = y_pred[:,:,0]*yval*pow(y_pred[:,:,1],-ids[0,kflag])
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]*yval*pow(y_pred[:,:,3*ii+1],-ids[0,kflag])
        yloss = yloss +kb.square(ymom-y_actual[:,:,kflag]*yval)
        
        # Constraint to match M_{2,0}
        kflag = 3
        yval = pow(y_pred[:,:,1],ids[0,kflag])
        for ii in range(1,npoints):
            yval = yval*pow(y_pred[:,:,3*ii+1],ids[0,kflag])
        ymom = y_pred[:,:,0]*yval*pow(y_pred[:,:,1],-ids[0,kflag])
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]*yval*pow(y_pred[:,:,3*ii+1],-ids[0,kflag])
        yloss = yloss +kb.square(ymom-y_actual[:,:,kflag]*yval)
        
        
        # Constraint to match M_{0,2}
        kflag = 5
        ymom = y_pred[:,:,0]*pow(y_pred[:,:,2],ids[1,kflag])
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+2],ids[1,kflag])
        yloss = yloss +kb.square(ymom-y_actual[:,:,kflag])
            
            
        
#        loss_moms = [22,23,24,25,26,27,28,29,30]
#        for kflag in loss_moms:
#        #kflag = 25
#            ymom = y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,kflag])*pow(y_pred[:,:,2],ids[1,kflag])
#            for ii in range(1,npoints):
#                ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,kflag])*pow(y_pred[:,:,3*ii+2],ids[1,kflag])
#            yloss = yloss +kb.square(ymom-y_actual[:,:,kflag]) 
        
    
        # RHS for evolution of moment M_{1,0}
        kflag = 2
        ymom = y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,kflag])*pow(y_pred[:,:,2],ids[1,kflag])
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,kflag])*pow(y_pred[:,:,3*ii+2],ids[1,kflag])
        yloss = yloss +kb.square(ymom-y_actual[:,:,kflag])
        
        # RHS for evolution of moment M_{0,1}
        y_RHS = -1.5*y_actual[:,:,22] -(4.0/Re)*y_actual[:,:,23] +y_actual[:,:,24] -y_actual[:,:,31]*y_actual[:,:,25]
        #y_RHS = -1.5*y_actual[:,:,22] -(4.0/Re)*y_actual[:,:,23] -y_actual[:,:,31]*y_actual[:,:,25]
        ymom = -1.5*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,22])*pow(y_pred[:,:,2],ids[1,22])
        ymom = ymom -(4.0/Re)*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,23])*pow(y_pred[:,:,2],ids[1,23])
        ymom = ymom +y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,24])*pow(y_pred[:,:,2],ids[1,24])
        ymom = ymom -y_actual[:,:,31]*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,25])*pow(y_pred[:,:,2],ids[1,25])
        for ii in range(1,npoints):
            ymom = ymom -1.5*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,22])*pow(y_pred[:,:,3*ii+2],ids[1,22])
            ymom = ymom -(4.0/Re)*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,23])*pow(y_pred[:,:,3*ii+2],ids[1,23])
            ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,24])*pow(y_pred[:,:,3*ii+2],ids[1,24])
            ymom = ymom -y_actual[:,:,31]*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,25])*pow(y_pred[:,:,3*ii+2],ids[1,25])
        yloss = yloss +kb.square(ymom -y_RHS)
        
        # RHS for evolution of moment_{2,0}
        kflag = 4
        yval = y_pred[:,:,1]
        for ii in range(1,npoints):
            yval = yval*y_pred[:,:,3*ii+1]
        ymom = y_pred[:,:,0]*yval*pow(y_pred[:,:,1],-ids[0,kflag])*pow(y_pred[:,:,2],ids[1,kflag])
        for ii in range(1,npoints):
            ymom = ymom +y_pred[:,:,3*ii]*yval*pow(y_pred[:,:,3*ii+1],-ids[0,kflag])*pow(y_pred[:,:,3*ii+2],ids[1,kflag])
        yloss = yloss +kb.square(ymom-y_actual[:,:,kflag]*yval)
        
        # RHS for evolution of moment M_{1,1}
        y_RHS = -0.5*y_actual[:,:,5] -(4.0/Re)*y_actual[:,:,26] +y_actual[:,:,27] -y_actual[:,:,31]
        ymom = -0.5*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,5])*pow(y_pred[:,:,2],ids[1,5])
        ymom = ymom -(4.0/Re)*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,26])*pow(y_pred[:,:,2],ids[1,26])
        ymom = ymom +y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,27])*pow(y_pred[:,:,2],ids[1,27])
        ymom = ymom -y_actual[:,:,31]
        for ii in range(1,npoints):
            ymom = ymom -0.5*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,5])*pow(y_pred[:,:,3*ii+2],ids[1,5])
            ymom = ymom -(4.0/Re)*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,26])*pow(y_pred[:,:,3*ii+2],ids[1,26])
            ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,27])*pow(y_pred[:,:,3*ii+2],ids[1,27])
            ymom = ymom -y_actual[:,:,31]
        yloss = yloss +kb.square(ymom -y_RHS)
        
        # RHS for evolution of moment M_{0,2}
        y_RHS = -3.0*y_actual[:,:,28] -(8.0/Re)*y_actual[:,:,29] +2.0*y_actual[:,:,30] -2.0*y_actual[:,:,31]*y_actual[:,:,26]
        ymom = -3.0*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,28])*pow(y_pred[:,:,2],ids[1,28])
        ymom = ymom -(8.0/Re)*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,29])*pow(y_pred[:,:,2],ids[1,29])
        ymom = ymom +2.0*y_pred[:,:,0]*pow(y_pred[:,:,1],-ids[0,30])*pow(y_pred[:,:,2],ids[1,30])
        ymom = ymom -2.0*y_actual[:,:,31]*pow(y_pred[:,:,1],-ids[0,26])*pow(y_pred[:,:,2],ids[1,26])
        for ii in range(1,npoints):
            ymom = ymom -3.0*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,28])*pow(y_pred[:,:,3*ii+2],ids[1,28])
            ymom = ymom -(8.0/Re)*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,29])*pow(y_pred[:,:,3*ii+2],ids[1,29])
            ymom = ymom +2.0*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,30])*pow(y_pred[:,:,3*ii+2],ids[1,30])
            ymom = ymom -2.0*y_actual[:,:,31]*y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],-ids[0,26])*pow(y_pred[:,:,3*ii+2],ids[1,26])
        yloss = yloss +kb.square(ymom -y_RHS)
        
        
        return yloss
    return custom_loss








def LM_Training(input_data,output_data,qbmm_HM,max_in,max_out,total_times,test_cases,cases,approach,used_features,output_dim,ml_dim):
    
    time_start = 0
    time_history = 256
    time_jump  = 1
    sub_sample = 1
    train_times = total_times-time_history*time_jump-time_start
    train_times = int(np.floor(train_times//sub_sample))

    test_size = len(test_cases)
    #used_features = 5
    input_train = np.zeros((test_size*train_times,time_history,used_features))
    output_train  = np.zeros((test_size*train_times,time_history,output_dim))

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,used_features):
                for pp in range(0,time_history):
                    input_train[ii*train_times+tt,pp,jj]  =  input_data[test_cases[ii],jj,sub_sample*tt+time_jump*pp+time_start]/max_in[test_cases[ii],jj]

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,output_dim):
                for pp in range(0,time_history):
                    output_train[ii*train_times+tt,pp,jj]  =  output_data[test_cases[ii], jj,tt+time_jump*pp+time_start]/max_out[test_cases[ii],jj]

    
    train_end = round(test_size*train_times*0.9)
    
    
    npoints = int(ml_dim//3)
    
    ids = np.zeros((2,31))
    ids[0, 0] = 0.0; ids[1, 0] = 0.0
    
    ids[0, 1] = 1.0; ids[1, 1] = 0.0
    ids[0, 2] = 0.0; ids[1, 2] = 1.0
    
    ids[0, 3] = 2.0; ids[1, 3] = 0.0
    ids[0, 4] = 1.0; ids[1, 4] = 1.0
    ids[0, 5] = 0.0; ids[1, 5] = 2.0
    
    ids[0, 6] = 3.0; ids[1, 6] = 0.0
    ids[0, 7] = 2.0; ids[1, 7] = 1.0
    ids[0, 8] = 1.0; ids[1, 8] = 2.0
    ids[0, 9] = 0.0; ids[1, 9] = 3.0
    
    ids[0,10] = 4.0; ids[1,10] = 0.0
    ids[0,11] = 3.0; ids[1,11] = 1.0
    ids[0,12] = 2.0; ids[1,12] = 2.0
    ids[0,13] = 1.0; ids[1,13] = 3.0
    ids[0,14] = 0.0; ids[1,14] = 4.0
    
    ids[0,15] = 5.0; ids[1,15] = 0.0
    ids[0,16] = 4.0; ids[1,16] = 1.0
    ids[0,17] = 3.0; ids[1,17] = 2.0
    ids[0,18] = 2.0; ids[1,18] = 3.0
    ids[0,19] = 1.0; ids[1,19] = 4.0
    ids[0,20] = 0.0; ids[1,20] = 5.0
    
    ids[0,21] = 3.0*(1.0-1.4); ids[1,21] = 0.0
    
    ids[0,22] =-1.0; ids[1,22] = 2.0
    ids[0,23] =-2.0; ids[1,23] = 1.0
    ids[0,24] =-4.0; ids[1,24] = 0.0
    ids[0,25] =-1.0; ids[1,25] = 0.0
    ids[0,26] =-1.0; ids[1,26] = 1.0
    ids[0,27] =-3.0; ids[1,27] = 0.0
    ids[0,28] =-1.0; ids[1,28] = 3.0
    ids[0,29] =-2.0; ids[1,29] = 2.0
    ids[0,30] =-4.0; ids[1,30] = 1.0
    
    loss_data = loss_params(1000.0,input_train[:,:,5],ids,npoints)
    
    
    
    
    
    
    

    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.LSTM(input_shape=(None,used_features),units=time_history, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    #model1.add(tf.keras.layers.Dense(ml_dim))
    #model1.add(tf.keras.layers.Dropout(rate=0.05))
    model1.add(tf.keras.layers.Dense(ml_dim, activation='linear'))
    if (approach =="1"):
        model1.compile(loss='mse', optimizer='adam' )
    elif (approach == "2"):
        model1.compile(loss=custom_loss, optimizer='adam' )
    elif (approach == "3"):
        model1.compile(loss='mse', optimizer='adam' )
    elif (approach == "4"):
        tf.keras.optimizers.Adam(learning_rate=1.0*pow(10.0,-7))
        model1.compile(loss=param_loss(alpha=loss_data), optimizer='Adam' )
    hist1 = model1.fit(input_train[0:train_end,:,:], output_train[0:train_end,:,:], batch_size=32, epochs=5, validation_data=(input_train[train_end:test_size*train_times,:,:],output_train[train_end:test_size*train_times,:,:]))
    if (cases == "constant"):
        model1.save('Neural_Nets/LM_Constant_MLQBMM_Approach'+approach+'.h5')
    elif (cases == "sine"):
        model1.save('Neural_Nets/LM_Sinusoidal_MLQBMM_Approach'+approach+'.h5')
    elif (cases == "random"):
        if (approach == "1"):
            model1.save('Neural_Nets/LM_Random_MLQBMM_Approach'+approach+'.h5')
        elif (approach == "2"):
            model1.save('Neural_Nets/LM_Random_MLQBMM_Approach'+approach+'_Weights'+str(ml_dim//3)+'_Pressure.h5')
        elif (approach == "4"):
            model1.save('Neural_Nets/LM_Random_MLQBMM_Approach'+approach+'_Weights'+str(ml_dim//3)+'.h5')
    
    return;
    































def HM_Testing(input_data,output_data,qbmm_HM,max_in,max_out,total_times,used_features,test_cases,cases):

    if (cases == "constant"):
        model1 = tf.keras.models.load_model('HM_MLQBMM.h5')
    elif (cases == "sine"):
        model1 = tf.keras.models.load_model('HM_Sinusoidal_MLQBMM.h5')
    elif (cases == "random"):
        model1 = tf.keras.models.load_model('HM_Random_MLQBMM.h5')

    test_size = len(test_cases)

    input_test   = np.zeros((test_size,total_times,used_features))
    output_test  = np.zeros((test_size,total_times,4))

    for ii in range(0,test_size):
        for jj in range(0,used_features):
            for pp in range(0,total_times):
                input_test[ii,pp,jj] = input_data[test_cases[ii],jj,pp]/max_in[test_cases[ii],jj]
            
    output_test = model1.predict(input_test)

    predictions = np.zeros((test_size,4,total_times))
    for ii in range(0,test_size):
        for tt in range(0,total_times):
            for jj in range(0,4):
                predictions[ii,jj,tt] = output_test[ii,tt,jj]*max_out[test_cases[ii],jj]+1.0*qbmm_HM[test_cases[ii],jj,tt]
    
    return predictions;    
    
    
    




def LM_Testing_Approach1(input_data,output_data,qbmm_LM,max_in,max_out,total_times,used_features,test_cases,cases,output_dim,LM_QBMM,LM_MC, LM_pressure,lm_times):

    if (cases == "constant"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Constant_MLQBMM_Approach1.h5')
    elif (cases == "sine"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Sinusoidal_MLQBMM_Approach1.h5')
    elif (cases == "random"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach1.h5',compile=False)
        #model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach1.h5', custom_objects={'custom_loss': custom_loss})

    test_size = len(test_cases)
    
    scale_times = (lm_times-1)//(total_times-1)
    input_test   = np.zeros((scale_times*test_size,total_times,used_features))
    output_test  = np.zeros((scale_times*test_size,total_times,output_dim))

                
    for ii in range(0,test_size):
        for kk in range(0,scale_times):
            for pp in range(0,total_times-1):
                for jj in range(0,5):
                    input_test[scale_times*ii+kk,pp,jj] = LM_QBMM[test_cases[ii],jj,scale_times*pp+kk]/max_in[test_cases[ii],jj]
                input_test[scale_times*ii+kk,pp,5] = LM_pressure[test_cases[ii],scale_times*pp+kk]/max_in[test_cases[ii],5]
        for jj in range(0,5):
            input_test[scale_times*ii,total_times-1,jj] = LM_QBMM[test_cases[ii],jj,lm_times-1]/max_in[test_cases[ii],jj]
        input_test[scale_times*ii,total_times-1,5] = LM_pressure[test_cases[ii],lm_times-1]/max_in[test_cases[ii],5]
            
    output_test = model1.predict(input_test)

    LM_predictions = np.zeros((test_size,5,lm_times))
    for ii in range(0,test_size):
        for kk in range(0,scale_times):
            for tt in range(0,total_times-1):
                for jj in range(0,5):
                    LM_predictions[ii,jj,scale_times*tt+kk] = output_test[scale_times*ii+kk,tt,jj]*max_out[test_cases[ii],jj]+1.0*LM_QBMM[test_cases[ii],jj,scale_times*tt+kk]
        for jj in range(0,5):
            LM_predictions[ii,jj,lm_times-1] = output_test[scale_times*ii,total_times-1,jj]*max_out[test_cases[ii],jj]+1.0*LM_QBMM[test_cases[ii],jj,lm_times-1]
    
    return LM_predictions;    
    




    










def LM_Testing_Approach4(input_data,output_data,qbmm_LM,max_in,max_out,total_times,used_features,test_cases,cases,approach,output_dim,ml_dim,LM_QBMM,LM_MC, LM_pressure,lm_times,input_size):

    if (cases == "constant"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Constant_MLQBMM_Approach4.h5')
    elif (cases == "sine"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Sinusoidal_MLQBMM_Approach4.h5')
    elif (cases == "random"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach'+approach+'_Weights'+str(ml_dim//3)+'.h5',compile=False)
        #model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach1.h5', custom_objects={'custom_loss': custom_loss})

    test_size = len(test_cases)
    
    scale_times = (lm_times-1)//(total_times-1)
    input_test   = np.zeros((scale_times*test_size,total_times,used_features))
    output_test  = np.zeros((scale_times*test_size,total_times,output_dim))

                
    for ii in range(0,test_size):
        for kk in range(0,scale_times):
            for pp in range(0,total_times-1):
                for jj in range(0,input_size-1):
                    input_test[scale_times*ii+kk,pp,jj] = LM_QBMM[test_cases[ii],jj,scale_times*pp+kk]/max_in[test_cases[ii],jj]
                input_test[scale_times*ii+kk,pp,input_size-1] = LM_pressure[test_cases[ii],scale_times*pp+kk]/max_in[test_cases[ii],input_size-1]
        for jj in range(0,input_size-1):
            input_test[scale_times*ii,total_times-1,jj] = LM_QBMM[test_cases[ii],jj,lm_times-1]/max_in[test_cases[ii],jj]
        input_test[scale_times*ii,total_times-1,input_size-1] = LM_pressure[test_cases[ii],lm_times-1]/max_in[test_cases[ii],input_size-1]
            
    output_test = model1.predict(input_test)
    
    predictions = np.zeros((test_size,ml_dim,lm_times))
    
    for ii in range(0,test_size):
        for kk in range(0,scale_times):
            for tt in range(0,total_times-1):
                for jj in range(0,ml_dim):
                    predictions[ii,jj,scale_times*tt+kk] = output_test[scale_times*ii+kk,tt,jj]
        for jj in range(0,ml_dim):
            predictions[ii,jj,lm_times-1] = output_test[scale_times*ii,total_times-1,jj]


    Weight_predictions = np.zeros((len(test_cases),ml_dim,total_times),dtype=float)
    LM_RHS = np.zeros((len(test_cases),5,lm_times))
    abscissas = ml_dim//3
    LM_predictions = np.zeros((test_size,30+ml_dim,lm_times))
    LM_predictions[:,0:30+ml_dim,:] = LM_QBMM[:,0:30+ml_dim,:]
    LM_predictions[:,30:30+ml_dim,:] = LM_QBMM[:,30:30+ml_dim,:]
    time_hist = 256
    input_flag = np.zeros((1,time_hist,used_features))
    dt = 0.0125
    Re = 1000.0
    for ii in range(0,4):
    #for ii in range(0,test_size):
        print(ii)
        for tt in range(0,lm_times):
            M10 = 0.0
            M01 = 0.0
            M20 = 0.0
            M11 = 0.0
            M02 = 0.0
            
            M30 = 0.0
            M21 = 0.0
            M12 = 0.0
            M03 = 0.0
            
            M40 = 0.0
            M31 = 0.0
            M22 = 0.0
            M13 = 0.0
            M04 = 0.0
            
            M50 = 0.0
            M41 = 0.0
            M32 = 0.0
            M23 = 0.0
            M14 = 0.0
            M05 = 0.0
            
            M3g = 0.0
            
            Mm11 = 0.0
            Mm30 = 0.0
            Mm12 = 0.0
            Mm21 = 0.0
            Mm40 = 0.0
            Mm10 = 0.0
            Mm13 = 0.0
            Mm22 = 0.0
            Mm41 = 0.0
            tflag = tt-1
            
            xi_con  = LM_predictions[ii,0,tflag]
            xid_con = LM_predictions[ii,1,tflag]
            #xi_con  = 1.0
            #xid_con = 0.0
            
            if (tt > 16200):
                for pp in range(0,time_hist):
                    for jj in range(0,5):
                        input_flag[0,pp,jj] = LM_predictions[ii,jj,tt-(time_hist-1-pp)*scale_times-1]/max_in[ii,jj]
                    input_flag[0,pp,5] = LM_pressure[ii,tt-(time_hist-1-pp)*scale_times-1]/max_in[ii,5]
                output_flag = model1.predict(input_flag)
                for jj in range(0,ml_dim):
                    predictions[ii,jj,tt-1] = output_flag[0,time_hist-1,jj]
            
            for pp in range(0,abscissas):
#                M01 = LM_MC[ii,1,tflag]
#                M11 = LM_MC[ii,3,tflag]
#                M02 = LM_MC[ii,4,tflag]
#                Mm12 = LM_MC[ii,21,tflag]
#                Mm21 = LM_MC[ii,22,tflag]
#                Mm40 = LM_MC[ii,23,tflag]
#                Mm10 = LM_MC[ii,24,tflag]
#                
#                Mm11 = LM_MC[ii,25,tflag]
#                Mm30 = LM_MC[ii,26,tflag]
#                
#                Mm13 = LM_MC[ii,27,tflag]
#                Mm22 = LM_MC[ii,28,tflag]
#                Mm41 = LM_MC[ii,29,tflag]
                if (LM_predictions[ii,0,tflag] < 2.6):
                    weight_val = LM_predictions[ii,30+3*pp,tflag]+predictions[ii,3*pp,tflag]
                    xi_val     = LM_predictions[ii,30+3*pp+1,tflag]+predictions[ii,3*pp+1,tflag]
                    xid_val    = LM_predictions[ii,30+3*pp+2,tflag]+predictions[ii,3*pp+2,tflag]
                else:
                    weight_val = LM_predictions[ii,30+3*pp,tflag]
                    xi_val     = LM_predictions[ii,30+3*pp+1,tflag]
                    xid_val    = LM_predictions[ii,30+3*pp+2,tflag]
                    
                Weight_predictions[ii,3*pp+0,tflag] = weight_val
                Weight_predictions[ii,3*pp+1,tflag] = xi_val
                Weight_predictions[ii,3*pp+2,tflag] = xid_val
                
                
                
                M10 = M10+weight_val*pow(xi_val,1)
                M01 = M01+weight_val*pow(xid_val,1)
                M20 = M20+weight_val*pow(xi_val,2)
                M11 = M11+weight_val*xi_val*xid_val
                M02 = M02+weight_val*pow(xid_val,2)
                
                M30 = M30 +weight_val*pow(xi_val,3)*pow(xid_val,0)
                M21 = M21 +weight_val*pow(xi_val,2)*pow(xid_val,1)
                M12 = M12 +weight_val*pow(xi_val,1)*pow(xid_val,2)
                M03 = M03 +weight_val*pow(xi_val,0)*pow(xid_val,3)
                
                M40 = M40 +weight_val*pow(xi_val,4)*pow(xid_val,0)
                M31 = M31 +weight_val*pow(xi_val,3)*pow(xid_val,1)
                M22 = M22 +weight_val*pow(xi_val,2)*pow(xid_val,2)
                M13 = M13 +weight_val*pow(xi_val,1)*pow(xid_val,3)
                M04 = M04 +weight_val*pow(xi_val,0)*pow(xid_val,4)
                
                M50 = M50 +weight_val*pow(xi_val,5)*pow(xid_val,0)
                M41 = M41 +weight_val*pow(xi_val,4)*pow(xid_val,1)
                M32 = M32 +weight_val*pow(xi_val,3)*pow(xid_val,2)
                M23 = M23 +weight_val*pow(xi_val,2)*pow(xid_val,3)
                M14 = M14 +weight_val*pow(xi_val,1)*pow(xid_val,4)
                M05 = M05 +weight_val*pow(xi_val,0)*pow(xid_val,5)
                
                
                M3g = M3g +weight_val*pow(xi_val,3.0*(1.0-1.4))*pow(xid_val,0)
                
                Mm11 = Mm11 +weight_val*pow(xi_val,-1)*pow(xid_val,1)
                Mm30 = Mm30 +weight_val*pow(xi_val,-3)*pow(xid_val,0)
                Mm40 = Mm40 +weight_val*pow(xi_val,-4)*pow(xid_val,0)
                Mm10 = Mm10 +weight_val*pow(xi_val,-1)*pow(xid_val,0)
                Mm12 = Mm12 +weight_val*pow(xi_val,-1)*pow(xid_val,2)
                Mm21 = Mm21 +weight_val*pow(xi_val,-2)*pow(xid_val,1)
                Mm13 = Mm13 +weight_val*pow(xi_val,-1)*pow(xid_val,3)
                Mm22 = Mm22 +weight_val*pow(xi_val,-2)*pow(xid_val,2)
                Mm41 = Mm41 +weight_val*pow(xi_val,-4)*pow(xid_val,1)
                
            LM_predictions[ii,0,tflag] = M10
            LM_predictions[ii,1,tflag] = M01
            LM_predictions[ii,2,tflag] = M20
            LM_predictions[ii,3,tflag] = M11
            LM_predictions[ii,4,tflag] = M02
            
            LM_predictions[ii,5,tflag] = M30
            LM_predictions[ii,6,tflag] = M21
            LM_predictions[ii,7,tflag] = M12
            LM_predictions[ii,8,tflag] = M03
            
            LM_predictions[ii, 9,tflag] = M40
            LM_predictions[ii,10,tflag] = M31
            LM_predictions[ii,11,tflag] = M22
            LM_predictions[ii,12,tflag] = M13
            LM_predictions[ii,13,tflag] = M04
            
            LM_predictions[ii,14,tflag] = M50
            LM_predictions[ii,15,tflag] = M41
            LM_predictions[ii,16,tflag] = M32
            LM_predictions[ii,17,tflag] = M23
            LM_predictions[ii,18,tflag] = M14
            LM_predictions[ii,19,tflag] = M05
            
            LM_predictions[ii,20,tflag] = M3g
            
                
            LM_RHS[ii,0,tflag] = M01
            LM_RHS[ii,1,tflag] = (-1.5*Mm12 -(4.0/Re)*Mm21 +Mm40 -LM_pressure[ii,tflag]*Mm10)
            LM_RHS[ii,2,tflag] = 2.0*M11
            LM_RHS[ii,3,tflag] = (-0.5*M02 -(4.0/Re)*Mm11 +Mm30 -LM_pressure[ii,tflag])
            LM_RHS[ii,4,tflag] = (-3.0*Mm13 -(8.0/Re)*Mm22 +2.0*Mm41 -2.0*LM_pressure[ii,tflag]*Mm11)
                
            if (tt > 16200):
                LM_predictions[ii,0,tt] = M10 +(dt/12.0)*(23.0*LM_RHS[ii,0,tflag]-16.0*LM_RHS[ii,0,tflag-1]+5.0*LM_RHS[ii,0,tflag-2])
                LM_predictions[ii,1,tt] = M01 +(dt/12.0)*(23.0*LM_RHS[ii,1,tflag]-16.0*LM_RHS[ii,1,tflag-1]+5.0*LM_RHS[ii,1,tflag-2])
                LM_predictions[ii,2,tt] = M20 +(dt/12.0)*(23.0*LM_RHS[ii,2,tflag]-16.0*LM_RHS[ii,2,tflag-1]+5.0*LM_RHS[ii,2,tflag-2])
                LM_predictions[ii,3,tt] = M11 +(dt/12.0)*(23.0*LM_RHS[ii,3,tflag]-16.0*LM_RHS[ii,3,tflag-1]+5.0*LM_RHS[ii,3,tflag-2])
                LM_predictions[ii,4,tt] = M02 +(dt/12.0)*(23.0*LM_RHS[ii,4,tflag]-16.0*LM_RHS[ii,4,tflag-1]+5.0*LM_RHS[ii,4,tflag-2])
                
#                LM_predictions[ii,0,tt] = M10 +(dt)*LM_RHS[ii,0,tflag]
#                LM_predictions[ii,1,tt] = M01 +(dt)*LM_RHS[ii,1,tflag]
#                LM_predictions[ii,2,tt] = M20 +(dt)*LM_RHS[ii,2,tflag]
#                LM_predictions[ii,3,tt] = M11 +(dt)*LM_RHS[ii,3,tflag]
#                LM_predictions[ii,4,tt] = M02 +(dt)*LM_RHS[ii,4,tflag]
                
#                sigmaR  = np.sqrt(LM_predictions[ii,2,tt] -pow(LM_predictions[ii,0,tt],2) )
#                sigmaRd = np.sqrt(np.max([0.0, LM_predictions[ii,4,tt] -pow(LM_predictions[ii,1,tt],2)]) )
#                val_flag = (LM_predictions[ii,3,tt] -LM_predictions[ii,0,tt]*LM_predictions[ii,1,tt])/np.sqrt(LM_predictions[ii,2,tt]-LM_predictions[ii,0,tt]*LM_predictions[ii,0,tt])
#                LM_predictions[ii,31-1+3*0+0,tt] = 0.25
#                LM_predictions[ii,31-1+3*0+1,tt] = LM_predictions[ii,0,tt] +sigmaR
#                LM_predictions[ii,31-1+3*0+2,tt] = LM_predictions[ii,1,tt]+val_flag +np.sqrt( LM_predictions[ii,4,tt] -val_flag*val_flag -pow(LM_predictions[ii,1,tt],2) )
#                LM_predictions[ii,31-1+3*1+0,tt] = 0.25
#                LM_predictions[ii,31-1+3*1+1,tt] = LM_predictions[ii,0,tt] +sigmaR
#                LM_predictions[ii,31-1+3*1+2,tt] = LM_predictions[ii,1,tt]+val_flag -np.sqrt( LM_predictions[ii,4,tt] -val_flag*val_flag -pow(LM_predictions[ii,1,tt],2) )
#                LM_predictions[ii,31-1+3*2+0,tt] = 0.25
#                LM_predictions[ii,31-1+3*2+1,tt] = LM_predictions[ii,0,tt] -sigmaR
#                LM_predictions[ii,31-1+3*2+2,tt] = LM_predictions[ii,1,tt]-val_flag +np.sqrt( LM_predictions[ii,4,tt] -val_flag*val_flag -pow(LM_predictions[ii,1,tt],2) )
#                LM_predictions[ii,31-1+3*3+0,tt] = 0.25
#                LM_predictions[ii,31-1+3*3+1,tt] = LM_predictions[ii,0,tt] -sigmaR
#                LM_predictions[ii,31-1+3*3+2,tt] = LM_predictions[ii,1,tt]-val_flag -np.sqrt( LM_predictions[ii,4,tt] -val_flag*val_flag -pow(LM_predictions[ii,1,tt],2) )
#                for kk in range(4,ml_dim//3):
#                    LM_predictions[ii,31-1+3*kk,tt]   = 0.00
#                    LM_predictions[ii,31-1+3*kk+1,tt]  = LM_predictions[ii,0,tt]
#                    LM_predictions[ii,31-1+3*kk+2,tt] = LM_predictions[ii,1,tt]



    
    return LM_predictions, Weight_predictions;    
























    
    














    