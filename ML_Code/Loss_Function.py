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
        for ii in range(0,npoints):
            yloss = yloss +100.0*kb.relu(-(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii]))   
            
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
        loss_coef = alpha.mom_scale_coeffs
        
        
        counter = int(-1)
        for kflag in loss_mom:
            counter = counter+1
            ymom = (y_actual[:,:,32]+y_pred[:,:,0])*pow(y_actual[:,:,33]+y_pred[:,:,1],ids[0,kflag])*pow(y_actual[:,:,34]+y_pred[:,:,2],ids[1,kflag])
            for ii in range(1,npoints):
                ymom = ymom +(y_actual[:,:,32+3*ii]+y_pred[:,:,3*ii])*pow(y_actual[:,:,32+3*ii+1]+y_pred[:,:,3*ii+1],ids[0,kflag])*pow(y_actual[:,:,32+3*ii+2]+y_pred[:,:,3*ii+2],ids[1,kflag])
            if (kflag == 5):
                yloss = yloss +10.0*kb.square(ymom-y_actual[:,:,kflag])/pow(loss_coef[0,kflag-1],2)
            else:
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
 