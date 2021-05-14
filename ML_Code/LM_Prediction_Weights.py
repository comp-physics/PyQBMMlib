#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:43:19 2021

@author: alexis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:16:40 2021

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

from Form_ML_Data import Load_LM_Data_Output_Weights
from Form_ML_Data import Load_Data_QBMM
from ML_Training  import LM_Training
from ML_Training  import LM_Testing


# Load MC data
cases = "sine";
#cases = "sine";
output_data, qbmm_LM, max_out, total_cases, total_times, T, input_data, max_in, weights, abscissas = Load_LM_Data_Output_Weights(cases)
# Load QBMM data
#input_data, max_in = Load_Data_QBMM(cases,T,total_times)


used_features = 5
# Train neural-net for High-moments predictions
if (cases == "constant"):
    train_cases = [0,1,3,5,7,9,11,13,15]
elif (cases == "sine"):
#    train_cases = [0,2,4,6,8,10,12,14,16,
#                   34+0,34+2,34+4,34+6,34+8,34+10,34+12,34+14,34+16,
#                   68+0,68+2,68+4,68+6,68+8,68+10,68+12,68+14,68+16,
#                   102+0,102+2,102+4,102+6,102+8,102+10,102+12,102+14,102+16]
    train_cases = [0,2,4,6,8,10,12,14,16]



#LM_Training(input_data,output_data,qbmm_LM,max_in,max_out,total_times,train_cases,cases,used_features)
#
#if (cases == "constant"):
#    model1 = tf.keras.models.load_model('LM_MLQBMM.h5')
#elif (cases == "sine"):
#    model1 = tf.keras.models.load_model('LM_Sinusoidal_MLQBMM.h5')
#
#test_cases = [ii for ii in range(0,119)]
#predictions = LM_Testing(input_data,output_data,qbmm_LM,max_in,max_out,total_times,used_features,test_cases,cases)
#if (cases == "constant"):
#    sio.savemat('LM_MLQBMM.mat',{'predictions':predictions})
#elif (cases == "sine"):
#    sio.savemat('LM_Sinusoidal_MLQBMM.mat',{'predictions':predictions,'input_data':input_data,'output_data':output_data})

