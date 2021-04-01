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

from Form_ML_Data import Load_Data_MC
from Form_ML_Data import Load_Data_QBMM
from ML_Training  import HM_Training
from ML_Training  import HM_Testing


# Load MC data
output_data, max_out, total_cases, total_times, T = Load_Data_MC()
# Load QBMM data
input_data, max_in = Load_Data_QBMM(T,total_times)


# Train neural-net for High-moments predictions
#HM_Training(input_data,output_data,max_in,max_out,total_times)


model1 = tf.keras.models.load_model('HM_MLQBMM.h5')


test_cases = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
used_features = 5

predictions = HM_Testing(input_data,output_data,max_in,max_out,total_times,used_features,test_cases)


sio.savemat('HM_MLQBMM.mat',{'predictions':predictions})


