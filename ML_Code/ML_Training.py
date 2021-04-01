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


def HM_Training(input_data,output_data,max_in,max_out,total_times):
    
    time_start = 0
    time_history = 32
    time_jump = 1
    train_times = total_times-time_history*time_jump-time_start

    test_cases = [0,2,4,6,8,10,12,14]
    test_size = len(test_cases)
    used_features = 5
    input_train = np.zeros((test_size*train_times,time_history,used_features))
    output_train  = np.zeros((test_size*train_times,time_history,4))

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,used_features):
                for pp in range(0,time_history):
                    input_train[ii*train_times+tt,pp,jj]  =  input_data[test_cases[ii],jj,tt+time_jump*pp+time_start]/max_in[test_cases[ii],jj]

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,4):
                for pp in range(0,time_history):
                    output_train[ii*train_times+tt,pp,jj]  =  output_data[test_cases[ii], jj,tt+time_jump*pp+time_start]/max_out[test_cases[ii],jj]


    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.LSTM(input_shape=(None,used_features),units=time_history, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model1.add(tf.keras.layers.Dense(4))
    model1.add(tf.keras.layers.Dense(4, activation='linear'))
    model1.compile(loss='mse', optimizer='adam' )
    hist1 = model1.fit(input_train[:,:,:], output_train[:,:,:], batch_size=32, epochs=100, validation_data=(input_train[0:10,:,:],output_train[0:10,:,:]))
    model1.save('HM_MLQBMM.h5')
    
    return;
    


def HM_Testing(input_data,output_data,max_in,max_out,total_times,used_features,test_cases):

    model1 = tf.keras.models.load_model('HM_MLQBMM.h5')

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
                predictions[ii,jj,tt] = output_test[ii,tt,jj]*max_out[test_cases[ii],jj]
    
    return predictions;    
    
    
    
    