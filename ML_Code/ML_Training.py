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


#def custom_loss(y_actual,y_pred):
#    custom_loss=kb.square(y_actual[:,:,0]-y_pred[:,:,0])+kb.square(y_actual[:,:,0]*y_actual[:,:,1]-y_pred[:,:,1])+kb.square(y_actual[:,:,0]*y_actual[:,:,2]-y_pred[:,:,2])
#    return custom_loss
#keras.losses.custom_loss = custom_loss



def custom_loss(y_actual,y_pred):
    
    total_abscissas = int(3)
    
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
    
    loss_cases = [0,
                  1,2,
                  3,4,5,
                  6,7,8,9]
    loss_size  = len(loss_cases)
    
    for kk in range(0,loss_size):
        kflag = loss_cases[kk]
        ymom = y_pred[:,:,0]*pow(y_pred[:,:,1],ids[0,kflag])*pow(y_pred[:,:,2],ids[1,kflag])
        for ii in range(1,total_abscissas):
            ymom = ymom +y_pred[:,:,3*ii]*pow(y_pred[:,:,3*ii+1],ids[0,kflag])*pow(y_pred[:,:,3*ii+2],ids[1,kflag])
        if (kk == 0):
            yloss = kb.square(ymom-y_actual[:,:,kflag])
        else:
            yloss = yloss +kb.square(ymom-y_actual[:,:,kflag])
    
    
    custom_loss=yloss
    #custom_loss2=kb.square(y_actual[:,:,0]-y_pred[:,:,0])+kb.square(y_actual[:,:,0]*y_actual[:,:,1]-y_pred[:,:,1])+kb.square(y_actual[:,:,0]*y_actual[:,:,2]-y_pred[:,:,2])
    return custom_loss
keras.losses.custom_loss = custom_loss






def HM_Training(input_data,output_data,qbmm_HM,max_in,max_out,total_times,test_cases,cases,used_features):
    
    time_start = 0
    time_history = 128
    time_jump = 1
    sub_sample = 1;
    train_times = total_times-time_history*time_jump-time_start
    train_times = int(np.floor(train_times//sub_sample))

    test_size = len(test_cases)
    #used_features = 5
    input_train = np.zeros((test_size*train_times,time_history,used_features))
    output_train  = np.zeros((test_size*train_times,time_history,4))

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,used_features):
                for pp in range(0,time_history):
                    input_train[ii*train_times+tt,pp,jj]  =  input_data[test_cases[ii],jj,sub_sample*tt+time_jump*pp+time_start]/max_in[test_cases[ii],jj]

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,4):
                for pp in range(0,time_history):
                    output_train[ii*train_times+tt,pp,jj]  =  output_data[test_cases[ii], jj,tt+time_jump*pp+time_start]/max_out[test_cases[ii],jj]

#    sub_sample = 256;
#    total_samples = int(np.floor(test_size*train_times//sub_sample))
#    input_final = np.zeros((total_samples,time_history,used_features))
#    output_final  = np.zeros((total_samples,time_history,4))
#    
#    for tt in range(0,total_samples):
#        input_final[tt,:,:] = input_train[tt*sub_sample,:,:]
#        output_final[tt,:,:] = output_train[tt*sub_sample,:,:]
    

    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.LSTM(input_shape=(None,used_features),units=time_history, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model1.add(tf.keras.layers.Dense(4))
    model1.add(tf.keras.layers.Dense(4, activation='linear'))
    model1.compile(loss='mse', optimizer='adam' )
    hist1 = model1.fit(input_train[:,:,:], output_train[:,:,:], batch_size=32, epochs=100, validation_data=(input_train[0:10,:,:],output_train[0:10,:,:]))
    if (cases == "constant"):
        model1.save('HM_MLQBMM.h5')
    elif (cases == "sine"):
        model1.save('HM_Sinusoidal_MLQBMM.h5')
    elif (cases == "random"):
        model1.save('HM_Random_MLQBMM.h5')
    
    return;
    



def LM_Training(input_data,output_data,qbmm_HM,max_in,max_out,total_times,test_cases,cases,approach,used_features,output_dim,ml_dim):
    
    time_start = 0
    time_history = 128
    time_jump = 1
    sub_sample = 1;
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

    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.LSTM(input_shape=(None,used_features),units=time_history, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    #model1.add(tf.keras.layers.Dense(ml_dim))
    model1.add(tf.keras.layers.Dense(ml_dim, activation='linear'))
    if (approach =="1"):
        model1.compile(loss='mse', optimizer='adam' )
    elif (approach == "2"):
        model1.compile(loss=custom_loss, optimizer='adam' )
    elif (approach == "3"):
        model1.compile(loss='mse', optimizer='adam' )
    elif (approach == "4"):
        model1.compile(loss=custom_loss, optimizer='adam' )
    hist1 = model1.fit(input_train[0:train_end,:,:], output_train[0:train_end,:,:], batch_size=32, epochs=20, validation_data=(input_train[train_end:test_size*train_times,:,:],output_train[train_end:test_size*train_times,:,:]))
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
    

























def RHSM_Training(input_data,output_data,qbmm_HM,max_in,max_out,total_times,test_cases,cases,used_features):
    
    time_start = 0
    time_history = 128
    time_jump = 1
    sub_sample = 1;
    train_times = total_times-time_history*time_jump-time_start
    train_times = int(np.floor(train_times//sub_sample))

    test_size = len(test_cases)
    #used_features = 5
    input_train = np.zeros((test_size*train_times,time_history,used_features))
    output_train  = np.zeros((test_size*train_times,time_history,9))

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,used_features):
                for pp in range(0,time_history):
                    input_train[ii*train_times+tt,pp,jj]  =  input_data[test_cases[ii],jj,sub_sample*tt+time_jump*pp+time_start]/max_in[test_cases[ii],jj]

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,9):
                for pp in range(0,time_history):
                    output_train[ii*train_times+tt,pp,jj]  =  output_data[test_cases[ii], jj,tt+time_jump*pp+time_start]/max_out[test_cases[ii],jj]

    

    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.LSTM(input_shape=(None,used_features),units=time_history, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model1.add(tf.keras.layers.Dense(5))
    model1.add(tf.keras.layers.Dense(9, activation='linear'))
    model1.compile(loss='mse', optimizer='adam' )
    hist1 = model1.fit(input_train[:,:,:], output_train[:,:,:], batch_size=32, epochs=50, validation_data=(input_train[0:10,:,:],output_train[0:10,:,:]))
    if (cases == "constant"):
        model1.save('RHSM_MLQBMM.h5')
    elif (cases == "sine"):
        model1.save('RHSM_Sinusoidal_MLQBMM.h5')
    elif (cases == "random"):
        model1.save('RHSM_Random_MLQBMM.h5')
    
    return;
    






def ThreeM_Training(input_data,output_data,qbmm_HM,max_in,max_out,total_times,test_cases,cases,used_features):
    
    time_start = 0
    time_history = 128
    time_jump = 1
    sub_sample = 1;
    train_times = total_times-time_history*time_jump-time_start
    train_times = int(np.floor(train_times//sub_sample))

    test_size = len(test_cases)
    #used_features = 5
    input_train = np.zeros((test_size*train_times,time_history,used_features))
    output_train  = np.zeros((test_size*train_times,time_history,4))

    for ii in range(0,test_size):
        for tt in range(0,train_times):
            for jj in range(0,used_features):
                for pp in range(0,time_history):
                    input_train[ii*train_times+tt,pp,jj]  =  input_data[test_cases[ii],jj,sub_sample*tt+time_jump*pp+time_start]/max_in[test_cases[ii],jj]

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
    hist1 = model1.fit(input_train[0:(test_size-2)*train_times,:,:], output_train[0:(test_size-2)*train_times,:,:], batch_size=32, epochs=10, validation_data=(input_train[(test_size-2)*train_times:(test_size)*train_times,:,:],output_train[(test_size-2)*train_times:(test_size)*train_times,:,:]))
    if (cases == "constant"):
        model1.save('3M_MLQBMM.h5')
    elif (cases == "sine"):
        model1.save('3M_Sinusoidal_MLQBMM.h5')
    elif (cases == "random"):
        model1.save('3M_Random_MLQBMM.h5')
    
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
    




    




def LM_Testing_Approach2(input_data,output_data,qbmm_LM,max_in,max_out,total_times,used_features,test_cases,cases,approach,output_dim,ml_dim,LM_QBMM,LM_MC, LM_pressure,lm_times,input_size):

    if (cases == "constant"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Constant_MLQBMM_Approach4.h5')
    elif (cases == "sine"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Sinusoidal_MLQBMM_Approach4.h5')
    elif (cases == "random"):
        model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach'+approach+'_Weights'+str(ml_dim//3)+'_Pressure.h5',compile=False)
        #model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach1.h5', custom_objects={'custom_loss': custom_loss})

    test_size = len(test_cases)
    
    scale_times = (lm_times-1)//(total_times-1)
    input_test   = np.zeros((scale_times*test_size,total_times,used_features))
    output_test  = np.zeros((scale_times*test_size,total_times,output_dim))

                
    for ii in range(0,test_size):
        for kk in range(0,scale_times):
            for pp in range(0,total_times-1):
                for jj in range(0,input_size-1):
                    input_test[scale_times*ii+kk,pp,jj] = LM_MC[test_cases[ii],jj,scale_times*pp+kk]/max_in[test_cases[ii],jj]
                input_test[scale_times*ii+kk,pp,input_size-1] = LM_pressure[test_cases[ii],scale_times*pp+kk]/max_in[test_cases[ii],input_size-1]
        for jj in range(0,input_size-1):
            input_test[scale_times*ii,total_times-1,jj] = LM_MC[test_cases[ii],jj,lm_times-1]/max_in[test_cases[ii],jj]
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


    LM_RHS = np.zeros((len(test_cases),5,lm_times))
    abscissas = ml_dim//3
    LM_predictions = np.zeros((test_size,5,lm_times))
    time_hist = 16
    input_flag = np.zeros((1,time_hist,used_features))
    dt = 0.01
    Re = 1000.0
    for ii in range(0,test_size):
        print(ii)
        for tt in range(0,lm_times):
            M10 = 0.0
            M01 = 0.0
            M20 = 0.0
            M11 = 0.0
            M02 = 0.0
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
            tflag2 = tt-2
            
#            if (tt > 2000):
#                for pp in range(0,time_hist):
#                    for jj in range(0,5):
#                        input_flag[0,pp,jj] = LM_predictions[ii,jj,tt-(time_hist-1-pp)*scale_times-1]/max_in[ii,jj]
#                    input_flag[0,pp,5] = LM_pressure[test_cases[ii],scale_times*pp+kk]/max_in[ii,5]
#                output_flag = model1.predict(input_flag)
#                for jj in range(0,ml_dim):
#                    predictions[ii,jj,tt-1] = output_flag[0,time_hist-1,jj]
            
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
                
#                M10 = M10+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+1,tflag]
#                M01 = M01+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]
#                M20 = M20+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+1,tflag]*predictions[ii,3*pp+1,tflag]
#                M11 = M11+predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],1.0)*predictions[ii,3*pp+2,tflag]
#                M02 = M02+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]*predictions[ii,3*pp+2,tflag]
#                Mm11 = Mm11 +predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]/predictions[ii,3*pp+1,tflag]
#                Mm30 = Mm30 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-3)
#                Mm40 = Mm40 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-4)
#                Mm10 = Mm10 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-1)
#                Mm12 = Mm12 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],2)/predictions[ii,3*pp+1,tflag]
#                Mm21 = Mm21 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],1)/pow(predictions[ii,3*pp+1,tflag],2)
#                Mm13 = Mm13 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],3)/pow(predictions[ii,3*pp+1,tflag],1)
#                Mm22 = Mm22 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],2)/pow(predictions[ii,3*pp+1,tflag],2)
#                Mm41 = Mm41 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],1)/pow(predictions[ii,3*pp+1,tflag],4)
                
                LM_predictions[ii,0,tt] = LM_predictions[ii,0,tt] +predictions[ii,3*pp,tt]*pow(predictions[ii,3*pp+1,tt],1.0)
                LM_predictions[ii,1,tt] = LM_predictions[ii,1,tt] +predictions[ii,3*pp,tt]*predictions[ii,3*pp+2,tt]
                LM_predictions[ii,2,tt] = LM_predictions[ii,2,tt] +predictions[ii,3*pp,tt]*pow(predictions[ii,3*pp+1,tt],2.0)
                LM_predictions[ii,3,tt] = LM_predictions[ii,3,tt] +predictions[ii,3*pp,tt]*pow(predictions[ii,3*pp+1,tt],1.0)*predictions[ii,3*pp+2,tt]
                LM_predictions[ii,4,tt] = LM_predictions[ii,4,tt] +predictions[ii,3*pp,tt]*predictions[ii,3*pp+2,tt]*predictions[ii,3*pp+2,tt]
                
#            LM_RHS[ii,0,tflag] = M01
#            LM_RHS[ii,1,tflag] = (-1.5*Mm12 -(4.0/Re)*Mm21 +Mm40 -LM_pressure[ii,tflag]*Mm10)
#            LM_RHS[ii,2,tflag] = 2.0*M11
#            LM_RHS[ii,3,tflag] = (-0.5*M02 -(4.0/Re)*Mm11 +Mm30 -LM_pressure[ii,tflag])
#            LM_RHS[ii,4,tflag] = (-3.0*Mm13 -(8.0/Re)*Mm22 +2.0*Mm41 -2.0*LM_pressure[ii,tflag]*Mm11)
                

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
                    input_test[scale_times*ii+kk,pp,jj] = LM_MC[test_cases[ii],jj,scale_times*pp+kk]/max_in[test_cases[ii],jj]
                input_test[scale_times*ii+kk,pp,input_size-1] = LM_pressure[test_cases[ii],scale_times*pp+kk]/max_in[test_cases[ii],input_size-1]
        for jj in range(0,input_size-1):
            input_test[scale_times*ii,total_times-1,jj] = LM_MC[test_cases[ii],jj,lm_times-1]/max_in[test_cases[ii],jj]
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


    LM_RHS = np.zeros((len(test_cases),5,lm_times))
    abscissas = ml_dim//3
    LM_predictions = np.zeros((test_size,5,lm_times))
    LM_predictions[:,:,:] = LM_MC[:,0:5,:]
    time_hist = 128
    input_flag = np.zeros((1,time_hist,used_features))
    dt = 0.10
    Re = 1000.0
    for ii in range(20,30):
    #for ii in range(0,test_size):
        print(ii)
        for tt in range(0,lm_times):
            M10 = 0.0
            M01 = 0.0
            M20 = 0.0
            M11 = 0.0
            M02 = 0.0
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
            
            if (tt > 1900):
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
                
                M10 = M10+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+1,tflag]
                M01 = M01+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]
                M20 = M20+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+1,tflag]*predictions[ii,3*pp+1,tflag]
                M11 = M11+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+1,tflag]*predictions[ii,3*pp+2,tflag]
                M02 = M02+predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]*predictions[ii,3*pp+2,tflag]
                Mm11 = Mm11 +predictions[ii,3*pp,tflag]*predictions[ii,3*pp+2,tflag]/predictions[ii,3*pp+1,tflag]
                Mm30 = Mm30 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-3)
                Mm40 = Mm40 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-4)
                Mm10 = Mm10 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+1,tflag],-1)
                Mm12 = Mm12 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],2)/predictions[ii,3*pp+1,tflag]
                Mm21 = Mm21 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],1)/pow(predictions[ii,3*pp+1,tflag],2)
                Mm13 = Mm13 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],3)/pow(predictions[ii,3*pp+1,tflag],1)
                Mm22 = Mm22 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],2)/pow(predictions[ii,3*pp+1,tflag],2)
                Mm41 = Mm41 +predictions[ii,3*pp,tflag]*pow(predictions[ii,3*pp+2,tflag],1)/pow(predictions[ii,3*pp+1,tflag],4)
                
            LM_predictions[ii,0,tflag] = M10
            LM_predictions[ii,1,tflag] = M01
            LM_predictions[ii,2,tflag] = M20
            LM_predictions[ii,3,tflag] = M11
            LM_predictions[ii,4,tflag] = M02
                
            LM_RHS[ii,0,tflag] = M01
            LM_RHS[ii,1,tflag] = (-1.5*Mm12 -(4.0/Re)*Mm21 +Mm40 -LM_pressure[ii,tflag]*Mm10)
            LM_RHS[ii,2,tflag] = 2.0*M11
            LM_RHS[ii,3,tflag] = (-0.5*M02 -(4.0/Re)*Mm11 +Mm30 -LM_pressure[ii,tflag])
            LM_RHS[ii,4,tflag] = (-3.0*Mm13 -(8.0/Re)*Mm22 +2.0*Mm41 -2.0*LM_pressure[ii,tflag]*Mm11)
                
            if (tt > 1900):
                LM_predictions[ii,0,tt] = M10 +(dt/12.0)*(23.0*LM_RHS[ii,0,tflag]-16.0*LM_RHS[ii,0,tflag-1]+5.0*LM_RHS[ii,0,tflag-2])
                LM_predictions[ii,1,tt] = M01 +(dt/12.0)*(23.0*LM_RHS[ii,1,tflag]-16.0*LM_RHS[ii,1,tflag-1]+5.0*LM_RHS[ii,1,tflag-2])
                LM_predictions[ii,2,tt] = M20 +(dt/12.0)*(23.0*LM_RHS[ii,2,tflag]-16.0*LM_RHS[ii,2,tflag-1]+5.0*LM_RHS[ii,2,tflag-2])
                LM_predictions[ii,3,tt] = M11 +(dt/12.0)*(23.0*LM_RHS[ii,3,tflag]-16.0*LM_RHS[ii,3,tflag-1]+5.0*LM_RHS[ii,3,tflag-2])
                LM_predictions[ii,4,tt] = M02 +(dt/12.0)*(23.0*LM_RHS[ii,4,tflag]-16.0*LM_RHS[ii,4,tflag-1]+5.0*LM_RHS[ii,4,tflag-2])
                
#    for ii in range(0,test_size):
#        for kk in range(0,scale_times):
#            for tt in range(0,total_times-1):
#                LM_predictions[ii,0,scale_times*tt+kk] = predictions[scale_times*ii+kk,tt,jj]*max_out[test_cases[ii],jj]+1.0*LM_QBMM[test_cases[ii],jj,scale_times*tt+kk]
#                for jj in range(0,2):
#                    LM_predictions[ii,jj,scale_times*tt+kk] = output_test[scale_times*ii+kk,tt,jj]*max_out[test_cases[ii],jj]+1.0*LM_QBMM[test_cases[ii],jj,scale_times*tt+kk]
#        for jj in range(0,5):
#            LM_predictions[ii,jj,lm_times-1] = output_test[scale_times*ii,total_times-1,jj]*max_out[test_cases[ii],jj]+1.0*LM_QBMM[test_cases[ii],jj,lm_times-1]
    
    return LM_predictions;    
























    
    














    