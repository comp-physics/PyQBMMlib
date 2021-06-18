#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:57:04 2021

@author: alexis
"""

import numpy as np
print('Importing scipy... ', end='', flush=True)
import scipy.io as sio
from scipy import interpolate
print('Done!')

print('Importing Tensorflow... ', end='', flush=True)
import tensorflow as tf
print('Done!')
print('Importing Keras... ', end='', flush=True)
from tensorflow import keras 
print('Done!')


from loss_params import loss_params
from Loss_Function import param_loss





class ML_Config:
    def __init__(self, approach, cases, method):
        self.approach = approach
        self.cases    = cases
        self.method   = method
        
        if (self.cases == "random"):
            self.folder_name = '../data/Random_Forcing/'
            self.mc_file_format = 'MC_HM_Random_Pressure_Realization';
            self.qbmm_HM_file_format = 'QBMM_HM_Random_Pressure_Realization';
            self.MC_file_names = []
            self.QBMM_file_names = []
            self.MC_file_names   = [self.folder_name+self.mc_file_format+str(ii)+'.mat' for ii in range(1,5,1)]
            self.QBMM_file_names = [self.folder_name+self.qbmm_HM_file_format+str(ii)+'.mat' for ii in range(1,5,1)]
            
        elif (self.cases == "constant"):
            self.folder_name = '../data/Constant_Forcing/';
            self.mc_file_format = 'MC_HM_Constant_Pressure';
            self.qbmm_HM_file_format = 'QBMM_HM_Constant_Pressure';
            self.MC_file_names = []
            self.MC_file_names = [self.folder_name+self.mc_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
            self.QBMM_file_names = []
            self.QBMM_file_names = [self.folder_name+self.qbmm_HM_file_format+str(ii)+'.mat' for ii in range(20,100,5)]
            
        
        return
    
    
    def ml_data(self,abscissas):
        if (self.approach == "4"):
            self.ml_data4(abscissas)
            
        return
    
    
    
    def ML_CHyQMOM(self, LM_predictions, predictions):
        
        #Weight_predictions = np.zeros(self.ml_dim,dtype=float)
        LM_predictions[0:30] = 0.0
        for pp in range(0,self.abscissas):
            weight_val = LM_predictions[30+3*pp]  +predictions[3*pp]
            xi_val     = LM_predictions[30+3*pp+1]+predictions[3*pp+1]
            xid_val    = LM_predictions[30+3*pp+2]+predictions[3*pp+2]
            
            LM_predictions[30+3*pp] = weight_val
            LM_predictions[30+3*pp+1] = xi_val
            LM_predictions[30+3*pp+2] = xid_val
            
            for kk in range(0,30):
                LM_predictions[kk] = LM_predictions[kk] +weight_val*pow(xi_val,self.ids[0,kk+1])*pow(xid_val,self.ids[1,kk+1])        
        
        return
    
    
    def comp_abcissas(self, input_data, total_times):
        
        output_data = np.zeros((self.ml_dim, total_times), dtype=float)
        for tt in range(0,total_times):
            val_flag = (input_data[3,tt] -input_data[0,tt]*input_data[1,tt])/np.sqrt(input_data[2,tt]-input_data[0,tt]*input_data[0,tt])
            output_data[3*0+0,tt] = 0.25
            output_data[3*0+1,tt] = input_data[0,tt] +np.sqrt( input_data[2,tt] -pow(input_data[0,tt],2) )
            output_data[3*0+2,tt] = input_data[1,tt]+val_flag +np.sqrt( input_data[4,tt] -val_flag*val_flag -pow(input_data[1,tt],2) )
            output_data[3*1+0,tt] = 0.25
            output_data[3*1+1,tt] = input_data[0,tt] +np.sqrt( input_data[2,tt] -pow(input_data[0,tt],2) )
            output_data[3*1+2,tt] = input_data[1,tt]+val_flag -np.sqrt( input_data[4,tt] -val_flag*val_flag -pow(input_data[1,tt],2) )
            output_data[3*2+0,tt] = 0.25
            output_data[3*2+1,tt] = input_data[0,tt] -np.sqrt( input_data[2,tt] -pow(input_data[0,tt],2) )
            output_data[3*2+2,tt] = input_data[1,tt]-val_flag +np.sqrt( input_data[4,tt] -val_flag*val_flag -pow(input_data[1,tt],2) )
            output_data[3*3+0,tt] = 0.25
            output_data[3*3+1,tt] = input_data[0,tt] -np.sqrt( input_data[2,tt] -pow(input_data[0,tt],2) )
            output_data[3*3+2,tt] = input_data[1,tt]-val_flag -np.sqrt( input_data[4,tt] -val_flag*val_flag -pow(input_data[1,tt],2) )
            for ii in range(4,self.abscissas):
                output_data[3*ii,tt]   = 0.00
                output_data[3*ii+1,tt] = input_data[0,tt]
                output_data[3*ii+2,tt] = input_data[1,tt]
        
        return output_data
    
    
    
    def ml_data4(self, abscissas):
        
        self.input_vals  = [1,2,3,4,5] #mu10, mu01, mu20, mu11, mu02
        self.output_vals = [0,
                            1,2,
                            3,4,5,
                            6,7,8,9,
                            10,11,12,13,14,
                            15,16,17,18,19,20,
                            21,
                            22,23,24,25,26,27,28,29,30] #mu00, mu10, mu01
        self.abscissas   = abscissas
        self.ml_dim     = int(3*self.abscissas)
        self.input_size = np.size(self.input_vals)+1
        self.output_dim = np.size(self.output_vals)+1+self.ml_dim
        
        self.mom_scale_coeffs = np.zeros((1,30),dtype=float)
        self.rhs_scale_coeffs = np.zeros((1,30),dtype=float)
        scale_flag = 0.0
        
        
        self.used_features = self.input_size
        self.total_cases = len(self.MC_file_names)
        for kk in range(0,self.total_cases):
            mat_data = sio.loadmat(self.MC_file_names[kk])
            moments  = mat_data['moments']
            pressure = mat_data['pressure']
            if (kk == 0):
                self.max_out = np.zeros((self.total_cases,self.output_dim))
                self.max_in  = np.zeros((self.total_cases,self.input_size))
                self.total_times = 30001
                self.lm_times = 30001
                self.QBMM_Data = np.zeros((self.total_cases,30+self.ml_dim,self.lm_times))
                self.MC_Data   = np.zeros((self.total_cases,30+self.ml_dim,self.lm_times))
                self.Pressure = np.zeros((self.total_cases,self.lm_times))
                self.output_data = np.zeros((self.total_cases,self.output_dim,self.total_times))
                self.input_data = np.zeros((self.total_cases,self.input_size,self.total_times),dtype=float)
                qbmm_moments = np.zeros((self.total_cases,self.input_size,self.total_times),dtype=float)
                T_old = mat_data['T']
                self.dt = 0.01
                self.T = [ self.dt*ii for ii in range(0,self.total_times) ]
                self.T_MC = [ self.dt*ii for ii in range(0,self.lm_times) ]
        
            for ii in range(0,self.output_dim-1-self.ml_dim):
                self.output_data[kk,ii,:] = moments[self.output_vals[ii],0:self.lm_times:1]
                
                
            mat_data = sio.loadmat(self.QBMM_file_names[kk])
            T_qbmm = mat_data['T']
            qbmm_moments  = mat_data['moments']
            
            for ii in range(0,self.input_size-1):
                f = interpolate.InterpolatedUnivariateSpline(T_old,moments[self.input_vals[ii],:],k=2)
                #f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,self.input_vals[ii]],k=2)
                self.input_data[kk,ii,:] = f(self.T)
                    
            f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
            self.input_data[kk,self.input_size-1,:] = f(self.T)
            self.output_data[kk,self.output_dim-1-self.ml_dim,:] = f(self.T)
            self.MC_Data[kk,self.output_dim-1-self.ml_dim,:] = f(self.T)
            self.QBMM_Data[kk,self.output_dim-1-self.ml_dim,:] = f(self.T)
                
            for ii in range(0,30):
                f = interpolate.InterpolatedUnivariateSpline(self.T,moments[1+ii,:],k=2)
                self.MC_Data[kk,ii,:] = f(self.T_MC)
                f = interpolate.InterpolatedUnivariateSpline(T_qbmm,qbmm_moments[:,1+ii],k=2)
                self.QBMM_Data[kk,ii,:] = f(self.T_MC)
            f = interpolate.InterpolatedUnivariateSpline(T_old,pressure,k=2)
            self.Pressure[kk,:] = f(self.T_MC)
        
            for ii in range(0,self.input_size):
                self.max_in[kk,ii] = 1.0
            for ii in range(0,self.output_dim):
                self.max_out[kk,ii] = 1.0
            for ii in range(0,30):
                scale_flag = np.max(self.MC_Data[kk,ii,:])
                if (scale_flag > self.mom_scale_coeffs[0,ii]):
                    self.mom_scale_coeffs[0,ii] = scale_flag
            for ii in range(0,30):
                for tt in range(0,self.lm_times-1):
                    scale_flag = (self.MC_Data[kk,ii,tt+1]-self.MC_Data[kk,ii,tt])/self.dt
                    self.rhs_scale_coeffs[0,ii] = np.max([self.rhs_scale_coeffs[0,ii],scale_flag])
                
            self.output_data[kk,np.size(self.output_vals)+1:self.output_dim,:] = self.comp_abcissas(self.input_data[kk,0:5,:], self.total_times)
            self.MC_Data[kk,np.size(self.output_vals)-1:np.size(self.output_vals)-1+self.ml_dim,:] = self.comp_abcissas(self.MC_Data[kk,0:5,:], self.lm_times)
            self.QBMM_Data[kk,np.size(self.output_vals)-1:np.size(self.output_vals)-1+self.ml_dim,:] = self.comp_abcissas(self.QBMM_Data[kk,0:5,:], self.lm_times)

            
        for ii in range(0,self.input_size):        
            self.max_in[:,ii] = np.max(self.max_in[:,ii])
        
        for ii in range(0,self.output_dim):
            self.max_out[:,ii] = np.max(self.max_out[:,ii])
        
        
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
        self.ids = ids
        
        self.Re = 1000.0
        if (self.cases == "random"):
            self.model_name = 'Neural_Nets/LM_Random_MLQBMM_Approach'+self.approach+'_Weights'+str(self.abscissas)+'.h5'
        
        
        return
    
    def training_init(self):
        
        self.time_start   = 0000
        self.time_history = 256
        self.hidden_units = 10
        self.time_jump    = 1
        self.sub_sample   = 32
        self.learning_rate = 1.0*pow(10.0,-7)
        self.batch_size = 1
        self.epochs = 10
        
        self.sigmaR_tol  = pow(10,-4)
        self.sigmaRd_tol = pow(10,-4)
        
        return
    
    
    def ml_training(self,train_cases):
        
        self.training_init()
        
        self.train_cases  = train_cases
        
        self.train_times  = self.total_times-self.time_history*self.time_jump-self.time_start
        self.train_times  = int(np.floor(self.train_times//self.sub_sample))

        self.train_size = len(self.train_cases)
        input_train     = np.zeros((self.train_size*self.train_times,self.time_history,self.used_features))
        output_train    = np.zeros((self.train_size*self.train_times,self.time_history,self.output_dim))
        #output_train    = np.zeros((self.train_size*self.train_times,1,self.output_dim))

        for ii in range(0,self.train_size):
            for tt in range(0,self.train_times):
                for jj in range(0,self.used_features):
                    for pp in range(0,self.time_history):
                        input_train[ii*self.train_times+tt,pp,jj]  =  self.input_data[self.train_cases[ii],jj,self.sub_sample*tt+self.time_jump*pp+self.time_start]/self.max_in[self.train_cases[ii],jj]

        for ii in range(0,self.train_size):
            for tt in range(0,self.train_times):
                for jj in range(0,self.output_dim):
                    for pp in range(0,self.time_history):
                        output_train[ii*self.train_times+tt,pp,jj]  =  self.output_data[self.train_cases[ii], jj,tt+self.time_jump*pp+self.time_start]/self.max_out[self.train_cases[ii],jj]

    
        self.train_end = round(self.train_size*self.train_times*0.9)
        
        loss_data = loss_params(self.Re,input_train[:,:,5],self.ids,self.abscissas, self.mom_scale_coeffs, self.rhs_scale_coeffs)
        
        inputs1 = tf.keras.layers.Input(shape=(None,self.used_features),batch_size=self.batch_size)
        lstm, hidden, cell = tf.keras.layers.LSTM(units=self.hidden_units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=True, go_backwards=False, stateful=True, unroll=False)(inputs1)
        dense = tf.keras.layers.Dense(self.ml_dim, activation='linear')(lstm)
        model1 = tf.keras.models.Model(inputs=inputs1, outputs=dense)
        model2 = tf.keras.models.Model(inputs=inputs1, outputs=[dense,hidden,cell])
        model1.compile(loss=param_loss(alpha=loss_data), optimizer='Adam' )
        #tf.keras.backend.set_value(model1.optimizer.learning_rate, self.learning_rate)
        hist1 = model1.fit(input_train[0:self.train_end,:,:], output_train[0:self.train_end,:,:], batch_size = self.batch_size, epochs=self.epochs, validation_data=(input_train[self.train_end:self.train_size*self.train_times,:,:],output_train[self.train_end:self.train_size*self.train_times,:,:]))
        
        
#        model1 = tf.keras.Sequential()
#        model1.add(tf.keras.layers.LSTM(input_shape=(None,self.used_features),units=self.hidden_units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))
#        model1.add(tf.keras.layers.Dense(self.ml_dim, activation='linear'))
#        if (self.approach == "4"):
#            tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
#            model1.compile(loss=param_loss(alpha=loss_data), optimizer='Adam' )
#        hist1 = model1.fit(input_train[0:self.train_end,:,:], output_train[0:self.train_end,:,:], batch_size = self.batch_size, epochs=self.epochs, validation_data=(input_train[self.train_end:self.train_size*self.train_times,:,:],output_train[self.train_end:self.train_size*self.train_times,:,:]))
        
        
#        self.input_w  = model1.layers[0].get_weights()[0]
#        self.hidden_w = model1.layers[0].get_weights()[1]
#        self.bias_w   = model1.layers[0].get_weights()[2]
#        self.hidden_w[5,5] = 1.0
#        weights_flag = self.input_w, self.hidden_w, self.bias_w
#        model1.layers[0].set_weights(weights_flag)
#        print(self.input_w.shape)
#        print(self.hidden_w.shape)
#        print(self.bias_w.shape)

        
        #model1.save_weights("ML_Predictions/snitch.h5")
        #self.states = model1.layers[0].weights
        #sio.savemat('ML_Predictions/Snitch.mat',{'state_c':model1.layers[0].get_weights(),'hist1':hist1})
        
        if (self.cases == "random"):
            self.model_name = 'Neural_Nets/LM_Random_MLQBMM_Approach'+self.approach+'_Weights'+str(self.abscissas)+'.h5'
            model1.save(self.model_name)
            #self.model_name = 'Neural_Nets/LM_Random_MLQBMM_Approach'+self.approach+'_Weights'+str(self.abscissas)+'.h5'
            #model2.save(self.model_name)
        
        
        return
    
    
    
    def RHS_Eval(self, LM_predictions, Pressure):
        
        LM_RHS = np.zeros(5,dtype=float)
        LM_RHS[0] = LM_predictions[1]
        LM_RHS[1] = (-1.5*LM_predictions[21] -(4.0/self.Re)*LM_predictions[22] +LM_predictions[23] -Pressure*LM_predictions[24])
        LM_RHS[2] = 2.0*LM_predictions[3]
        LM_RHS[3] = (-0.5*LM_predictions[4] -(4.0/self.Re)*LM_predictions[25] +LM_predictions[26] -Pressure)
        LM_RHS[4] = (-3.0*LM_predictions[27] -(8.0/self.Re)*LM_predictions[28] +2.0*LM_predictions[29] -2.0*Pressure*LM_predictions[25])
        
        return LM_RHS;
    
    def ML_Pred(self, ii, tt):
        
        input_flag = np.zeros((1,1,self.input_size),dtype=float)
        for jj in range(0,5):
            input_flag[0,0,jj] = self.LM_predictions[ii,jj,tt]/self.max_in[ii,jj]
        input_flag[0,0,5] = self.Pressure[ii,tt]/self.max_in[ii,5]
        output_flag = self.model1.predict(input_flag)
        
        for jj in range(0,self.ml_dim):
            self.predictions[ii,jj,tt] = output_flag[0,0,jj] 
        
#        input_flag = np.zeros((1,self.time_history,self.input_size),dtype=float)
#        for pp in range(0,self.time_history):
#            for jj in range(0,5):
#                input_flag[0,pp,jj] = self.LM_predictions[ii,jj,tt-(self.time_history-1-pp)*self.time_jump]/self.max_in[ii,jj]
#            input_flag[0,pp,5] = self.Pressure[ii,tt-(self.time_history-1-pp)*self.time_jump]/self.max_in[ii,5]
#        
#        output_flag = self.model1.predict(input_flag)
#        for jj in range(0,self.ml_dim):
#            self.predictions[ii,jj,tt] = output_flag[0,self.time_history-1,jj]        
        
        return
    
    def CHyQMOM_Pred(self, LM_predictions):
        
        QMOM = np.zeros(self.ml_dim,dtype=float)
        
        
        if ( LM_predictions[2] -pow(LM_predictions[0],2) < self.sigmaR_tol ):
            sigmaR = np.sqrt(self.sigmaR_tol)
        else:
            sigmaR   = np.sqrt(LM_predictions[2] -pow(LM_predictions[0],2) )
                
        val_flag = (LM_predictions[3] -LM_predictions[0]*LM_predictions[1])/sigmaR
        
        if ( LM_predictions[4] -val_flag*val_flag -pow(LM_predictions[1],2) < self.sigmaRd_tol ):
            sigmaRd  = np.sqrt( self.sigmaRd_tol )
        else:
            sigmaRd  = np.sqrt( LM_predictions[4] -val_flag*val_flag -pow(LM_predictions[1],2) )
        
        QMOM[0] = 0.25
        QMOM[3] = 0.25
        QMOM[6] = 0.25
        QMOM[9] = 0.25
        
        QMOM[1]  = LM_predictions[0] +sigmaR
        QMOM[4]  = LM_predictions[0] +sigmaR
        QMOM[7]  = LM_predictions[0] -sigmaR
        QMOM[10] = LM_predictions[0] -sigmaR
        
        QMOM[2]  = LM_predictions[1] +val_flag +sigmaRd
        QMOM[5]  = LM_predictions[1] +val_flag -sigmaRd
        QMOM[8]  = LM_predictions[1] -val_flag +sigmaRd
        QMOM[11] = LM_predictions[1] -val_flag -sigmaRd
        
        for kk in range(4,self.abscissas):
            QMOM[3*kk  ] = 0.0
            QMOM[3*kk+1] = LM_predictions[0]
            QMOM[3*kk+2] = LM_predictions[1]
        
        return QMOM;
    
    
    
    
    def time_evolve(self,ii):
        
        if (self.method == "Adams-Bashforth"):
            self.Adams_Bashforth(ii)
        elif (self.method == "Euler"):
            self.Euler_evolution(ii)
        elif (self.method == "Runge-Kutta4"):
            self.RK4_evolution(ii)
        return
    
    
    def Adams_Bashforth(self,ii):
        
        for tt in range(0,self.lm_times):
            tflag = tt-1
            
            self.LM_predictions[ii,:,tflag], self.Weight_predictions[ii,:,tflag] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tflag], self.predictions[ii,:,tflag])
            
            self.LM_RHS[ii,:,tflag] = self.RHS_Eval(self.LM_predictions[ii,:,tflag], self.Pressure[ii,tflag])
                   
            for kk in range(0,5):
                self.LM_predictions[ii,kk,tt] = self.LM_predictions[ii,kk,tflag] +(self.dt/12.0)*(23.0*self.LM_RHS[ii,kk,tflag]-16.0*self.LM_RHS[ii,kk,tflag-1]+5.0*self.LM_RHS[ii,kk,tflag-2])
        
        return
    
    
    
    def RK4_evolution(self,ii):
        
        Flag_Weights = np.zeros(30+self.ml_dim,dtype=float)
        Mom_Flag = np.zeros(30,dtype=float)
        y1 = np.zeros(30+self.ml_dim,dtype=float)
        y2 = np.zeros(30+self.ml_dim,dtype=float)
        tstart = self.run_start
        tend   = self.run_end
        press_flag = np.zeros(2,dtype=float)
        pred_flag  = np.zeros(self.ml_dim,dtype=float)
        for tt in range(tstart,tend):
            
            if (tt == tstart):
                Flag_Weights[:] = self.LM_predictions[ii,:,tt]
                
            self.ML_Pred(ii, tt)
            pred_flag[:] = self.predictions[ii,:,tt]
            
            press_flag[0] = self.Pressure[ii,tt]
            press_flag[1] = self.Pressure[ii,tt+1]
            self.RK4_step(ii,tt,self.dt,Flag_Weights,press_flag)
            
            self.LM_predictions[ii,:,tt+1] = Flag_Weights[:]
            y1[:] = Flag_Weights[:]
            
            #Do the same with dt/2
            tol = 1.0
            sub_step = int(2)
            while(tol > 1.0*pow(10,-7)):
                Flag_Weights[:] = self.LM_predictions[ii,:,tt]
                dt_flag = self.dt/float(sub_step)
                #self.predictions[ii,:,tt] = pred_flag[:]
                for jj in range(0,sub_step):
#                    if (jj == 1):
#                        self.predictions[ii,:,tt] = 0.0
                    press_flag[0] = self.Pressure[ii,tt]*float(sub_step-jj)/float(sub_step) +self.Pressure[ii,tt+1]*float(jj)/float(sub_step)
                    press_flag[1] = self.Pressure[ii,tt]*float(sub_step-1-jj)/float(sub_step) +self.Pressure[ii,tt+1]*float(jj+1)/float(sub_step)
                    self.RK4_step(ii,tt,dt_flag,Flag_Weights,press_flag)

                    
                y2[:] = Flag_Weights[:]
                tol = np.abs(y2[4]-y1[4])/y1[4]
                y1[:] = Flag_Weights[:]
                sub_step = 2*sub_step
            
            self.LM_predictions[ii,:,tt+1] = Flag_Weights[:]        
        
        return
    
    def RK4_step(self,ii,tflag,dt,Flag_Weights,press_flag):
        
        LM_flag = np.zeros(5,dtype=float)
        RHS_flag = np.zeros((4,5),dtype=float)
        prev_sol = np.zeros(30+self.ml_dim,dtype=float)
        prev_sol[:] = Flag_Weights[:]
        
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        RHS_flag[0,0:5] = self.RHS_Eval(Flag_Weights, press_flag[0])
        for kk in range(0,5):
            LM_flag[kk] = prev_sol[kk] +0.5*dt*RHS_flag[0,kk]     
        
        
        Flag_Weights[30:30+self.ml_dim] = self.CHyQMOM_Pred(LM_flag[0:5]) # Compute the QBMM abscissas 
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag]) # Compute hybrid QBMM abscissas
        RHS_flag[1,0:5] = self.RHS_Eval(Flag_Weights, 0.5*(press_flag[0]+press_flag[1]))
        for kk in range(0,5):
            LM_flag[kk] = prev_sol[kk] +0.5*dt*RHS_flag[1,kk]                
            
        
        Flag_Weights[30:30+self.ml_dim] = self.CHyQMOM_Pred(LM_flag[0:5])
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        RHS_flag[2,0:5] = self.RHS_Eval(Flag_Weights, 0.5*(press_flag[0]+press_flag[1]))
        for kk in range(0,5):
            LM_flag[kk] = prev_sol[kk] +1.0*dt*RHS_flag[2,kk]                
            
        
        Flag_Weights[30:30+self.ml_dim] = self.CHyQMOM_Pred(LM_flag[0:5])
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        RHS_flag[3,0:5] = self.RHS_Eval(Flag_Weights, press_flag[1])
        Flag_Weights[:] = prev_sol[:]
        for kk in range(0,5):
            Flag_Weights[kk] = Flag_Weights[kk] +(dt/6.0)*(RHS_flag[0,kk]+2.0*RHS_flag[1,kk]+2.0*RHS_flag[2,kk]+RHS_flag[3,kk])                
        Flag_Weights[30:30+self.ml_dim] = self.CHyQMOM_Pred(Flag_Weights[0:5])
        
        #self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        
        return
    
    def Euler_evolution(self,ii):
        
        Flag_Weights = np.zeros(30+self.ml_dim,dtype=float)
        Mom_Flag = np.zeros(30,dtype=float)
        y1 = np.zeros(30+self.ml_dim,dtype=float)
        y2 = np.zeros(30+self.ml_dim,dtype=float)
        tstart = 4200
        tend   = self.lm_times-1
        for tt in range(tstart,tend):
            
            if (tt == tstart):
                Flag_Weights[:] = self.LM_predictions[ii,:,tt]
            
            press_flag = self.Pressure[ii,tt]
            self.Euler_step(ii,tt,self.dt,Flag_Weights,press_flag)
            
            self.LM_predictions[ii,:,tt+1] = Flag_Weights[:]
            y1[:] = Flag_Weights[:]
            
            #Do the same with dt/2
            tol = 1.0
            sub_step = int(2)
            while(tol > 2.0*pow(10,-4)):
                Flag_Weights[:] = self.LM_predictions[ii,:,tt]
                dt_flag = self.dt/float(sub_step)
                for jj in range(0,sub_step):
                    press_flag = self.Pressure[ii,tt]*float(sub_step-jj)/float(sub_step) +self.Pressure[ii,tt+1]*float(jj)/float(sub_step)
                    self.Euler_step(ii,tt,dt_flag,Flag_Weights,press_flag)
                    
                y2[:] = Flag_Weights[:]
                tol = np.abs(y2[4]-y1[4])/y1[4]
                y1[:] = Flag_Weights[:]
                sub_step = 2*sub_step
            
            self.LM_predictions[ii,:,tt+1] = Flag_Weights[:]  
        
        return
    
    def Euler_step(self,ii,tflag,dt,Flag_Weights,press_flag):
                    
#        if (self.LM_predictions[ii,0,tflag] < 0.25):
#            self.ML_Pred(ii, tflag)
            
        #self.LM_predictions[ii,:,tflag], self.Weight_predictions[ii,:,tflag] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tflag], self.predictions[ii,:,tflag])
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        
        
        RHS_flag = np.zeros(5,dtype=float)
        
        #self.LM_RHS[ii,:,tflag] = self.RHS_Eval(self.LM_predictions[ii,:,tflag], self.Pressure[ii,tflag])
        RHS_flag[0:5] = self.RHS_Eval(Flag_Weights, press_flag)
        
        for kk in range(0,5):
            #self.LM_predictions[ii,kk,tflag+1] = self.LM_predictions[ii,kk,tflag] +dt*RHS_flag[kk]
            Flag_Weights[kk] = Flag_Weights[kk] +dt*RHS_flag[kk]
                
        #self.LM_predictions[ii,30:30+self.ml_dim,tflag+1] = self.CHyQMOM_Pred(self.LM_predictions[ii,0:5,tflag+1])
        Flag_Weights[30:30+self.ml_dim] = self.CHyQMOM_Pred(Flag_Weights[0:5])
        self.ML_CHyQMOM(Flag_Weights, self.predictions[ii,:,tflag])
        
        
        return
    
    
    
    
    
    
    def Runge_Kutta4(self,ii):
        
        RHS_flag = np.zeros((4,5),dtype=float)
        LM_flag  = np.zeros(5,dtype=float)
        for tt in range(1000,self.lm_times-2):
            
            
            
            LM_flag[0:5] = self.LM_predictions[ii,0:5,tt+1]
            #self.ML_Pred(ii, tt)
            self.LM_predictions[ii,:,tt], self.Weight_predictions[ii,:,tt] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tt], self.predictions[ii,:,tt])
            RHS_flag[0,:] = self.RHS_Eval(self.LM_predictions[ii,:,tt], self.Pressure[ii,tt])
            for kk in range(0,5):
                self.LM_predictions[ii,kk,tt+1] = self.LM_predictions[ii,kk,tt] +self.dt*RHS_flag[0,kk]                
            self.LM_predictions[ii,30:30+self.ml_dim,tt+1] = self.CHyQMOM_Pred(self.LM_predictions[ii,0:5,tt+1])
            
            #self.ML_Pred(ii, tt+1)
            self.LM_predictions[ii,:,tt+1], self.Weight_predictions[ii,:,tt] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tt+1], self.predictions[ii,:,tt+1])
            RHS_flag[1,:] = self.RHS_Eval(self.LM_predictions[ii,:,tt+1], self.Pressure[ii,tt+1])
            for kk in range(0,5):
                self.LM_predictions[ii,kk,tt+1] = self.LM_predictions[ii,kk,tt] +self.dt*RHS_flag[1,kk]                
            self.LM_predictions[ii,30:30+self.ml_dim,tt+1] = self.CHyQMOM_Pred(self.LM_predictions[ii,0:5,tt+1])
            
            
            #self.ML_Pred(ii, tt+1)
            self.LM_predictions[ii,:,tt+1], self.Weight_predictions[ii,:,tt] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tt+1], self.predictions[ii,:,tt+1])
            RHS_flag[2,:] = self.RHS_Eval(self.LM_predictions[ii,:,tt+1], self.Pressure[ii,tt+1])
            for kk in range(0,5):
                self.LM_predictions[ii,kk,tt+2] = self.LM_predictions[ii,kk,tt] +self.dt*RHS_flag[2,kk]                
            self.LM_predictions[ii,30:30+self.ml_dim,tt+1] = self.CHyQMOM_Pred(self.LM_predictions[ii,0:5,tt+1])
            
            #self.ML_Pred(ii, tt+2)
            self.LM_predictions[ii,:,tt+2], self.Weight_predictions[ii,:,tt] = self.ML_CHyQMOM(self.LM_predictions[ii,:,tt+2], self.predictions[ii,:,tt+2])
            RHS_flag[3,:] = self.RHS_Eval(self.LM_predictions[ii,:,tt+2], self.Pressure[ii,tt+2])
            for kk in range(0,5):
                self.LM_predictions[ii,kk,tt+2] = self.LM_predictions[ii,kk,tt] +(self.dt/6.0)*(RHS_flag[0,kk]+2.0*RHS_flag[1,kk]+2.0*RHS_flag[2,kk]+RHS_flag[3,kk])               
            self.LM_predictions[ii,30:30+self.ml_dim,tt+2] = self.CHyQMOM_Pred(self.LM_predictions[ii,0:5,tt+2])
            
            self.LM_predictions[ii,0:5,tt+1] = LM_flag[0:5]
            
        return
    
    
    def ml_testing(self,test_cases):
        
        self.test_cases = test_cases
        
        if (self.cases == "random"):
            model1 = tf.keras.models.load_model(self.model_name,compile=False)
            #model1 = tf.keras.models.load_model('Neural_Nets/LM_Random_MLQBMM_Approach'+self.approach+'_Weights'+str(self.ml_dim//3)+'.h5',compile=False)
        self.model1 = model1

        self.test_size = len(self.test_cases)
        
        self.run_start = 500
        self.run_end   = self.lm_times-1
    
        self.scale_times = (self.lm_times-1)//(self.total_times-1)
        run_times = (self.total_times-1)//self.time_jump
        input_test   = np.zeros((self.scale_times*self.test_size*self.time_jump,run_times+1,self.used_features))
        output_test  = np.zeros((self.scale_times*self.test_size*self.time_jump,run_times+1,self.output_dim))
        
#        for ii in range(0,self.test_size):
#            for kk in range(0,self.time_jump):
#                for pp in range(0,run_times):
#                    for jj in range(0,self.input_size-1):
#                        input_test[self.time_jump*ii+kk,pp,jj] = self.MC_Data[self.test_cases[ii],jj,self.time_jump*pp+kk]/self.max_in[self.test_cases[ii],jj]
#                    input_test[self.time_jump*ii+kk,pp,self.input_size-1] = self.Pressure[self.test_cases[ii],self.time_jump*pp+kk]/self.max_in[self.test_cases[ii],self.input_size-1]
#            for jj in range(0,self.input_size-1):
#                input_test[self.time_jump*ii,run_times,jj] = self.MC_Data[self.test_cases[ii],jj,self.lm_times-1]/self.max_in[self.test_cases[ii],jj]
#            input_test[self.time_jump*ii,run_times,self.input_size-1] = self.Pressure[self.test_cases[ii],self.lm_times-1]/self.max_in[self.test_cases[ii],self.input_size-1]
#        #output_test, hidden_state, cell_state = model1.predict(input_test)
#        output_test = model1.predict(input_test)
        #print(output_test.shape)
        #print(hidden_state.shape)
        #print(cell_state.shape)
        
        #model1.  reset_states(states=[np.ones((batch_size, nodes)), np.ones((batch_size, nodes))])
        #output_test, hidden_state, cell_state = tf.keras.models.Model(input_test, states=[hidden_state,cell_state])
        
        #print(model1.layers[0].states[0])
        #print(model1.layers[0].states[1])
        
        
    
        self.predictions = np.zeros((self.test_size,self.ml_dim,self.lm_times))
    
#        for ii in range(0,self.test_size):
#            for kk in range(0,self.time_jump):
#                for tt in range(0,run_times):
#                    for jj in range(0,self.ml_dim):
#                        self.predictions[ii,jj,self.time_jump*tt+kk] = output_test[self.time_jump*ii+kk,tt,jj]
#            for jj in range(0,self.ml_dim):
#                self.predictions[ii,jj,self.lm_times-1] = output_test[self.time_jump*ii,run_times,jj]
        
        
        self.Weight_predictions = np.zeros((len(self.test_cases),self.ml_dim,self.total_times),dtype=float)
        self.LM_RHS = np.zeros((len(self.test_cases),5,self.lm_times))
        self.LM_predictions = np.zeros((self.test_size,30+self.ml_dim,self.lm_times))
        self.LM_predictions[:,0:30+self.ml_dim,:]  = self.MC_Data[:,0:30+self.ml_dim,:]
        self.LM_predictions[:,30:30+self.ml_dim,:] = self.MC_Data[:,30:30+self.ml_dim,:]
        time_hist = 256
        input_flag = np.zeros((1,time_hist,self.used_features))
        
        for ii in range(0,4):
            print(ii)
            
            input_test   = np.zeros((1,self.run_start,self.used_features))
            output_test  = np.zeros((1,self.run_start,self.output_dim))
            model1.reset_states()
            
            for pp in range(0,self.run_start):
                for jj in range(0,self.input_size-1):
                    input_test[0,pp,jj] = self.MC_Data[self.test_cases[ii],jj,pp]/self.max_in[self.test_cases[ii],jj]
                input_test[0,pp,self.input_size-1] = self.Pressure[self.test_cases[ii],pp]/self.max_in[self.test_cases[ii],self.input_size-1]
                    
            output_test = model1.predict(input_test)
            
            for tt in range(0,self.run_start):
                for jj in range(0,self.ml_dim):
                    self.predictions[ii,jj,tt] = output_test[0,tt,jj]
            
            
            
            
            self.time_evolve(ii)
                    

        if (self.cases == "random"):
            sio.savemat('ML_Predictions/LM_Random_MLQBMM_Approach'+self.approach+'_Weights'+str(self.ml_dim//3)+'.mat',{'Weights_predictions':self.Weight_predictions,'predictions':self.LM_predictions,'LM_MC':self.MC_Data,'LM_QBMM':self.QBMM_Data,'LM_pressure':self.Pressure,'max_in':self.max_in,'max_out':self.max_out})

        
        return
        
        

        

        
        