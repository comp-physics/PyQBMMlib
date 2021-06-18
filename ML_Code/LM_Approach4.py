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

from Form_ML_Data import Load_LM_Data_Output
#from Form_ML_Data import Load_Data_QBMM
from ML_Training  import LM_Training
from ML_Training  import LM_Testing_Approach4
from loss_params  import loss_params
from ML_Data      import ML_Config



# Load MC data
approach = "4"
cases = "random"
abscissas = int(4)
#method = "Euler"
#method = "Adams-Bashforth"
method = "Runge-Kutta4"


ml_config = ML_Config(approach,cases,method)
ml_config.ml_data(abscissas)

ml_config.training_init()
#train_cases = [0,1,2,3]
#ml_config.ml_training(train_cases)

test_cases = [ii for ii in range(0,4)]
ml_config.ml_testing(test_cases)





