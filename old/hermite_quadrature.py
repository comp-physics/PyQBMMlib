# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:21:54 2015

@author: kongbo
"""

import numpy as np
from scipy import linalg as la
from second_quadrature import hermite_secquad


class hermite_quadrature:
    
    def __init__(self,n1,nd):
    # Make sure number of nodes is no larger than 5.
        if n1 <= 0:
          print("Non-positive number of nodes. Program exits.")
          exit()
        else:
         self.n1 = n1
         self.nd = nd
         self.nt = n1**nd
         self.w  = np.zeros(self.nt)
         self.ox = np.zeros([self.nt,nd])
         self.x  = np.zeros([self.nt,nd])
         self.threshold = 10**(-8)
        
        hsq = hermite_secquad(n1)
        limit = np.zeros(2)
        limit[1] = 1.0
        hsq.calc_secquad(limit,1.0,0.0,1.0)
    
        if(nd==1):
            for i in range(self.n1): 
                self.w[i]  = hsq.sw[i]
                self.ox[i] = hsq.sx[i]
        elif(nd==2):
            for i in range(self.n1):
                for j in range(self.n1):
                    p = i*self.n1 + j
                    self.w[p] = hsq.sw[i]*hsq.sw[j]
                    self.ox[p][0] = hsq.sx[i]
                    self.ox[p][1] = hsq.sx[j]
                    if(self.w[p] < self.threshold):
                        self.w[p] = 0.0
        else:
            
            for i in range(self.n1):
                for j in range(self.n1):
                    for k in range(self.n1):  
                        p = i*self.n1*self.n1 + j*self.n1+k
                        self.w[p] = hsq.sw[i]*hsq.sw[j]*hsq.sw[k]
                        self.ox[p][0] = hsq.sx[i]
                        self.ox[p][1] = hsq.sx[j]
                        self.ox[p][2] = hsq.sx[k]
                        if(self.w[p] < self.threshold):
                            self.w[p] = 0.0

    
    def calc_eigen(self, mu,theta,A):
          # A = 11, 22,12,33,13,23
      
          z  = np.zeros([self.nd,self.nd])
          Em = np.zeros([self.nd,self.nd])
          
          if(self.nd == 1):
              
              for i in range(self.n1): 
                  self.x[i] = theta*self.ox[i] + mu[0]
              return 
          elif(self.nd == 2):
              
              z[0][0] = theta*A[0]
              z[0][1] = theta*A[2]
              z[1][0] = theta*A[2]
              z[1][1] = theta*A[1]
              
          else:
              z[0][0] = theta*A[0]
              z[0][1] = theta*A[2]
              z[0][2] = theta*A[4]
              z[1][0] = theta*A[2]
              z[1][1] = theta*A[1]
              z[1][2] = theta*A[5]
              z[2][0] = theta*A[4]
              z[2][1] = theta*A[5]
              z[2][2] = theta*A[3]
      
          e, d = la.eig(z)
          
          for i in range(self.nd):
              if e[i] >= 0 :
                   Em[i,i] = np.sqrt(e[i].real)
              else:
                   Em[i,i] = 0.0
                   
          z = np.dot(d,Em)
          
          if(self.nd==2):
            for i in range(self.n1):
                for j in range(self.n1):
                    p = i*self.n1 + j
                    self.x[p][0] = z[0][0]*self.ox[p][0] + z[0][1]*self.ox[p][1] + mu[0]
                    self.x[p][1] = z[1][0]*self.ox[p][0] + z[1][1]*self.ox[p][1] + mu[1]
          else:
            for i in range(self.n1):
                for j in range(self.n1):
                    for k in range(self.n1):  
                        p = i*self.n1*self.n1 + j*self.n1+k
                        self.x[p][0] = z[0][0]*self.ox[p][0] + z[0][1]*self.ox[p][1] + z[0][2]*self.ox[p][2] + mu[0]
                        self.x[p][1] = z[1][0]*self.ox[p][0] + z[1][1]*self.ox[p][1] + z[1][2]*self.ox[p][2] + mu[1]
                        self.x[p][2] = z[2][0]*self.ox[p][0] + z[2][1]*self.ox[p][1] + z[2][2]*self.ox[p][2] + mu[2]
          
          
        
    def calc_chol(self, mu,theta,A,d):
          # A = 11, 22,12,33,13,23
          # d = 0,1,2  ---x,y,z
          z = np.zeros([self.nd,self.nd])
          if(self.nd == 1):
              for i in range(self.n1): 
                  self.x[i] = theta*self.ox[i] + mu[0]
              return 
              
          elif(self.nd == 2):
             
              z[0][1] = theta*A[2]
              
              if( d == 0):
                  z[0][0] = theta*A[0]
                  z[1][1] = theta*A[1]
              
                  z = la.cholesky(z)
                  z = z.T
                  
                  for i in range(self.n1):
                      for j in range(self.n1):
                          p = i*self.n1 + j
                          self.x[p][0] = z[0][0]*self.ox[p][0] + z[0][1]*self.ox[p][1] + mu[0]
                          self.x[p][1] = z[1][0]*self.ox[p][0] + z[1][1]*self.ox[p][1] + mu[1] 
              else:
                  
                  z[0][0] = theta*A[1]
                  z[1][1] = theta*A[0]
              
                  z = la.cholesky(z)
                  z = z.T
                  
                  for i in range(self.n1):
                      for j in range(self.n1):
                          p = i*self.n1 + j
                          self.x[p][1] = z[0][0]*self.ox[p][1] + z[0][1]*self.ox[p][0] + mu[1]
                          self.x[p][0] = z[1][0]*self.ox[p][1] + z[1][1]*self.ox[p][0] + mu[0] 
              
          else:
              
              t = np.zeros([self.nd,self.nd])
              t[0][0] = theta*A[0]
              t[1][1] = theta*A[1]
              t[0][1] = theta*A[2]
              
              t[2][2] = theta*A[3]
              t[0][2] = theta*A[4]
              t[1][2] = theta*A[5]
              
              if(d == 0 ):
                  #xyz 012
                  
                  z = la.cholesky(t)
                  z = z.T
                  
                  for i in range(self.n1):
                    for j in range(self.n1):
                        for k in range(self.n1):  
                            p = i*self.n1*self.n1 + j*self.n1+k
                            self.x[p][0] = z[0][0]*self.ox[p][0] + z[0][1]*self.ox[p][1] + z[0][2]*self.ox[p][2] + mu[0]
                            self.x[p][1] = z[1][0]*self.ox[p][0] + z[1][1]*self.ox[p][1] + z[1][2]*self.ox[p][2] + mu[1]
                            self.x[p][2] = z[2][0]*self.ox[p][0] + z[2][1]*self.ox[p][1] + z[2][2]*self.ox[p][2] + mu[2]
                          
              elif(d == 1):
                  #yzx  120
                  z[0][0] = t[1][1]
                  z[1][1] = t[2][2]
                  z[0][1] = t[1][2]
                  
                  z[2][2] = t[0][0]
                  z[0][2] = t[0][1]
                  z[1][2] = t[0][2]
                                       
                                   
                  z = la.cholesky(z)
                  z = z.T
                  
                  for i in range(self.n1):
                    for j in range(self.n1):
                        for k in range(self.n1):  
                            p = i*self.n1*self.n1 + j*self.n1+k
                            self.x[p][1] = z[0][0]*self.ox[p][1] + z[0][1]*self.ox[p][2] + z[0][2]*self.ox[p][0] + mu[1]
                            self.x[p][2] = z[1][0]*self.ox[p][1] + z[1][1]*self.ox[p][2] + z[1][2]*self.ox[p][0] + mu[2]
                            self.x[p][0] = z[2][0]*self.ox[p][1] + z[2][1]*self.ox[p][2] + z[2][2]*self.ox[p][0] + mu[0]
                              
                 
              else :
                 #zxy 201
                  z[0][0] = t[2][2]
                  z[1][1] = t[0][0]
                  z[0][1] = t[0][2]
                  
                  z[2][2] = t[1][1]
                  z[0][2] = t[1][2]
                  z[1][2] = t[0][1]
                                   
                                   
                  z = la.cholesky(z)
                  z = z.T
                  
                  for i in range(self.n1):
                    for j in range(self.n1):
                        for k in range(self.n1):  
                            p = i*self.n1*self.n1 + j*self.n1+k
                            self.x[p][2] = z[0][0]*self.ox[p][2] + z[0][1]*self.ox[p][0] + z[0][2]*self.ox[p][1] + mu[2]
                            self.x[p][0] = z[1][0]*self.ox[p][2] + z[1][1]*self.ox[p][0] + z[1][2]*self.ox[p][1] + mu[0]
                            self.x[p][1] = z[2][0]*self.ox[p][2] + z[2][1]*self.ox[p][0] + z[2][2]*self.ox[p][1] + mu[1]
                          
