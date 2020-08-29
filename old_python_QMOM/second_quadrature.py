# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:41:08 2015

@author: kongbo
"""

from scipy import zeros,sqrt,exp
from scipy.special import gamma
from scipy.linalg import eig

class secquad:    

  def __init__(self,nsqn):
    # Make sure number of nodes is no larger than 5.
    if nsqn <= 0:
      print("Non-positive number of second quadrature nodes. Program exits.")
      exit()
    else:
     self.nsqn = nsqn
     self.sw = zeros(nsqn)
     self.sx = zeros(nsqn)
     self.ab = zeros([nsqn,2])
     self.z = zeros([nsqn,nsqn])
     
  def calc_secquad(self,limit, w, x, sig):
     
    self.sw[:] = 0.0
    self.sx[:] = 0.0
    	
    if sig/(limit[1]-limit[0]) < 1e-8  :
    		self.sw[0] = w
    		self.sx[0] = x
    		return
    
    self.solveRecuCoef(limit,x,sig)
    self.solveSecNodes()
    self.scalebackNodes(limit,w,x,sig)
      
 
  def solveRecuCoef(self,limit,x,sig):
      pass
  
  def scalebackNodes(self,limit,w,x,sig):
      pass
  
  def solveSecNodes(self):

     for i in range(0,self.nsqn-1) :
          self.z[i][i] = self.ab[i][0]
          self.z[i][i+1] =  sqrt(self.ab[i+1][1])
          self.z[i+1][i] = self.z[i][i+1]
	
     self.z[self.nsqn-1][self.nsqn-1] = self.ab[self.nsqn-1][0]

     e,v =  eig(self.z)

     for i in range(0,self.nsqn) :
         self.sw[i] = v[0][i]**2
         self.sx[i] = e[i].real

     self.sw = self.sw/sum(self.sw) ;


class hermite_secquad(secquad):
     
  def solveRecuCoef(self,limit,x,sig):
      for i in range(1,self.nsqn):
          self.ab[i][1]= i
  
  def scalebackNodes(self,limit,w,x,sig):
      self.sw = self.sw*w
      self.sx = x + sig*self.sx
      
class jacobi_secquad(secquad):
     
  def solveRecuCoef(self,limit,x,sig):
      alpha = (limit[1]-x)/sig - 1 
      beta  =  (x-limit[0])/sig - 1 
      
      if alpha <= -1 or beta <= -1 :
		if (alpha <= -1):
			alpha = -1+1e-14
		if (beta <= -1):
			beta = -1+1e-14
      
      if((alpha<=-1) or (beta<=-1)) :
         print " jacobiSecondQuadrature parameter(s) out of range ! "
         exit()
      
      self.ab[0][0] = (beta-alpha)/(alpha+beta+2)
      self.ab[0][1] = 0.0  # pow(2,(alpha+beta+1))*gamma.calc(alpha+1)*gamma.calc(beta+1)/gamma.calc(alpha+beta+2);

      nab = zeros(self.nsqn-1)
      for i in range(0,self.nsqn-1):
          nab[i] =2*(i+1.0)+alpha+beta
      
      for i in range(1,self.nsqn):
		self.ab[i][0] =(beta*beta-alpha*alpha)/(nab[i-1]*(nab[i-1]+2.0))
      
      self.ab[1][1]=4*(alpha+1)*(beta+1)/((alpha+beta+2)*(alpha+beta+2)*(alpha+beta+3))
      
      for i in range(2,self.nsqn):
		self.ab[i][1]=4*(i+alpha)*(i+beta)*(i)*(i+alpha+beta)/((nab[i-1]*nab[i-1])*(nab[i-1]+1)*(nab[i-1]-1))
  
  
  def scalebackNodes(self,limit,w,x,sig):
      self.sw = self.sw*w
      self.sx = (1.0 + self.sx)/2.0
      self.sx = limit[0] + (limit[1]-limit[0])*self.sx



class laguerre_secquad(secquad):
     
  def solveRecuCoef(self,limit,x,sig):
      
      alpha = (x-limit[0]-sig)/sig 
      if(alpha<=-1):
          print " laguerreSecondQuadrature parameter(s) out of range ! "
          exit()

      self.ab[0][0] = alpha+1 
      self.ab[0][1] = gamma(alpha+1)
      
      for i in range(1,self.nsqn):
		self.ab[i][0] =2*i+alpha+1.0
      for i in range(1,self.nsqn):
		self.ab[i][1]= i*(i+alpha)
  
  def scalebackNodes(self,limit,w,x,sig):
      self.sw = self.sw*w
      self.sx = limit[0] + sig*self.sx
      
      
      
class stieltjes_secquad(secquad):
     
  def solveRecuCoef(self,limit,x,sig):
      E = exp(sig**2/2.0)
      self.ab[0][0] = E 
      self.ab[0][1] = 0.0
      
      for i in range(1,self.nsqn):
		self.ab[i][0]= ((E**2+1.0)*pow(E,2*i) - 1.0)*pow(E,2*i-1);
		self.ab[i][1]= (pow(E,2*i) - 1.0)*pow(E,6*i-4);

  def scalebackNodes(self,limit,w,x,sig):
      self.sw[:] = self.sw[:]*w
      self.sx[:] = self.sx[:]*x 