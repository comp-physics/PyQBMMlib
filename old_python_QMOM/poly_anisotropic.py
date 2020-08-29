# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:24:33 2015

@author: kongbo
"""
import numpy as np
from extended_quadrature import eqmoms
from adaptive_Wheeler import adaptive_Wheeler
from numpy.linalg import inv

 
class poly_aniso(object):
    
    eq_sigmalimit  = 1.0E-6      
    mono_epslim = 5.0E-5  
    imat  = np.array([ 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    small_eps = 1E-6
    s0 = 0.5
    nn = 0; nd = 0; neqn = 0; nac = 0; naod = 0
    nsm = 0 ;  nsum = 0  
    vso = 0; svmi = 0; uvmi = 0; nsh = 0 
    inv_type =''
    
    def __init__(self,nn,neqn,nd,pdf_type,inv_type):
        
        self.nn = nn
        self.nd = nd
        self.neqn = neqn
        
        self.nsm = 1
        self.nsum = 1
        
        self.svmi = 0
        self.uvmi = 0
        self.vso = 0
        self.nsh = 0  
        
        if nd == 2:
            self.nac = 3
            self.naod = 1
        else:
            self.nac = 6
            self.naod = 3
    
        if inv_type in ('Gaussian', 'Beta', 'Gamma', 'Lognormal'):
            self.nsm = 2*self.neqn + 1
            self.nsum = self.neqn + 2
            self.inv_type =  inv_type      
    
        elif inv_type == 'QMOM':
            self.nsm = 2*self.nn
            self.nsum = self.nn
            self.neqn = self.nn
            
        else:
            print 'Wrong moment inversion type!!!', inv_type
            exit()
    
        if pdf_type == 'Number':
            self.vso = 3
            self.svmi = 3    
            self.uvmi = self.nsum - 1
            self.nsh = 4 - self.nsum
            
        elif pdf_type == 'Volume':
            self.vso = 0
            self.svmi = 0
            self.uvmi = 0
            self.nsh = 0
            
        else:
            print 'PDF_TYPE Can only be either NUMBER or VOLUME!!!', pdf_type
            exit()
        
#        self.w = np.zeros(nn)
#        self.s = np.zeros(nn)
#        self.u = np.zeros([nn,nd])
#        self.t = np.zeros(nn)
#        self.a = np.zeros(self.nac)
#        self.sq = np.zeros(neqn)
       
    def update_smoms(self,ncsn,w,s):  
        sm = np.zeros(self.nsm)     
        
        if ncsn > 1 :
            for j in range(self.nsm):
                for i in range(ncsn):
                    sm[j] +=  w[i]*s[i]**j
        else :
            for j in range(self.nsm):
                sm[j] +=  w*s**j
            
        return sm 
        
    def update_sumoms(self,ncsn,w,s,u,t,a):
        
        su1m  =  np.zeros([self.nsum,self.nd])
        su2dm =  np.zeros([self.nsum,self.nd])
        su2om =  np.zeros(self.naod) 
        
        for i in range(ncsn):
            for j in range(self.nsum):    
                tmp = w[i]*(s[i]**(j+ self.nsh))
                su1m[j,0]  +=  tmp*u[i,0]
                su1m[j,1]  +=  tmp*u[i,1]
                su2dm[j,0] +=  tmp*(u[i,0]**2 + t[i]*a[0])
                su2dm[j,1] +=  tmp*(u[i,1]**2 + t[i]*a[1])
                
            su2om[0] += w[i]*(s[i]**self.vso)*(u[i,0]*u[i,1] + t[i]*a[2])
            
        if self.nd == 3:
            for i in range(ncsn):
                for j in range(self.nsum):    
                    tmp = w[i]*(s[i]**(j+ self.nsh))
                    su1m[j,2]  +=  tmp*u[i,2]
                    su2dm[j,2] +=  tmp*(u[i,2]**2 + t[i]*a[3])
                    
                su2om[1] += w[i]*(s[i]**self.vso)(u[i,0]*u[i,2] + t[i]*a[4])       
                su2om[2] += w[i]*(s[i]**self.vso)(u[i,1]*u[i,2] + t[i]*a[5])     
                
        return su1m,su2dm,su2om
                
    def update_sizes(self,sm): 
        
        eabs = 1.0E-4
        if self.nn == self.neqn : 
            w, sq, ncsn, _ = adaptive_Wheeler(sm,self.nn,eabs)
            s = sq
        else:

            eq = eqmoms[self.inv_type](self.neqn)
            eq.mom_inv(sm) 
            w,s = eq.calc_secQuad(self.nn/self.neqn)
            ncsn = self.nn
            
            pm = np.zeros(2*self.nsum)     
            for j in range(2*self.nsum):
                for i in range(ncsn):
                    pm[j] +=  w[i]*s[i]**j
                    
            wq, sq, nout, _ = adaptive_Wheeler(pm,self.nsum,eabs)
            
        return w,s,sq,ncsn
    
    def qmom_vel(self,ncsn,su1m,su2dm,w,s): 
        
        amat = np.zeros([ncsn,ncsn])
        
        for i in range(ncsn):
            for j in range(ncsn):
                amat[i,j] += w[j]*s[j]**(i+self.nsh)
    
        ainv = inv(amat)
        
        u = np.dot(ainv,su1m)
        
        tsrc  = np.zeros(ncsn)
        
        if self.nd == 2 :
            for i in range(ncsn):
                for j in range(ncsn):
                    tsrc[i] += w[j]*s[j]**(i+self.nsh)*(u[j,0]**2+u[j,1]**2)
                    
            for i in range(ncsn):
                tsrc[i] = 0.5*(su2dm[i,0]+su2dm[i,1] - tsrc[i])
        else:
            for i in range(ncsn):
                for j in range(ncsn):
                    tsrc[i] += w[j]*s[j]**(i+self.nsh)*(u[j,0]**2+u[j,1]**2+u[j,3]**2)
            for i in range(ncsn):
                tsrc[i] = 1.0/3.0*(su2dm[i,0]+su2dm[i,1] +su2dm[i,2] - tsrc[i])
                
        t =  np.dot(ainv,tsrc)
        
        return u,t
    
    
    def piecewise(self,s,sx,nsx):
        last = nsx-1
        y = np.zeros(nsx)
        if s < sx[0] :
            y[0] = 1.0
        elif s >= sx[last] :
            y[last] = 1.0
        else:
            for i in range(last):
                if s>= sx[i] and s< sx[i+1] :
                    y[i] = (sx[i+1] - s)/( sx[i+1] - sx[i])
                    y[i+1] = 1.0 - y[i]
        return y
               
    def eqmom_vel(self,ncsn,su1m,su2dm,w,s,sq):
        
#        neqn2 = self.neqn+2
#        sx = np.zeros(neqn2)
#        sx[0] = min(s)
#        sx[neqn2-1] = max(s)
#        for i in range(self.neqn):
#            sx[i+1] = sq[i]
       
        g = np.zeros([self.nsum,self.nn])
        for i in range(ncsn) :
            g[:,i] = self.piecewise(s[i], sq, self.nsum)
            
        gmat = np.zeros([self.nsum,self.nsum])
        for i in range(self.nsum):
            for a in range(ncsn):
                tmp = w[a]*s[a]**(i+self.nsh)
                for j in range(self.nsum):
                    gmat[i,j] += tmp*g[j,a]
        
        ginv = inv(gmat)
        
        ucoef = np.transpose(np.dot(ginv,su1m))
        u =  np.transpose(np.dot(ucoef,g))
        
        tsrc  = np.zeros([self.nsum])
        
        if self.nd == 2 :
            for i in range(self.nsum):
                for j in range(ncsn):
                    tsrc[i] += w[j]*s[j]**(i+self.nsh)*(u[j,0]**2+u[j,1]**2)
                    
            for i in range(self.nsum):
                tsrc[i] = 0.5*(su2dm[i,0]+su2dm[i,1] - tsrc[i])
        else:
            for i in range(self.nsum):
                for j in range(ncsn):
                    tsrc[i] += w[j]*s[j]**(i+self.nsh)*(u[j,0]**2+u[j,1]**2+u[j,3]**2)
            for i in range(self.nsum):
                tsrc[i] = 1.0/3.0*(su2dm[i,0]+su2dm[i,1] +su2dm[i,2] - tsrc[i])
                
        tcoef =  np.transpose(np.dot(ginv,tsrc))
                
        t =   np.dot(tcoef,g)
    
        return u,t
        
    def update_vels(self,ncsn,sm,su1m,su2dm,su2om,w,s,sq):     
        
        if self.nn == self.neqn : 
            u,t = self.qmom_vel(ncsn,su1m,su2dm,w,s)
        else:
            u,t = self.eqmom_vel(ncsn,su1m,su2dm,w,s,sq)
        
        a = self.update_covartsr(ncsn,su2dm,su2om,w,s,u,t)
            
        return u,t,a
    
        
    def update_covartsr(self,ncsn,su2dm,su2om,w,s,u,t):
        
        a = np.zeros(self.nac)
        
        a[0] = su2dm[self.uvmi,0]
        a[1] = su2dm[self.uvmi,1]
        a[2] = su2om[0]
        
        tint = 0.0
        u2int = np.zeros(self.nac)
        for i in range(ncsn):
            tmp = w[i]*(s[i]**self.vso)
            u2int[0] +=  tmp*(u[i,0]**2)
            u2int[1] +=  tmp*(u[i,1]**2)
            u2int[2] +=  tmp*(u[i,0]*u[i,1])
            tint  += tmp*t[i]
            
        if self.nd == 3 :
            a[3] = su2dm[self.uvmi,2]
            a[4] = su2om[1]
            a[5] = su2om[2]
            for i in range(ncsn):
                tmp = w[i]*(s[i]**self.vso)
                u2int[3] += tmp*(u[i,2]**2)
                u2int[4] += tmp*(u[i,0]*u[i,2])
                u2int[5] += tmp*(u[i,1]*u[i,2])
    
        a = (a - u2int)/tint
    
        return a
        
        
    def inversion(self,sm,su1m,su2dm,su2om):     
        
        w,s,sq,ncsn = self.update_sizes(sm)
        u,t,a = self.update_vels(ncsn,sm,su1m,su2dm,su2om,w,s,sq)
        
        return w,s,u,t,a,ncsn
