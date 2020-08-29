# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 11:42:36 2015

@author: kongbo
"""

from scipy import zeros,array,log,exp,sqrt,pi
from scipy.linalg import det
from scipy.special import betaln,gammaln
#from univariateInversion.adaptive_Wheeler import adaptive_Wheeler
#from univariateInversion.second_quadrature  import hermite_secquad,jacobi_secquad
#from univariateInversion.second_quadrature  import laguerre_secquad,stieltjes_secquad
from adaptive_Wheeler import adaptive_Wheeler
from second_quadrature  import hermite_secquad,jacobi_secquad
from second_quadrature  import laguerre_secquad,stieltjes_secquad
from  sys import exit 

eqmoms = {}

class eqmom(object):

  eabs = 1.e-5
  itmax = 1000 # maximum number of iteration
  mtol = 1e-13 # tolerance of moments conservation
  ftol = 1e-14 # tolerance of target function
  stol = 1e-14 # tolerance of sigma
  ztol = 1e-14 # tolerance for function of zbrent method
  sigmaPertubation = 1e-3

  @staticmethod
  def getType(string,nn) :
    if string == 'Beta':
          return beta_eqmom(nn)
    elif string == 'Gamma':
          return gamma_eqmom(nn)
    elif string == 'Lognorm':
          return lognorm_eqmom(nn)
    elif string == 'Gaussian':
          return gaussian_eqmom(nn)
    else:
          print "Wrong EQMOM type!!!"
          exit()

  def __init__(self,nn):
    # Make sure number of nodes is no larger than 5.
    if nn > 5:
      print("Exceed maximum number of nodes for beta EQMOM. Program exits.")
      exit()
    elif nn <= 0:
      print("Non-positive number of nodes. Program exits.")
      exit()
    else:
     self.nn = nn
     self.nm = 2*nn+1
     self.a = 0.0
     self.d = 1.0
     self.sig = 0.0
     self.m0 = 0.0
     self.ncsn = nn
     self.moms = zeros(self.nm)
     self.weig = zeros(nn)
     self.absc = zeros(nn)
     self.ms = zeros(self.nm )
     self.mt = zeros(self.nm )


  def mom_inv(self,moms):
     
    if (moms[0] <= 0.0)  : 
        print 'Moments are not realizable, mom[0] <= 0.0'
        exit()
        
    if moms[0] <= 1e-15:
        print("m[0] is too small. Return 0 solutions.")
        
    if len(moms) != self.nm:
        print("Not enough moments to generate "+str(self.nn)+" nodes. Program exits.")
        exit()
    else:
        self.moms = moms

    self.weig[:] = 0.0
    self.absc[:]  = 0.0
    self.sig = 0.0
    self.nout = 0
    
    self.rescale_moms()

    if( self.nn == 1) :
        self.solve_1node()
    else:
        self.solve_sigma()

    self.scaleback_nodes()


  def calc_moms(self):
     
    self.rescale_nodes()

    self.calc_mt(self.sig)

    self.scaleback_moms()
    
    self.scaleback_nodes()
      
  def calc_secQuad(self, nsqn):    
     pass

  def ms2mt(self,sig):
     pass

  def m2ms(self,sig):
     pass
 
  def targetHdet(self,sig):
     return 0.0
     
  def calc_mt(self,sig):
      
   self.ms[:] = 0.0

   for i in range(0,self.nm):
        for j in range(0,self.nn):
            self.ms[i] =  self.ms[i] + self.weig[j]*(self.absc[j]**i)

   self.ms2mt(sig)

  def rescale_moms(self):
     
    self.m0 = self.moms[0];

    if( (self.a != 0.0) or (self.d != 1.0) ) : 
        aPow = zeros(self.nn*2+1)
        dPow = zeros(self.nn*2+1)

        for i in range(0,len(aPow)) :
            aPow[i] = self.a**i
            dPow[i] = self.d**i
            
        self.mt[0] =  self.moms[0] 
        self.mt[1] = (self.moms[1] -  self.a*self.mt[0])/self.d 
        self.mt[2] = (self.moms[2] - (aPow[2]*self.mt[0] + 2*self.a*self.d*self.mt[1]))/dPow[2] 

        if(self.nn>=2) :
            self.mt[3] = (self.moms[3] - (aPow[3]*self.mt[0] + 3*aPow[2]*self.d*self.mt[1] +  3*self.a*dPow[2]*self.mt[2]))/dPow[3] ;
            self.mt[4] = (self.moms[4] - (aPow[4]*self.mt[0] + 4*aPow[3]*self.d*self.mt[1] +  6*aPow[2]*dPow[2]*self.mt[2] +  4*self.a*dPow[3]*self.mt[3]))/dPow[4] ;

        if(self.nn>=3):
            self.mt[5] = (self.moms[5] - (aPow[5]*self.mt[0] + 5*aPow[4]*self.d*self.mt[1] + 10*aPow[3]*dPow[2]*self.mt[2] + 10*aPow[2]*dPow[3]*self.mt[3] +  5*self.a*dPow[4]*self.mt[4]))/dPow[5] 
            self.mt[6] = (self.moms[6] - (aPow[6]*self.mt[0] + 6*aPow[5]*self.d*self.mt[1] + 15*aPow[4]*dPow[2]*self.mt[2] + 20*aPow[3]*dPow[3]*self.mt[3] + 15*aPow[2]*dPow[4]*self.mt[4] +  6*self.a*dPow[5]*self.mt[5]))/dPow[6] 

        if(self.nn>=4):
            self.mt[7] = (self.moms[7] - (aPow[7]*self.mt[0] + 7*aPow[6]*self.d*self.mt[1] + 21*aPow[5]*dPow[2]*self.mt[2] + 35*aPow[4]*dPow[3]*self.mt[3] + 35*aPow[3]*dPow[4]*self.mt[4] + 21*aPow[2]*dPow[5]*self.mt[5] +  7*self.a*dPow[6]*self.mt[6]))/dPow[7]
            self.mt[8] = (self.moms[8] - (aPow[8]*self.mt[0] + 8*aPow[7]*self.d*self.mt[1] + 28*aPow[6]*dPow[2]*self.mt[2] + 56*aPow[5]*dPow[3]*self.mt[3] + 70*aPow[4]*dPow[4]*self.mt[4] + 56*aPow[3]*dPow[5]*self.mt[5] + 28*aPow[2]*dPow[6]*self.mt[6] + 8*self.a*dPow[7] *self.mt[7]))/dPow[8] 

        self.moms = self.mt
        
    self.moms[:] = self.moms[:]/self.m0
 
  def scaleback_moms(self):
      
    if( (self.a != 0.0) or (self.d != 1.0) ) :
        
        aPow = zeros(self.nn*2+1)
        dPow = zeros(self.nn*2+1)
        
        for i in range(0,len(aPow)) : 
            aPow[i] = self.a**i
            dPow[i] = self.d**i
            
        self.moms[0] = self.mt[0] ;
        self.moms[1] = self.a* self.mt[0] + self.d*self.mt[1] ;
        self.moms[2] = aPow[2]*self.mt[0] + 2*self.a*self.d*self.mt[1] + dPow[2]*self.mt[2] ;

        if(self.nn>=2) :
            self.moms[3] = aPow[3]*self.mt[0] + 3*aPow[2]*self.d*self.mt[1] +   3*self.a*dPow[2]*self.mt[2]  + dPow[3]*self.mt[3] ;
            self.moms[4] = aPow[4]*self.mt[0] + 4*aPow[3]*self.d*self.mt[1] +  6*aPow[2]*dPow[2]*self.mt[2] +  4*self.a*dPow[3]*self.mt[3] + dPow[4]*self.mt[4] ;
        
        if(self.nn>=3) :
            self.moms[5] = aPow[5]*self.mt[0] + 5*aPow[4]*self.d*self.mt[1] + 10*aPow[3]*dPow[2]*self.mt[2] + 10*aPow[2]*dPow[3]*self.mt[3] +  5*self.a*dPow[4]*self.mt[4] +  dPow[5]*self.mt[5] ;
            self.moms[6] = aPow[6]*self.mt[0] + 6*aPow[5]*self.d*self.mt[1] + 15*aPow[4]*dPow[2]*self.mt[2] + 20*aPow[3]*dPow[3]*self.mt[3] + 15*aPow[2]*dPow[4]*self.mt[4] +  6*self.a*dPow[5]*self.mt[5] + dPow[6]*self.mt[6] 
        
        if(self.nn>=4) :
            self.moms[7] = aPow[7]*self.mt[0] + 7*aPow[6]*self.d*self.mt[1] + 21*aPow[5]*dPow[2]*self.mt[2] + 35*aPow[4]*dPow[3]*self.mt[3] + 35*aPow[3]*dPow[4]*self.mt[4] + 21*aPow[2]*dPow[5]*self.mt[5] +  7*self.a *dPow[7]*self.mt[6] + dPow[7]*self.mt[7] 
            self.moms[8] = aPow[8]*self.mt[0] + 8*aPow[7]*self.d*self.mt[1] + 28*aPow[6]*dPow[2]*self.mt[2] + 56*aPow[5]*dPow[3]*self.mt[3] + 70*aPow[4]*dPow[4]*self.mt[4] + 56*aPow[3]*dPow[5]*self.mt[5] + 28*aPow[2]*dPow[7]*self.mt[6] + 8*self.a*dPow[7]*self.mt[7] + dPow[8]*self.mt[8] 

    else :
        self.moms = self.mt
    
    self.moms = self.moms*self.m0
 
 
  def rescale_nodes(self):
    self.m0 = sum(self.weig)
    self.weig = self.weig / self.m0
    if( self.a!= 0.0 or self.d != 1.0 ) :
        self.sig =  self.sig / self.d
        self.absc = (self.absc - self.a) /self.d;
    
 
  def scaleback_nodes(self):
    self.weig = self.weig*self.m0
    if( self.a!= 0.0 or self.d != 1.0 ): 
        self.sig = self.sig * self.d
        self.absc = self.a + self.d*self.absc
 
  def initial_df(self):
    """ Return the initial approximate slope df to start iteration."""
    self.m2ms(0.0)
    df = 0.0
    if self.nn == 1:
        df = -self.ms[1]
    elif self.nn == 2:
        df = -6*self.ms[3]
    elif self.nn == 3:
        df = -15*self.ms[5]
    elif self.nn == 4:
        df = -28*self.ms[7]
    elif self.nn == 5:
        df = -45*self.ms[9]
    else:
        print("Wrong nnodes input!")
    return df
    
  def calc_sigmax(self) :
     return 1E2
     
  def solve_sigma(self):
    """ Return weights, nodes, sigma, number of nodes, and iteration number """
    # Find maximum value of sigma = total variance (upper bound)
    sigmax = self.calc_sigmax()
    method = True 
# First check sigma = 0 works or not
# Check if sigma = 0 gives negative Hankel determinant
# sig=0 must satisfy Hdet>=0 or else return with QMOM

    sig = 0.0;

    Hdet = self.targetHdet(sig)
    if (Hdet < 0): 
        self.update_nodes(sig)
        return

# Check if sigma = 0 is the solution
    f = self.target_func(sig)
    if (abs(f) < self.ftol) : 
        return

# Store best possible solution
    maxSig_HdetPos = sig;
    target_func_best = abs(f);
    sig_best = sig

# Look for sig > 0 for possible solutions, using  Secant Method (SM) or Bisection Method (BM)
# To start with Secant Method because it converges at a faster rate
# SM :: sig_new = sig - (f*dsig/df) where df = (f - f_old)  and dsig = (sig - sig_old)
    f_old = f
    sig_old = sig

# But to start with we do not know sig_old and f_old
# Therefore, we start with df_by_dsig can be computed by perturbing sig by a small amount
    dsig = self.sigmaPertubation

    sig_new = sig_old + dsig

    for ite in range(0,self.itmax) :
        
 #       print 'current sig:', sig_new
        
        Hdet = self.targetHdet(sig_new) 
#        print 'Hdet:', Hdet
        
        if (Hdet < 0) :
        # If Hdet < 0,  find the value of sig_Hdet_0 for which Hdet = 0 store it
            sig_Hdet_0 =  self.brentMethod(maxSig_HdetPos, sig_new)
            f = self.target_func(sig_Hdet_0)

            if ( abs(f) < target_func_best) :  # Store best possible solution obatined till now
                target_func_best = abs(f)
                sig_best  = sig_Hdet_0
        
        else :
                maxSig_HdetPos = max (maxSig_HdetPos, sig_new)

        # check for convergence
        f_new = self.target_func(sig_new)
#        print 'f_new', f_new
#        print 'sig_new', sig_new
#        exit() 
#        
        if (abs(f_new) < self.ftol) : 
           if( Hdet < 0.0):
              self.update_nodes(sig_best); # check if hdet is positive or not, if not use sig_best
           else :
              return   # converaged, sucess !!!
        else:
        # Store best possible solution obatined till now
           if (abs(f_new) < target_func_best) :
              target_func_best = abs(f_new)
              sig_best  = sig_new
        
    # check if the iteration is stalled or not
        df = f_new - f_old
        dsig = sig_new - sig_old
        if( abs (df) < self.ftol or abs (dsig) < self.stol) :
            if ( abs(f_new) < target_func_best):
                sig_best  = sig_new
            self.update_nodes(sig_best);
            return 

    # new sigma guess
        if (f_new >= 0 and method ) :
            f_old   = f
            sig_old = sig
            f   = f_new
            sig = sig_new

            # Secant Method
            dsig = -f*dsig/df
            sig_new = sig + dsig

            #check the sigma range
            sig_new = max(sig_new,0) 
            if(sig_new >= sigmax) :
                sig_new = (sig+sigmax)/2.0
        else :
    
            if (f*f_new < 0) :
                f_old   = f
                sig_old = sig

            f   = f_new
            sig = sig_new

            # Bisection Method
            sig_new = 0.5*( sig + sig_old)
            method = False

    # if j >= itermax, and still not converaged, just choose the best fit sigma so far.
    print  "too many iterations! : increase itmax \n" 

    f_new = self.target_func(sig_new)
    if ( abs(f_new) < target_func_best) :
        sig_best  = sig_new;

    self.update_nodes(sig_best)
    
    return
    
    
  def update_nodes(self, sig):
     
    self.m2ms(sig)
    self.weig[:] = 0
    self.absc[:] = 0
    eabs1 = max(self.eabs,sig/50)
    w, x, nout, _ = adaptive_Wheeler(self.ms,self.nn, eabs1)
    
    self.weig[0:nout] = w
    self.absc[0:nout] = x

    self.sig = sig
    self.ncsn = nout
 
  def target_func(self,sig):
    midx = self.nm-1
    self.update_nodes(sig)
    self.calc_mt(sig)

#    print 'tar sig',sig
#    print 'tar ms', self.ms 
#    print 'tar w', self.weig
#    print 'tar x', self.absc
#    print 'tar ncsn', self.ncsn
#    print 'targ mt',self.mt[midx]
#    print 
    
    return (self.moms[midx] - self.mt[midx] )/ self.moms[midx];


  def momDiffNorm(self, sig):
     
    self.update_nodes(sig) 
    self.calc_mt()

    norm = 0.0
    for i in range(0,self.nm):
        tmp = (self.mt[i]-self.moms[i])/self.moms[i];
        norm += tmp*tmp
    
    return norm
    
  def brentMethod(self,x1, x2):
    """ Return the root b found by Brent's method lying between x1 and x2. 
    Decrease b until beta_self.targetHdet > 0.
    """
    ITMAX = 100
    iimax = 100
    EPS = 1.0e-14
    a = x1
    b = x2
    c = x2
    fa = self.targetHdet(a)
    fb = self.targetHdet(b)
    if fa*fb > 0:
        print("Root must be bracketed in zbrent! Program exits.")
        exit()
    fc = fb
    for iter in range(1, ITMAX+1):
        if fb*fc > 0:
            c = a
            fc = fa
            d = b-a
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol1 = 2.0*EPS*abs(b)+0.5*self.ztol
        xm = 0.5*(c-b)
        if abs(xm) <= tol1 or fb == 0.0:
            fb = self.targetHdet(b)
            ii = 0
            while fb < 0:
                b = b-abs(tol1)
                fb = self.targetHdet(b)
                ii += 1
                if ii > iimax:
                    print("Cannot find a sigma for which Hdet >= 0. "\
                    "Program exits.")
                    exit()
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb/fa
            if a == c:
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0*xm*q-abs(tol1*q)
            min2 = abs(e*q)
            if 2.0*p < min(min1, min2):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            if xm > 0:
                b += abs(tol1)
            else:
                b -= abs(tol1)
        fb = self.targetHdet(b)
    print("Maximum number of iterations exceeded in zbrent. Program exits.")
    exit()

##########################################################################################

class beta_eqmom(eqmom) :

 def set_range(self,range):
     self.a = range[0]
     self.d = range[1] - range[0]
     
 def calc_secQuad(self, nsqn):    
   sw = zeros(self.nn*nsqn)
   sx = zeros(self.nn*nsqn)
   
   jsq = jacobi_secquad(nsqn)
   limit = zeros(2)
   limit[0] = self.a
   limit[1] = self.a+self.d
   for i in range(self.nn):
       jsq.calc_secquad(limit,self.weig[i], self.absc[i], self.sig)
       sw[i*nsqn:(i+1)*nsqn] = jsq.sw
       sx[i*nsqn:(i+1)*nsqn] = jsq.sx
       
   return sw , sx
      
 def calc_pdf(self, xp):    
     if(self.sig == 0):
         print "Sigma is zero, error!"
         exit()
              
     nxp = len(xp)
     pp = zeros(nxp)

     self.rescale_nodes()
     
     xn = (xp - self.a)/self.d
     for j in range(self.ncsn) :
         lam = self.absc[j]/self.sig
         mu = (1-self.absc[j])/self.sig
         for k in range(nxp) :
             if xn[k] != 0 and xn[k] != 1 :
                 lnf1 = (lam-1)*log(xn[k]) + (mu-1)*log(1.0 - xn[k]) - betaln(lam,mu) 
                 pp[k] = pp[k] +  self.weig[j]*exp(lnf1) 

     self.scaleback_nodes()        

     pp = pp/self.d
     return pp
   
 def calc_sigmax(self) :
     return 0.5

 def solve_1node(self) :
     
     self.weig[0] = 1.0
     self.absc[0] = self.moms[1] 
     self.sig    = (self.moms[1]*self.moms[1] - self.moms[2] )/(self.moms[2] - self.moms[1] )
     self.ncsn = 1

    
 def ms2mt(self, sig):
    """ Return mom calculated from mom_star."""
    m = zeros(self.nm)
    ms = self.ms
    
    m[0] = ms[0]
    m[1] = ms[1]
    m[2] = (ms[2]+sig*ms[1])/(1+sig)
    
    if self.nm >= 4:
        m[3] = ms[3]+3*sig*ms[2]+2*sig**2*ms[1]
        m[3] = m[3]/((1+sig)*(1+2*sig))
    if self.nm >= 5:
        m[4] = ms[4]+6*sig*ms[3]+11*sig**2*ms[2]+6*sig**3*ms[1]
        m[4] = m[4]/((1+sig)*(1+2*sig)*(1+3*sig))
        
    if self.nm >= 6:
        m[5] = ms[5]+10*sig*ms[4]+35*sig**2*ms[3]+50*sig**3*ms[2]\
        +24*sig**4*ms[1]
        m[5] = m[5]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig))
    if self.nm >= 7:
        m[6] = ms[6]+15*sig*ms[5]+85*sig**2*ms[4]+225*sig**3*ms[3]\
        +274*sig**4*ms[2]+120*sig**5*ms[1]
        m[6] = m[6]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig))
        
    if self.nm >= 8:
        m[7] = ms[7]+21*sig*ms[6]+175*sig**2*ms[5]+735*sig**3*ms[4]\
        +1624*sig**4*ms[3]+1764*sig**5*ms[2]+720*sig**6*ms[1]
        m[7] = m[7]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig))
    if self.nm >= 9:
        m[8] = ms[8]+28*sig*ms[7]+322*sig**2*ms[6]+1960*sig**3*ms[5]\
        +6769*sig**4*ms[4]+13132*sig**5*ms[3]+13068*sig**6*ms[2]\
        +5040*sig**7*ms[1]
        m[8] = m[8]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)\
                     *(1+7*sig))
    if self.nm >= 10:
        m[9] = ms[9]+36*sig*ms[8]+546*sig**2*ms[7]+4536*sig**3*ms[6]\
        +22449*sig**4*ms[5]+67284*sig**5*ms[4]+118124*sig**6*ms[3]\
        +109584*sig**7*ms[2]+40320*sig**8*ms[1]
        m[9] = m[9]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)\
                     *(1+7*sig)*(1+8*sig))
    if self.nm >= 11:
        m[10] = ms[10]+45*sig*ms[9]+870*sig**2*ms[8]+9450*sig**3*ms[7]\
        +63273*sig**4*ms[6]+269325*sig**5*ms[5]+723680*sig**6*ms[4]\
        +1172700*sig**7*ms[3]+1026576*sig**8*ms[2]+362880*sig**9*ms[1]
        m[10] = m[10]/((1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)\
                       *(1+6*sig)*(1+7*sig)*(1+8*sig)*(1+9*sig))
    if self.nm >= 12:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
        
    self.mt = m 
    

 def m2ms(self,sig):
   """ Return mom_star calculated from mom."""
   ms = zeros(self.nm)
   m = self.moms
        
   ms[0] = m[0]
   ms[1] = m[1]
   ms[2] = (1+sig)*m[2]-sig*ms[1]
   
   if self.nm >= 4:
        ms[3] = (1+sig)*(1+2*sig)*m[3]-3*sig*ms[2]-2*sig**2*ms[1]
   if self.nm >= 5:
        ms[4] = (1+sig)*(1+2*sig)*(1+3*sig)*m[4]-6*sig*ms[3]-11*sig**2*ms[2]\
        -6*sig**3*ms[1]
        
   if self.nm >= 6:
        ms[5] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*m[5]-10*sig*ms[4]\
        -35*sig**2*ms[3]-50*sig**3*ms[2]-24*sig**4*ms[1]
   if self.nm >= 7:
        ms[6] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*m[6]\
        -15*sig*ms[5]-85*sig**2*ms[4]-225*sig**3*ms[3]-274*sig**4*ms[2]\
        -120*sig**5*ms[1]
        
   if self.nm >= 8:
        ms[7] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)*m[7]\
        -21*sig*ms[6]-175*sig**2*ms[5]-735*sig**3*ms[4]-1624*sig**4*ms[3]\
        -1764*sig**5*ms[2]-720*sig**6*ms[1]
   if self.nm >= 9:
        ms[8] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)\
        *(1+7*sig)*m[8]-28*sig*ms[7]-322*sig**2*ms[6]-1960*sig**3*ms[5]\
        -6769*sig**4*ms[4]-13132*sig**5*ms[3]-13068*sig**6*ms[2]\
        -5040*sig**7*ms[1]
        
   if self.nm >= 10:
        ms[9] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)\
        *(1+7*sig)*(1+8*sig)*m[9]-36*sig*ms[8]-546*sig**2*ms[7]\
        -4536*sig**3*ms[6]-22449*sig**4*ms[5]-67284*sig**5*ms[4]\
        -118124*sig**6*ms[3]-109584*sig**7*ms[2]-40320*sig**8*ms[1]
   if self.nm >= 11:
        ms[10] = (1+sig)*(1+2*sig)*(1+3*sig)*(1+4*sig)*(1+5*sig)*(1+6*sig)\
        *(1+7*sig)*(1+8*sig)*(1+9*sig)*m[10]-45*sig*ms[9]-870*sig**2*ms[8]\
        -9450*sig**3*ms[7]-63273*sig**4*ms[6]-269325*sig**5*ms[5]\
        -723680*sig**6*ms[4]-1172700*sig**7*ms[3]-1026576*sig**8*ms[2]\
        -362880*sig**9*ms[1]
        
   if self.nm >= 12:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
    
   self.ms = ms


 def targetHdet(self,sig):
   """ Return the minimum Hankel determinants for beta EQMOM."""
   self.m2ms(sig)
   ms = self.ms
    
   minHdet = min(ms[0], ms[1])
   minHdet = min(minHdet, ms[0]-ms[1])
   if self.nn >= 2:
        H2 = array(([ms[0], ms[1]], [ms[1], ms[2]]))
        H3 = array(([ms[1], ms[2]], [ms[2], ms[3]]))
        H3bar = H2-H3
        minHdet = min(minHdet, ms[1]-ms[2])
        minHdet = min(minHdet, det(H2))
        minHdet = min(minHdet, det(H3))
        minHdet = min(minHdet, det(H3bar))
   if self.nn >= 3:
        H4 = array(([ms[0], ms[1], ms[2]], [ms[1], ms[2], ms[3]], \
        [ms[2], ms[3], ms[4]]))
        H5 = array(([ms[1], ms[2], ms[3]], [ms[2], ms[3], ms[4]], \
        [ms[3], ms[4], ms[5]]))
        H4bar = array(([ms[1]-ms[2], ms[2]-ms[3]], [ms[2]-ms[3], ms[3]-ms[4]]))
        H5bar = H4-H5
        minHdet = min(minHdet, det(H4))
        minHdet = min(minHdet, det(H5))
        minHdet = min(minHdet, det(H4bar))
        minHdet = min(minHdet, det(H5bar))
   if self.nn >= 4:
        H6 = array(([ms[0], ms[1], ms[2], ms[3]], \
        [ms[1], ms[2], ms[3], ms[4]], [ms[2], ms[3], ms[4], ms[5]], \
        [ms[3], ms[4], ms[5], ms[6]]))
        H7 = array(([ms[1], ms[2], ms[3], ms[4]], \
        [ms[2], ms[3], ms[4], ms[5]], [ms[3], ms[4], ms[5], ms[6]], \
        [ms[4], ms[5], ms[6], ms[7]]))
        H6bar = array(([ms[1]-ms[2], ms[2]-ms[3], ms[3]-ms[4]], \
        [ms[2]-ms[3], ms[3]-ms[4], ms[4]-ms[5]], \
        [ms[3]-ms[4], ms[4]-ms[5], ms[5]-ms[6]]))
        H7bar = H6-H7
        minHdet = min(minHdet, det(H6))
        minHdet = min(minHdet, det(H7))
        minHdet = min(minHdet, det(H6bar))
        minHdet = min(minHdet, det(H7bar))
   if self.nn >= 5:
        H8 = array(([ms[0], ms[1], ms[2], ms[3], ms[4]], \
        [ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]]))
        H9 = array(([ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]], \
        [ms[5], ms[6], ms[7], ms[8], ms[9]]))
        H8bar = array(([ms[1]-ms[2], ms[2]-ms[3], ms[3]-ms[4], ms[4]-ms[5]], \
        [ms[2]-ms[3], ms[3]-ms[4], ms[4]-ms[5], ms[5]-ms[6]], \
        [ms[3]-ms[4], ms[4]-ms[5], ms[5]-ms[6], ms[6]-ms[7]], \
        [ms[4]-ms[5], ms[5]-ms[6], ms[6]-ms[7], ms[7]-ms[8]]))
        H9bar = H8-H9
        minHdet = min(minHdet, det(H8))
        minHdet = min(minHdet, det(H9))
        minHdet = min(minHdet, det(H8bar))
        minHdet = min(minHdet, det(H9bar))
   if self.nn >= 6:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
   return minHdet     
     
##########################################################################################     

class gamma_eqmom(eqmom) :

 def set_range(self,range):
     self.a = range[0]
     self.d = 1.0
     
 def calc_secQuad(self,nsqn):    
   sw = zeros(self.nn*nsqn)
   sx = zeros(self.nn*nsqn)
   
   lsq = laguerre_secquad(nsqn)
   limit = zeros(2)
   limit[0] = self.a
   limit[1] = self.a+self.d
   for i in range(self.nn):
       lsq.calc_secquad(limit,self.weig[i], self.absc[i], self.sig)
       sw[i*nsqn:(i+1)*nsqn] = lsq.sw
       sx[i*nsqn:(i+1)*nsqn] = lsq.sx
       
   return sw , sx

 def calc_pdf(self, xp):    
     if(self.sig == 0):
         print "Sigma is zero, error!"
         exit()
         
     nxp = len(xp)
     pp = zeros(nxp)

     self.rescale_nodes()
     xn = xp - self.a
     for j in range(self.ncsn) :
         lam = self.absc[j]/self.sig 
         for k in range(nxp) :
             if xn[k] != 0  :
                 lnf1 = (lam- 1.0)*log(xn[k]) - xn[k]/self.sig - lam*log(self.sig) - gammaln(lam) 
                 pp[k] = pp[k] +  self.weig[j]*exp(lnf1) 
                
     self.scaleback_nodes()    
     return pp
     
 def calc_sigmax(self) :
     return 0.5
    

 def solve_1node(self) :
     
     self.weig[0] = 1.0
     self.absc[0] = self.moms[1] 
     self.sig = self.moms[2]/self.moms[1] - self.moms[1]
     self.ncsn = 1
     
 def ms2mt(self, sig):
    """ Return mom calculated from mom_star."""
    m = zeros(self.nm)
    ms = self.ms
    nmom = self.nm
    
    m[0] = ms[0]
    m[1] = ms[1]
    m[2] = ms[2]+sig*ms[1]
    if nmom >= 4:
        m[3] = ms[3]+3*sig*ms[2]+2*sig**2*ms[1]
    if nmom >= 5:
        m[4] = ms[4]+6*sig*ms[3]+11*sig**2*ms[2]+6*sig**3*ms[1]
    if nmom >= 6:
        m[5] = ms[5]+10*sig*ms[4]+35*sig**2*ms[3]+50*sig**3*ms[2]\
        +24*sig**4*ms[1]
    if nmom >= 7:
        m[6] = ms[6]+15*sig*ms[5]+85*sig**2*ms[4]+225*sig**3*ms[3]\
        +274*sig**4*ms[2]+120*sig**5*ms[1]
    if nmom >= 8:
        m[7] = ms[7]+21*sig*ms[6]+175*sig**2*ms[5]+735*sig**3*ms[4]\
        +1624*sig**4*ms[3]+1764*sig**5*ms[2]+720*sig**6*ms[1]
    if nmom >= 9:
        m[8] = ms[8]+28*sig*ms[7]+322*sig**2*ms[6]+1960*sig**3*ms[5]\
        +6769*sig**4*ms[4]+13132*sig**5*ms[3]+13068*sig**6*ms[2]\
        +5040*sig**7*ms[1]
    if nmom >= 10:
        m[9] = ms[9]+36*sig*ms[8]+546*sig**2*ms[7]+4536*sig**3*ms[6]\
        +22449*sig**4*ms[5]+67284*sig**5*ms[4]+118124*sig**6*ms[3]\
        +109584*sig**7*ms[2]+40320*sig**8*ms[1]
    if nmom >= 11:
        m[10] = ms[10]+45*sig*ms[9]+870*sig**2*ms[8]+9450*sig**3*ms[7]\
        +63273*sig**4*ms[6]+269325*sig**5*ms[5]+723680*sig**6*ms[4]\
        +1172700*sig**7*ms[3]+1026576*sig**8*ms[2]+362880*sig**9*ms[1]
    if nmom >= 12:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
        
    self.mt = m 
    

 def m2ms(self,sig):
    """ Return mom_star calculated from mom."""
    ms = zeros(self.nm)
    m = self.moms
    nmom = self.nm
    
    ms[0] = m[0]
    ms[1] = m[1]
    if nmom >= 3:
        ms[2] = m[2]-sig*ms[1]
    if nmom >= 4:
        ms[3] = m[3]-3*sig*ms[2]-2*sig**2*ms[1]
    if nmom >= 5:
        ms[4] = m[4]-6*sig*ms[3]-11*sig**2*ms[2]-6*sig**3*ms[1]
    if nmom >= 6:
        ms[5] = m[5]-10*sig*ms[4]-35*sig**2*ms[3]-50*sig**3*ms[2]\
        -24*sig**4*ms[1]
    if nmom >= 7:
        ms[6] = m[6]-15*sig*ms[5]-85*sig**2*ms[4]-225*sig**3*ms[3]\
        -274*sig**4*ms[2]-120*sig**5*ms[1]
    if nmom >= 8:
        ms[7] = m[7]-21*sig*ms[6]-175*sig**2*ms[5]-735*sig**3*ms[4]\
        -1624*sig**4*ms[3]-1764*sig**5*ms[2]-720*sig**6*ms[1]
    if nmom >= 9:
        ms[8] = m[8]-28*sig*ms[7]-322*sig**2*ms[6]-1960*sig**3*ms[5]\
        -6769*sig**4*ms[4]-13132*sig**5*ms[3]-13068*sig**6*ms[2]\
        -5040*sig**7*ms[1]
    if nmom >= 10:
        ms[9] = m[9]-36*sig*ms[8]-546*sig**2*ms[7]-4536*sig**3*ms[6]\
        -22449*sig**4*ms[5]-67284*sig**5*ms[4]-118124*sig**6*ms[3]\
        -109584*sig**7*ms[2]-40320*sig**8*ms[1]
    if nmom >= 11:
        ms[10] = m[10]-45*sig*ms[9]-870*sig**2*ms[8]-9450*sig**3*ms[7]\
        -63273*sig**4*ms[6]-269325*sig**5*ms[5]-723680*sig**6*ms[4]\
        -1172700*sig**7*ms[3]-1026576*sig**8*ms[2]-362880*sig**9*ms[1]
    if nmom >= 12:
        print("Too many EQMOM nodes try to be used. Program exits")
        exit()
    
    self.ms = ms


 def targetHdet(self,sig):
    """ Return the minimum Hankel determinants for gamma EQMOM."""
    self.m2ms(sig)
    ms = self.ms
    n = self.nn
    
    H00 = ms[0]
    H10 = ms[1]
    minHdet = min(H00, H10)
    if n >= 2:
        H01 = array(([ms[0], ms[1]], [ms[1], ms[2]]))
        H11 = array(([ms[1], ms[2]], [ms[2], ms[3]]))
        minHdet = min(minHdet, det(H01))
        minHdet = min(minHdet, det(H11))
    if n >= 3:
        H02 = array(([ms[0], ms[1], ms[2]], [ms[1], ms[2], ms[3]], \
        [ms[2], ms[3], ms[4]]))
        H12 = array(([ms[1], ms[2], ms[3]], [ms[2], ms[3], ms[4]], \
        [ms[3], ms[4], ms[5]]))
        minHdet = min(minHdet, det(H02))
        minHdet = min(minHdet, det(H12))
    if n >= 4:
        H03 = array(([ms[0], ms[1], ms[2], ms[3]], \
        [ms[1], ms[2], ms[3], ms[4]], [ms[2], ms[3], ms[4], ms[5]], \
        [ms[3], ms[4], ms[5], ms[6]]))
        H13 = array(([ms[1], ms[2], ms[3], ms[4]], \
        [ms[2], ms[3], ms[4], ms[5]], [ms[3], ms[4], ms[5], ms[6]], \
        [ms[4], ms[5], ms[6], ms[7]]))
        minHdet = min(minHdet, det(H03))
        minHdet = min(minHdet, det(H13))
    if n >= 5:
        H04 = array(([ms[0], ms[1], ms[2], ms[3], ms[4]], \
        [ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]]))
        H14 = array(([ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]], \
        [ms[5], ms[6], ms[7], ms[8], ms[9]]))
        minHdet = min(minHdet, det(H04))
        minHdet = min(minHdet, det(H14))
    if n >= 6:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
    return minHdet

##########################################################################################

class lognorm_eqmom(eqmom) :

 def set_range(self,range):
     if(range[0] > 0) : print "range[0] > 0, not supported" 
     self.a = 0.0
     self.d = 1.0
      
 def rescale_nodes(self):
    self.m0 = sum(self.weig)
    self.weig = self.weig / self.m0

 
 def scaleback_nodes(self):
    self.weig = self.weig*self.m0

 def calc_secQuad(self,nsqn):    
   sw = zeros(self.nn*nsqn)
   sx = zeros(self.nn*nsqn)
   
   ssq = stieltjes_secquad(nsqn)
   limit = zeros(2)
   limit[0] = self.a
   limit[1] = self.a+self.d
   for i in range(self.nn):
       ssq.calc_secquad(limit,self.weig[i], self.absc[i], self.sig)
       sw[i*nsqn:(i+1)*nsqn] = ssq.sw
       sx[i*nsqn:(i+1)*nsqn] = ssq.sx
       
   return sw , sx
   
 def calc_pdf(self, xp):   
     
     if(self.sig == 0):
         print "Sigma is zero, error!"
         exit()
         
     nxp = len(xp)
     pp = zeros(nxp)

     xn = xp - self.a
     for j in range(self.ncsn) :
         for k in range(nxp) :
             b = self.weig[j]/self.sig/sqrt(2*pi)
             if xn[k] != 0  :
                 t = (log(xn[k])-log(self.absc[j]))/self.sig
                 pp[k] = pp[k] +  b/xn[k]*exp(-t**2/2.0)       
     return pp

 def calc_sigmax(self) :
     
   z = sqrt(self.moms[0]*self.moms[2])/self.moms[1];
 
   sigmax = sqrt(2*log(z))
   sigmaxx = sigmax 
   
   if (self.nm > 3 ) :
	sigmax = sqrt(abs(self.moms[3]/(2*self.moms[1]))) 
	sigmaxx = max(sigmax, sigmaxx) ;
	
   if (self.nm  > 4 ) :
	sigmax = (self.moms[4]/(6*self.moms[1] ))**(1.0/3.0)
	sigmaxx = max(sigmax, sigmaxx) 
   
   return sigmaxx ;

 def solve_1node(self) :
     
     self.weig[0] = 1.0
     self.absc[0] = self.moms[1]**2/sqrt(self.moms[2])
     self.sig = sqrt(2.0*log(sqrt(self.moms[2])/self.moms[1]))


     self.ncsn = 1
     
 def ms2mt(self, sig):
   """ Return mom calculated from mom_star."""
   z = exp(0.5*(sig**2))

   for i in range(0,self.nm) :
	 self.mt[i] = pow(z,i*i)*self.ms[i]
    

 def m2ms(self,sig):
   """ Return mom_star calculated from mom."""
   zRev = 1.0/exp(0.5*(sig**2));

   for i in range(0,self.nm) :
	 self.ms[i] = pow(zRev,i*i)*self.moms[i]


# def targetHdet(self,sig):
#   """ Return the minimum Hankel determinants for lognorm EQMOM."""
#   self.m2ms(sig)
#
#   minHdet = 1e10
#   
#   H1 = zeros((self.nn, self.nn))
#   H2 = zeros((self.nn, self.nn))
#
#   for i in range(0,self.nn) :
#	for j in range(0, self.nn):
#         H1[i][j] = self.ms[i+j]
#         H2[i][j] = self.ms[i+j+1]
#    
#   for i in range(1,self.nn+1) :
#	minHdet = min(minHdet,det(H1,i))
#	if(minHdet < 0):
#         return minHdet
#	minHdet = min(minHdet,det(H2,i))
#	if(minHdet < 0) :
#         return minHdet
#
#   return minHdet

 def targetHdet(self,sig):
    """ Return the minimum Hankel determinants for gamma EQMOM."""
    self.m2ms(sig)
    ms = self.ms
    n = self.nn
    
    H00 = ms[0]
    H10 = ms[1]
    minHdet = min(H00, H10)
    if n >= 2:
        H01 = array(([ms[0], ms[1]], [ms[1], ms[2]]))
        H11 = array(([ms[1], ms[2]], [ms[2], ms[3]]))
        minHdet = min(minHdet, det(H01))
        minHdet = min(minHdet, det(H11))
    if n >= 3:
        H02 = array(([ms[0], ms[1], ms[2]], [ms[1], ms[2], ms[3]], \
        [ms[2], ms[3], ms[4]]))
        H12 = array(([ms[1], ms[2], ms[3]], [ms[2], ms[3], ms[4]], \
        [ms[3], ms[4], ms[5]]))
        minHdet = min(minHdet, det(H02))
        minHdet = min(minHdet, det(H12))
    if n >= 4:
        H03 = array(([ms[0], ms[1], ms[2], ms[3]], \
        [ms[1], ms[2], ms[3], ms[4]], [ms[2], ms[3], ms[4], ms[5]], \
        [ms[3], ms[4], ms[5], ms[6]]))
        H13 = array(([ms[1], ms[2], ms[3], ms[4]], \
        [ms[2], ms[3], ms[4], ms[5]], [ms[3], ms[4], ms[5], ms[6]], \
        [ms[4], ms[5], ms[6], ms[7]]))
        minHdet = min(minHdet, det(H03))
        minHdet = min(minHdet, det(H13))
    if n >= 5:
        H04 = array(([ms[0], ms[1], ms[2], ms[3], ms[4]], \
        [ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]]))
        H14 = array(([ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]], \
        [ms[5], ms[6], ms[7], ms[8], ms[9]]))
        minHdet = min(minHdet, det(H04))
        minHdet = min(minHdet, det(H14))
    if n >= 6:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
    return minHdet

##########################################################################################

class gaussian_eqmom(eqmom) :

 def __init__(self,nn):
   eqmom.__init__(self,nn)
   if self.nm >5:
        print("Too many EQMOM nodes. Program exits.")
        exit()
        
 def set_range(self,range):
     self.a = 0.0
     self.d = 1.0
     
 def calc_secQuad(self,nsqn):    
   sw = zeros(self.nn*nsqn)
   sx = zeros(self.nn*nsqn)
   
   hsq = hermite_secquad(nsqn)
   limit = zeros(2)   
   limit[0] = self.a
   limit[1] = self.a+self.d
   for i in range(self.nn):
       hsq.calc_secquad(limit,self.weig[i], self.absc[i], self.sig)
       sw[i*nsqn:(i+1)*nsqn] = hsq.sw
       sx[i*nsqn:(i+1)*nsqn] = hsq.sx
       
   return sw , sx

 def calc_pdf(self, xp):    
     
     if(self.sig == 0):
         print "Sigma is zero, error!"
         exit()
         
     nxp = len(xp)
     pp = zeros(nxp)
     for j in range(self.ncsn) :
         b = self.weig[j]/self.sig/sqrt(2*pi)
         for k in range(nxp) :
             t = (xp[k]-self.absc[j])/self.sig
             pp[k] = pp[k] +  b*exp(-t**2/2.0)  
             
     return pp    

 def calc_sigmax(self) :
   return 0.5
    
 def ms2mt(self, sig):
    """ Return mom calculated from mom_star."""
    
    m = zeros(self.nm)
    ms = self.ms
    sig2 = sig**2
    m[0] = ms[0]
    m[1] = ms[1]
    m[2] = ms[2]+sig2*ms[0]
    if(self.nn == 2):
        m[3] = ms[3]+3*sig2*ms[1]
        m[4] = ms[4]+6*sig2*ms[2]+ 3*sig2**2*ms[0]

    self.mt = m 

 def solve_1node(self) :   
     self.weig[0] = 1.0
     self.absc[0] = self.moms[1] 
     self.sig    = sqrt(self.moms[2] - self.moms[1] **2)
     self.ncsn = 1


 def m2ms(self,sig):
   """ Return mom_star calculated from mom."""
   
   m = self.moms
   sig2 = sig**2       
   
   self.ms[0] = m[0]
   self.ms[1] = m[1]
   self.ms[2] = m[2] - sig2*m[0]
   if(self.nn == 2):
       self.ms[3] = m[3] - 3*sig2*m[1]
       self.ms[4] = m[4] - 6*sig2*m[2] - 3*sig2**2*m[0]
        

 def solve_sigma(self):
    # Five moments are needed for 2-node Gaussian EQMOM.
    m = self.moms
            
    m = m/m[0]

    # Compute central moments e, q, and eta.
    e = (m[0]*m[2]-m[1]**2)/m[0]**2
    q = m[3]/m[0]-(m[1]/m[0])**3-3*m[1]/m[0]*e
    eta = m[4]/m[0] - 4*m[3]*m[1]/(m[0]*m[0]) + 6*m[2]*m[1]*m[1]/pow(m[0],3) - 3*pow(m[1]/m[0],4)
    
    if e <= 1e-8:
        self.weig[0] = m[0]
        self.absc[0] = m[1]/m[0]
        self.sig = 0
        self.ncsn = 1
        return 

    c1 = (eta/e**2-3)/6
    c2 = q**2/(4*e**3)
    tmp = sqrt(c1**3+c2**2)
    c3 = pow(c2 + tmp,1.0/3.0)
    sc = c3-c1/c3
    sig1 = sc.real
    sig1 = max(sig1, 0)
    sig1 = min(sig1, 1)   
    sig1 = (1-sig1)*e
    sig = sqrt(sig1)
    
    self.update_nodes(sig)
    
#    
#    macheps = spacing(1)
#    # Check if system of moments is well-defined.
#    if e <= 0.0:
#        print("The system is not well-defined, e <= 0.0")
#        exit() 
#    elif abs(q) < macheps:
#        if eta > 3*e**2:
#            self.sig = 0
#            self.ncsn = 0
#            print("The system is not well-defined, eta > 3e^2 when q = 0.")
#            exit()  
#        elif abs(eta-3*e**2) <= macheps:
#            self.absc[0] = m[1]/m[0]
#            self.weig[0] = m[0]
#            self.sig = sqrt(e)
#            self.ncsn = 1
#            return 
#        elif eta >= e**2:
#            x0 = ((3*e**2-eta)/2)**(0.25)
#            self.absc[0] = -x0+m[1]/m[0]
#            self.absc[1] = x0+m[1]/m[0]
#            self.weig[0] = 0.5*m[0]
#            self.weig[1] = self.weig[0]
#            self.sig = sqrt(e-x0**2)
#            self.ncsn = 2
#            return
#        else:
#            self.sig = 0
#            self.ncsn  = 0
#            print("The system is not well-defined, eta < e^2 when q = 0.")
#            exit()  
#            
#    elif eta <= e**2+q**2/e:
#        self.sig = 0
#        self.ncsn  = 0
#        print("The system is not well-defined, eta <= e^2 + q^2/e.")
#        exit()  
#    # Return one node if e is small.
#    if e <= 1e-8:
#        self.weig[0] = m[0]
#        self.absc[0] = m[1]/m[0]
#        self.sig = 0
#        self.ncsn = 1
#        return 
#    # Calculate sigma**2.
#    c1 = (eta/e**2-3)/6
#    c2 = q**2/(4*e**3)
#    tmp = sqrt(c1**3+c2**2)
#    c3 = (tmp.real+c2)**(1/3)
#    sig1 = e*(1-c3+c1/c3)
#    sig1 = max(sig1, 0)
#    sig1 = min(sig1, 1)
#    # Return one node if sigma**2 = e.
#    if abs(sig1-e) < macheps:
#        self.weig[0] = m[0]
#        self.absc[0] = m[1]/m[0]
#        self.sig = sqrt(e)
#        self.ncsn = 1
#        return 
#    # Calculate weights and nodes.
#    x0 = (q/2)/sqrt(q**2+4*e**3)
#    self.weig[0] = (0.5+x0)*m[0]
#    self.weig[1] = (0.5-x0)*m[0]
#    self.absc[0] = m[1]/m[0]-sqrt(self.weig[1]/self.weig[0]*e)
#    self.absc[1] = m[1]/m[0]+sqrt(self.weig[0]/self.weig[1]*e)
#    self.sig = sqrt(sig1)
#    self.ncsn = 2
    
    
    return
    
eqmoms['Beta'] = beta_eqmom
eqmoms['Gamma'] = gamma_eqmom
eqmoms['Gaussian'] = gaussian_eqmom
eqmoms['Lognorm'] = lognorm_eqmom