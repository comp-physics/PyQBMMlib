import scipy.stats as stats
import scipy.special as sc
import math
import numpy as np

def raw_gaussian_moments_univar(num_moments, mu, sigma):
    """
    This function returns raw 1D-Gaussian moments as a function of 
    mean (mu) and standard deviation (sigma)
    """
    moments = np.zeros( num_moments )
    for i_moment in range( num_moments ):
        moments[ i_moment ] = stats.norm.moment( i_moment, mu, sigma )
    return moments
        
def raw_gaussian_moments_bivar(indices, mu1, mu2, sig1, sig2):
    """
    This function returns raw 2D-Gaussian moments as a function of 
    means (mu_1,mu_2) and standard deviations (sigma_1,sigma_2)
    """

    num_moments = len(indices)
    moments = np.zeros( num_moments) 
    for i_moment in range( num_moments ):
        i = indices[i_moment][0]
        j = indices[i_moment][1]
        moments[ i_moment ] = (1./math.pi) * 2**((-4.+i+j)/2.) * \
          math.exp( -(mu1**2./(2. * sig1**2))-(mu2**2./(2 * sig2**2.)) ) * \
          sig1**(-1.+i) * sig2**(-1+j) * \
          ( \
            -math.sqrt(2.) * \
            (-1.+(-1.)**i) * \
            mu1 * \
            sc.gamma(1.+i/2.) * \
            sc.hyp1f1(1+i/2.,3./2.,mu1**2./(2. * sig1**2.))  \
            + (1.+(-1.)**i) * \
            sig1 * \
            sc.gamma((1.+i)/2.)  * \
            sc.hyp1f1((1.+i)/2.,1./2.,mu1**2./(2. * sig1**2.))
          ) * \
          (   \
            -math.sqrt(2.) * \
            (-1+(-1)**j) * \
            mu2 * \
            sc.gamma(1.+j/2.) * \
            sc.hyp1f1(1.+j/2.,3./2.,mu2**2./(2. * sig2**2.)) + \
            (1.+(-1)**j) * \
            sig2 * \
            sc.gamma((1.+j)/2.) * \
            sc.hyp1f1((1.+j)/2.,1./2.,mu2**2./(2. * sig2**2.)) \
          )
    return moments
