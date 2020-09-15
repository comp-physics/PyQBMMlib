import scipy.stats as stats
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
        
def raw_gaussian_moments_bivar(num_moments, mu_1, mu_2, sigma_1, sigma_2, index_1, index_2):
    """
    This function returns raw 2D-Gaussian moments as a function of 
    means (mu_1,mu_2) and standard deviations (sigma_1,sigma_2)
    """
    print('stats: raw_gaussian_moments_bivar: Error: not implemented yet')
    return []
