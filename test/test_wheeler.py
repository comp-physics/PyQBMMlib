import sys
sys.path.append('../src/')
from qbmm_manager import *
import numpy.polynomial.hermite as hermite_poly

def gauss_hermite(num_nodes):
    """
    This function returns Gauss-Hermite abscissas & weights     
    from numpy's Hermite-polynomial engine
    """
    h_abs, h_wts = hermite_poly.hermgauss( num_nodes )
    return h_abs, h_wts

def generate_gaussian_moments(mu, sigma):
    """
    This function returns the first 8 Gaussian moments,
    per Marchisio & Fox, pg. 52
    """
    moments = np.zeros( 8 )
    moments[0] = 1.0
    moments[1] = mu
    moments[2] = mu*mu + sigma*sigma
    moments[3] = mu**3.0 +  3.0 * mu * sigma**2.0
    moments[4] = mu**4.0 +  6.0 * (mu**2.0) * (sigma**2.0) +   3.0 * sigma**4.0
    moments[5] = mu**5.0 + 10.0 * (mu**3.0) * (sigma**2.0) +  15.0 * mu * sigma**4.0
    moments[6] = mu**6.0 + 15.0 * (mu**4.0) * (sigma**2.0) +  45.0 * (mu**2.0) * (sigma**4.0) + 15.0 * sigma**6.0
    moments[7] = mu**7.0 + 21.0 * (mu**5.0) * (sigma**2.0) + 105.0 * (mu**3.0) * (sigma**4.0) + 105.0 * mu * sigma**6.0
    
    return moments

def test_wheeler(test, mu, sigma, qbmm_mgr, tol):
    """
    This function tests QBMM Wheeler inversion by comparing
    against numpy's Gauss-Hermite for given mu and sigma
    """
    ###
    ### Reference solution
    num_nodes = 4
    sqrt_pi   = np.sqrt( np.pi )
    sqrt_two  = np.sqrt( 2.0 )
    h_abs, h_wts = gauss_hermite( num_nodes )
    g_abs = sqrt_two * sigma * h_abs + mu
    g_wts = h_wts / sqrt_pi
    
    ###
    ### QBMM
    moments = generate_gaussian_moments( mu, sigma )
    my_abs, my_wts = qbmm_mgr.moment_invert( moments )

    ###
    ### Errors & Report

    error_abs = np.linalg.norm( my_abs - g_abs )
    error_wts = np.linalg.norm( my_wts - g_wts )

    success = ( error_abs < tol and error_wts < tol )

    if not success:
        print('test_wheeler: Test(%i): Moments = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in moments ] ) ) % test ) 
        print('test_wheeler: Test(%i): Reference solution:' % test)
        print('\t abscissas = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in g_abs ] ) ) )
        print('\t weights   = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in g_wts ] ) ) )

        print('test_wheeler: Test(%i): QBMM solution:' % test)
        print('\t abscissas = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in my_abs ] ) ) )
        print('\t weights   = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in my_wts ] ) ) )

        print('test_wheeler: Test(%i): Errors:' % test)    
        print('\t abscissas: error = %.4E' % error_abs )
        print('\t weights:   error = %.4E' % error_wts )

    return success 

if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )
    
    ###
    ### Say hello
    print('test_wheeler: Testing Wheeler algorithm for moment inversion')
    
    ###
    ### QBMM Configuration
    print('test_wheeler: Configuring and initializing qbmm')
   
    nnodes = [ 1, 2, 3, 4, 5 ]
    for n in nnodes:
        config = {}
        config['qbmm'] = {}
        config['qbmm']['governing_dynamics']   = '4*x - 2*x**2'
        config['qbmm']['num_internal_coords']  = 1
        config['qbmm']['num_quadrature_nodes'] = n
        config['qbmm']['method']       = 'qmom'

        ###
        ### QBMM
        qbmm_mgr = qbmm_manager( config )

        ###
        ### Tests

        # Anticipate success
        tol = 1.0e-13  # Why?
        success = True
        
        # Test 1
        mu    = 0.0
        sigma = 1.0
        success *= test_wheeler( 1, mu, sigma, qbmm_mgr, tol )
        
        ### Test 2
        mu = 5.0
        success *= test_wheeler( 2, mu, sigma, qbmm_mgr, tol )

    if success == True:
        print('test_wheeler: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_wheeler: ' + '\033[91m' + 'test failed ' + '\033[0m')
        
    exit()
