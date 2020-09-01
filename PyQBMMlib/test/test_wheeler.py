import sys
sys.path.append('../src/')
from qbmm_manager import *
import numpy.polynomial.hermite as hermite_poly

def gauss_hermite(num_nodes):
    """
    This function returns Gauss-Hermite abscissas & weights     
    from numpy's Hermite-polynomial engine
    """
    g_abs, g_wts = hermite_poly.hermgauss( num_nodes )
    return g_abs, g_wts
    
if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )
    
    ###
    ### Say hello
    print('test_wheeler: Testing Wheeler algorithm for moment inversion')
    
    ###
    ### Reference solution (from numpy)
    print('test_wheeler: Computing reference solution (from numpy)')

    num_nodes = 4
    sqrt_pi   = sqrt( np.pi )
    sqrt_two  = sqrt( 2.0 )
    g_abs, g_wts = gauss_hermite( num_nodes )
    g_abs *= sqrt_two
    g_wts /= sqrt_pi
    
    ###
    ### QBMM Configuration
    print('test_wheeler: Configuring and initializing qbmm')
    
    config = {}
    config['governing_dynamics']   = ' dx + x = 1'
    config['num_internal_coords']  = 1
    config['num_quadrature_nodes'] = 4
    config['method']       = 'qmom'
    config['adaptive']     = False
    config['max_skewness'] = 30

    ###
    ### QBMM
    qbmm_mgr = qbmm_manager( config )
    raw_gaussian_moments = np.array( [ 1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0 ] )
    my_abs, my_wts = qbmm_mgr.moment_invert( raw_gaussian_moments )

    ###
    ### Errors & Report
    diff_abs = my_abs - g_abs
    diff_wts = my_wts - g_wts

    error_abs = np.sqrt( np.dot( diff_abs, diff_abs ) )
    error_wts = np.sqrt( np.dot( diff_wts, diff_wts ) )

    print('test_wheeler: Reference solution:')
    print('\t abscissas = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in g_abs ] ) ) )
    print('\t weights   = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in g_wts ] ) ) )

    print('test_wheeler: QBMM solution:')
    print('\t abscissas = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in my_abs ] ) ) )
    print('\t weights   = [{:s}]'.format( ', '.join( [ '{:.4e}'.format(p) for p in my_wts ] ) ) )

    print('test_wheeler: Errors:')    
    print('\t abscissas: error = %.4E' % error_abs )
    print('\t weights:   error = %.4E' % error_wts )

    tol = 1.0e-14
    success = True
    if( error_abs > tol and error_wts > tol ):
        success = False

    if success == True:
        color = '\033[92m'
        print('test_wheeler: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_wheeler: ' + '\033[91m' + 'test failed ' + '\033[0m' + '... check Wheeler!')
        
    exit
