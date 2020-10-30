from advancer import *
import sys
sys.path.append('../utils/')
from stats_util import *
from euler_util import *
from jets_util import *
from pretty_print_util import *
import cProfile

def flow_example():
    
    # In development
    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}

    config['qbmm']['flow'] = True
    config['qbmm']['governing_dynamics']   = ''
    config['qbmm']['num_internal_coords']  = 3
    config['qbmm']['num_quadrature_nodes'] = 27
    config['qbmm']['method']               = 'chyqmom'
    config['qbmm']['adaptive']             = False
    config['qbmm']['max_skewness']         = 30

    qbmm_mgr = qbmm_manager( config )
    indices  = qbmm_mgr.indices

    # Initial condition
    # mu1    = 1.0
    # mu2    = 1.0
    # mu3    = 1.0
    # sigma1 = 0.1
    # sigma2 = 0.1
    # sigma3 = 0.1
    # moments = raw_gaussian_moments_trivar( indices, mu1, mu2, mu3,
    #                                        sigma1, sigma2, sigma3 )

    moments_left, moments_right =  jet_initialize_moments( qbmm_mgr )

    xi_left,  wts_left  = qbmm_mgr.moment_invert( moments_left,  indices )
    xi_right, wts_right = qbmm_mgr.moment_invert( moments_right, indices )

    print(wts_left)
    print(wts_right)
    
    print(xi_left)
    print(xi_right)

    flux = moment_fluxes( indices, wts_left, wts_right, xi_left, xi_right )

    print(flux)
    
    return

def advance_example1d():

    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}
    
    config['qbmm']['flow'] = False
    config['qbmm']['governing_dynamics'] = '4*x - 2*x**2'
    config['qbmm']['num_internal_coords']  = 1
    config['qbmm']['num_quadrature_nodes'] = 2
    config['qbmm']['method']               = 'hyqmom'
    config['qbmm']['adaptive']             = False
    config['qbmm']['max_skewness']         = 30

    config['advancer']['method']     = 'RK23'
    config['advancer']['time_step']  = 0.1
    config['advancer']['final_time'] = 2.0
    config['advancer']['error_tol']  = 1.0e-5
    config['advancer']['num_steps']  = 10
    config['advancer']['num_steps_print'] = 1
    config['advancer']['num_steps_write'] = 1
    config['advancer']['output_dir']      = 'D/'
    config['advancer']['output_id']       = 'example_1D'
    config['advancer']['write_to']        = 'txt'
    
    advancer = time_advancer( config )

    # Initialize condition
    mu    = 1.0
    sigma = 0.1
    advancer.initialize_state_gaussian_univar( mu, sigma )

    # Run
    advancer.run()
    
    return


def advance_example2d():

    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}
    
    config['qbmm']['flow'] = False
    config['qbmm']['governing_dynamics'] = ' - x - xdot'
    config['qbmm']['num_internal_coords']  = 2
    config['qbmm']['num_quadrature_nodes'] = 4
    config['qbmm']['method']               = 'chyqmom'
    config['qbmm']['adaptive']             = False
    config['qbmm']['max_skewness']         = 30

    config['advancer']['method']     = 'RK23'
    config['advancer']['time_step']  = 1.e-5
    config['advancer']['final_time'] = 30.
    config['advancer']['error_tol']  = 1.0e-9
    config['advancer']['num_steps']  = 20000
    config['advancer']['num_steps_print'] = 1000
    config['advancer']['num_steps_write'] = 1000
    config['advancer']['output_dir']      = 'D/'
    config['advancer']['output_id']       = 'example_2D'
    config['advancer']['write_to']        = 'txt'
    
    advancer = time_advancer( config )

    # Initial condition
    mu1    = 1.0
    mu2    = 0.0
    sigma1 = 0.1
    sigma2 = 0.2

    advancer.initialize_state_gaussian_bivar( mu1, mu2, sigma1, sigma2 )

    # Run
    advancer.run()
    
    return


def advance_example2dp1():
    # In development!
    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}
    
    config['qbmm']['flow'] = False
    config['qbmm']['governing_dynamics'] = ' - x - xdot - r0'
    config['qbmm']['num_internal_coords']  = 2
    config['qbmm']['num_quadrature_nodes'] = 4
    config['qbmm']['method']               = 'chyqmom'
    config['qbmm']['adaptive']             = False
    config['qbmm']['max_skewness']         = 30
    config['qbmm']['polydisperse']  = True
    config['qbmm']['num_poly_nodes']  = 5
    config['qbmm']['poly_symbol']  = 'r0'

    config['advancer']['method']     = 'RK23'
    config['advancer']['time_step']  = 1.e-5
    config['advancer']['final_time'] = 30.
    config['advancer']['error_tol']  = 1.0e-5
    config['advancer']['num_steps']  = 20000
    config['advancer']['num_steps_print'] = 1000
    config['advancer']['num_steps_write'] = 1000
    config['advancer']['output_dir']      = 'D/'
    config['advancer']['output_id']       = 'example_2D'
    config['advancer']['write_to']        = 'txt'
    
    advancer = time_advancer( config )

    # Initial condition
    mu1    = 1.0
    mu2    = 0.0
    mu3    = 0.1
    sigma1 = 0.1
    sigma2 = 0.2

    advancer.initialize_state_gaussian_bivar( mu1, mu2, sigma1, sigma2 )

    advancer.run()
    
    return
    
if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )

    case = 'example_2D'

    if case == 'example_1D':
        advance_example1d()
    elif case == 'example_2D':
        advance_example2d()
    elif case == 'flow':
        flow_example()

    exit
