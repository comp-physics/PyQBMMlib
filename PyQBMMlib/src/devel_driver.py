from advancer import *
import sys
sys.path.append('../utils/')
from stats_util import *
from euler_util import *
from jets_rfox_util import *
from pretty_print_util import *
import cProfile

def flow_example():
    
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

    moments_left, moments_right =  jet_rfox_initialize_moments( qbmm_mgr )

    xi_left,  wts_left  = qbmm_mgr.moment_invert( moments_left,  indices )
    xi_right, wts_right = qbmm_mgr.moment_invert( moments_right, indices )

    print(wts_left)
    print(wts_right)
    
    print(xi_left)
    print(xi_right)

    flux = moment_fluxes( indices, wts_left, wts_right, xi_left, xi_right )

    #print(flux)
    
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
    config['advancer']['output_dir']      = './'
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
    
    # config['qbmm']['governing_dynamics'] = ' 4*x - 2*xdot**2'
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
    config['advancer']['error_tol']  = 1.0e-10
    config['advancer']['num_steps']  = 10000
    config['advancer']['num_steps_print'] = 1000
    config['advancer']['num_steps_write'] = 1000
    config['advancer']['output_dir']      = 'D/'
    config['advancer']['output_id']       = 'example_2D'
    config['advancer']['write_to']        = 'txt'
    
    advancer = time_advancer( config )

    # Initialize condition
    mu1    = 1.0
    mu2    = 0.0
    sigma1 = 0.1
    sigma2 = 0.2

    advancer.initialize_state_gaussian_bivar( mu1, mu2, sigma1, sigma2 )

    # Run
    advancer.run()
    
    return


def advance_example2dp1():

    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}
    
    # config['qbmm']['governing_dynamics'] = ' 4*x - 2*xdot**2'
    config['qbmm']['governing_dynamics'] = ' - x - xdot'
    config['qbmm']['num_internal_coords']  = 2
    config['qbmm']['num_quadrature_nodes'] = 4
    config['qbmm']['method']               = 'chyqmom'
    config['qbmm']['adaptive']             = False
    config['qbmm']['max_skewness']         = 30

    config['advancer']['method']     = 'RK23'
    config['advancer']['time_step']  = 1.e-5
    config['advancer']['final_time'] = 15.
    config['advancer']['error_tol']  = 1.0e-5
    config['advancer']['num_steps']  = 10000
    config['advancer']['num_steps_print'] = 1
    config['advancer']['num_steps_write'] = 1
    config['advancer']['output_dir']      = './'
    config['advancer']['output_id']       = 'example_2D'
    config['advancer']['write_to']        = 'txt'
    
    advancer = time_advancer( config )

    # Initialize condition
    mu1    = 1.0
    mu2    = 0.0
    sigma1 = 0.1
    sigma2 = 0.2

    advancer.initialize_state_gaussian_bivar( mu1, mu2, sigma1, sigma2 )

    # Run
    advancer.run()
    
    return
    
def transport_terms_example():

    config = {}
    config['qbmm'] = {}
    config['qbmm']['governing_dynamics'] = ' 0.3*x + 0.6*xdot/(x**4.1)'

    config['qbmm']['num_internal_coords']  = 2
    config['qbmm']['num_quadrature_nodes'] = 2
    config['qbmm']['method']       = 'chyqmom'
    config['qbmm']['adaptive']     = False
    config['qbmm']['max_skewness'] = 30

    qbmm_mgr = qbmm_manager( config )

    return

def moments_workflow_example():

    config = {}
    config['qbmm'] = {}
    config['advancer'] = {}
    config['qbmm']['governing_dynamics'] = '1+3*x+xdot'

    config['qbmm']['num_internal_coords']  = 3
    config['qbmm']['num_quadrature_nodes'] = 27
    config['qbmm']['method']       = 'chyqmom'
    config['qbmm']['adaptive']     = False

    config['advancer']['method']     = 'RK23'
    config['advancer']['time_step']  = 1.e-5
    config['advancer']['final_time'] = 15.
    config['advancer']['error_tol']  = 1.0e-5
    config['advancer']['num_steps']  = 10000
    config['advancer']['num_steps_print'] = 1
    config['advancer']['num_steps_write'] = 1
    config['advancer']['output_dir']      = './'
    config['advancer']['output_id']       = 'example_2D'
    config['advancer']['write_to']        = 'txt'


    advancer = time_advancer( config )

    qbmm_mgr    = qbmm_manager( config )
    num_moments = qbmm_mgr.num_moments
    
    mu1    = 1.0
    mu2    = 1.0
    mu3    = 1.0
    sigma1 = 0.1
    sigma2 = 0.1
    sigma3 = 0.1
    # moments = raw_gaussian_moments_trivar( num_moments, mu1, mu2, mu3, sigma1, sigma2, sigma3 )
    advancer.initialize_state_gaussian_trivar( mu1, mu2, mu3, sigma1, sigma2, sigma3 )
    indices = qbmm_mgr.indices
    print(indices)

    # message = 'devel_driver: main: '
    # f_array_pretty_print( message, 'moments', moments )
    # i_array_pretty_print( message, 'indices', indices )

    abscissas, weights = qbmm_mgr.moment_invert( advancer.state, indices )
    # f_array_pretty_print( message, 'weights', weights )
    # f_array_pretty_print( message, 'abscissas', abscissas )
    
    # projected_moments = qbmm_mgr.projection( weights, abscissas, indices )
    # f_array_pretty_print( message, 'projected_moments', projected_moments )

    # recons_error = np.abs( moments - projected_moments ).max()
    # f_scalar_pretty_print( message, 'recons_error', recons_error )

    return
    
if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )

    case = 'example_2D'

    if case == 'example_1D':
        advance_example1d()
    elif case == 'example_2D':
        advance_example2d()
        # cProfile.run('advance_example2d()')
    elif case == 'test':
        moments_workflow_example()
    elif case == 'flow':
        flow_example()
        
    
    ###
    ### [ecg] The following workflow will be encapsulated in a single
    ### qbmm_mgr function called compute_rhs. This function will be
    ### called from advancer, and will take as inputs only the
    ### moments. Here, the steps are laid out explicitly for
    ### development purposes, hence the name of the script
    ### (duh).
    ###
    
    #abscissas, weights = qbmm_mgr.moment_invert( moments )
    #f_array_pretty_print( message, 'weights', weights )
    #f_array_pretty_print( message, 'abscissas', abscissas )
    
    #projected_moments = qbmm_mgr.projection( weights, abscissas, indices )
    #f_array_pretty_print( message, 'projected_moments', projected_moments )

    #recons_error = np.abs( moments - projected_moments ).max()
    #f_scalar_pretty_print( message, 'recons_error', recons_error )

    #if config['qbmm']['num_internal_coords']==1:
    #    # Example coef/exp for xdot = 4x - 2x^2
    #    c0 = symbols('c0')
    #    coef = [ 4*c0, -2*c0 ]
    #    exp  = [   c0,  1+c0 ]

    # Esteban To-do:
    #   Loop below is example of taking Euler time steps
    #   1. I think 'coef' and 'exp' can be moved to the 'config' dictionary
    #       since they are only computed once, maybe they shouldn't be 
    #       passed as functions every time compute_rhs is called?
    #   2. Hyperbolic 1D needs to be able to access max skewness for 3-node closure
    #       max_skewness = config['qbmm']['max_skewness']
    #       wasn't working when i tried it. Look at inversion.py

    ### [ecg]:
    ### 1. coef and exp will be member variables in qbmm_mgr, which will
    ###    determine what they are based on config option governing_dynamics
    ###    using sympy. That way, they will be computed upon construction
    ###    and ready to be used when compute_rhs is called.
    ### 2. Fixed. Note that the routines in inversion.py should not take config
    ###    inputs, and they are only meant to be called from inside qbmm_mgr.
    # dt = 0.1
    # mom = moments
    # for i in range(100):
    #     print('--Time step: ',i, ' --Moments: ',mom)
    #     abscissas, weights = qbmm_mgr.moment_invert( mom )
    #     rhs = qbmm_mgr.compute_rhs(coef,exp,indices,weights,abscissas)
    #     proj = qbmm_mgr.projection( weights, abscissas, indices)
    #     mom = proj + dt*rhs

    exit
