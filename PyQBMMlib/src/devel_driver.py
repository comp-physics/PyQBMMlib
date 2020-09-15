from qbmm_manager import *
import sys
sys.path.append('../utils/')
from stats_util import *

def array_pretty_print(message, array_name, array):

    message += array_name + ' = [{:s}]'    
    print( message.format( ', '.join( [ '{:.4e}'.format(a) for a in array ] ) ) )

if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )
    
    config = {}
    config['qbmm'] = {}
    config['qbmm']['governing_dynamics'] = ' dx + x = 1'

    config['qbmm']['num_internal_coords']  = 1
    config['qbmm']['num_quadrature_nodes'] = 4
    config['qbmm']['method']       = 'qmom'
    config['qbmm']['adaptive']     = False
    config['qbmm']['max_skewness'] = 30

    qbmm_mgr    = qbmm_manager( config )
    num_moments = qbmm_mgr.num_moments
    
    mu    = 1.0
    sigma = 0.1
    moments = raw_gaussian_moments_univar( num_moments, mu, sigma )
    indices = qbmm_mgr.indices

    message = 'devel_driver: main: '
    array_pretty_print( message, 'moments', moments )
    array_pretty_print( message, 'indices', indices )

    ###
    ### [ecg] The following workflow will be encapsulated in a single
    ### qbmm_mgr function called compute_rhs. This function will be
    ### called from advancer, and will take as inputs only the
    ### moments. Here, the steps are laid out explicitly for
    ### development purposes, hence the name of the script
    ### (duh).
    ###
    
    weights, abscissas = qbmm_mgr.moment_invert( moments )
    array_pretty_print( message, 'weights', weights )
    array_pretty_print( message, 'abscissas', abscissas )
    
    quadrature = qbmm_mgr.quadrature( weights, abscissas )
    print quadrature
    
    exit
