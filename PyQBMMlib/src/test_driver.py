from qbmm_manager import *

if __name__ == '__main__':
    
    config = {}
    config['governing_dynamics']   = ' dx + x = 1'
    config['num_internal_coords']  = 1
    config['num_quadrature_nodes'] = 3
    config['method']       = 'hyqmom'
    config['max_skewness'] = 30

    qbmm_mgr = qbmm_manager( config )

    moments = np.zeros( config['num_quadrature_nodes'] )
    weights, abscissas = qbmm_mgr.moment_invert( moments )

    exit
