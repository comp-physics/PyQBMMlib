import sys
sys.path.append('../src/')
from qbmm_manager import *
sys.path.append('../utils/')
from stats_util import *

def test_project_1D(qbmm_mgr,mu,sig,tol):
    init_moments = raw_gaussian_moments_univar( qbmm_mgr.num_moments, mu, sig )

    abscissas, weights = qbmm_mgr.moment_invert( init_moments )
    projected_moments = qbmm_mgr.projection( weights, abscissas, qbmm_mgr.indices )

    err = np.linalg.norm(projected_moments - init_moments)
    success = ( err < tol )

    return success

def test_project_2D(qbmm_mgr,mu,sig,tol):
    init_moments = raw_gaussian_moments_bivar( 
            qbmm_mgr.indices,
            mu[0], mu[1], 
            sig[0], sig[1] 
        )

    abscissas, weights = qbmm_mgr.moment_invert( init_moments, qbmm_mgr.indices )
    projected_moments = qbmm_mgr.projection( weights, abscissas, qbmm_mgr.indices )

    err = np.linalg.norm(projected_moments - init_moments)
    success = ( err < tol )

    return success

if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )
    
    print('Unit test: test_projection')

    tol = 1.e-13  # Why?
    success = True
    
    # 1D
    print('Unit test: test 1d projection')
    config = {}
    config['qbmm'] = {}
    config['qbmm']['governing_dynamics']   = '4*x - 2*x**2'
    config['qbmm']['num_internal_coords']  = 1
    config['qbmm']['num_quadrature_nodes'] = 3
    config['qbmm']['method']       = 'qmom'
    qbmm_mgr = qbmm_manager( config )

    mu    = 0.0
    sigma = 1.1
    success *= test_project_1D(qbmm_mgr,mu,sigma,tol)

    # 2D
    print('Unit test: test 2d projection')
    config = {}
    config['qbmm'] = {}
    config['qbmm']['governing_dynamics']   = ' - xdot - x '
    config['qbmm']['num_internal_coords']  = 2
    config['qbmm']['num_quadrature_nodes'] = 4
    config['qbmm']['method']       = 'chyqmom'
    qbmm_mgr = qbmm_manager( config )
    
    mu    = [1.1,0.1]
    sigma = [1.0,1.0]
    success *= test_project_2D(qbmm_mgr,mu,sigma,tol)

    if success == True:
        print('test_projection: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_projection: ' + '\033[91m' + 'test failed ' + '\033[0m')
        
    exit()
