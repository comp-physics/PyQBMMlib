import sys
sys.path.append('../src/')
from qbmm_manager import *

def test_project_1D(mu,sig,tol):
    advancer.initialize_state_gaussian_univar( mu, sig )
    x,w = invert(moments)
    new_moments = project(x,w)
    err = np.norm(new_moments - moments)
    success = ( err < tol )
    return success

def test_project_2D(mu,sig,tol):
    advancer.initialize_state_gaussian_bivar( 
            mu[0], mu[1], 
            sig[0], sig[1] 
        )
    x,w = invert(moments)
    new_moments = project(x,w)
    err = np.norm(new_moments - moments)
    success = ( err < tol )
    return success

if __name__ == '__main__':

    np.set_printoptions( formatter = { 'float': '{: 0.4E}'.format } )
    
    print('Unit test: test_projection')

    tol = 1.e-13  # Why?
    success = True
    
    # 1D
    config = {}
    config['governing_dynamics']   = ' xdot + x = 1'
    config['num_internal_coords']  = 1
    config['num_quadrature_nodes'] = 4
    config['method']       = 'qmom'
    config['adaptive']     = False
    qbmm_mgr = qbmm_manager( config )

    mu    = 0.0
    sigma = 1.1
    success *= test_project_1D(mu,sigma,tol)

    # 2D
    config = {}
    config['governing_dynamics']   = ' xddot + xdot + x = 1'
    config['num_internal_coords']  = 2
    config['num_quadrature_nodes'] = 4
    config['method']       = 'chqmom'
    qbmm_mgr = qbmm_manager( config )
    
    mu    = [1.1,0.1]
    sigma = [1.0,1.0]
    success *= test_project_2D(mu,sigma,tol)

    if success == True:
        color = '\033[92m'
        print('test_projection: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_projection: ' + '\033[91m' + 'test failed ' + '\033[0m' + '... check Wheeler!')
        
    exit
