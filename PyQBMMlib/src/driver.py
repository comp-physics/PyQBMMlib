from qbmm_manager import *

if __name__ == '__main__':
    
    config = {}
    config['governing_dynamics']   = ' dx + x = 1'
    config['num_internal_coords']  = 1
    config['num_quadrature_nodes'] = 3
    config['method']       = 'qmom'
    config['adaptive']     = False
    config['max_skewness'] = 30

    qbmm_mgr = qbmm_manager( config )

    # Initialize moment set
    moments = np.zeros( config['num_quadrature_nodes'] )

    # n = {2, 2};
    # \[Mu]1 = 1.; \[Sigma]1 = 0.1;
    # P1 = LogNormalDistribution[Log[\[Mu]1], \[Sigma]1];
    # \[Mu]2 = 0; \[Sigma]2 = 0.2;
    # P2 = NormalDistribution[\[Mu]2, \[Sigma]2];
    # NDF = ProductDistribution[P1, P2];

    # test from p55 Marchisio + Fox 2013 exercise 3.2
    # moments = [ 1., 5., 26., 140., 778, 4450, 26140, 157400 ]
    # weights, abscissas = qbmm_mgr.inversion_algorithm( moments, config )








    print(weights,abscissas)
    print("expected result (order does not matter): w1 = 0.0459, w2 = 0.4541, w3 = 0.4541, and w4 = 0.0459.  x1 = -2.3344, x2 = -0.7420, x3 = 0.7420, and x4 = 2.3344.")

    exit
