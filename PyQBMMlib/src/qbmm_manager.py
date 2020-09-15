import numpy as np
from inversion import *


class qbmm_manager:

    def __init__(self, config):
        """
        Constructor. Takes in a config dictionary.
        """        
        self.governing_dynamics   = config['qbmm']['governing_dynamics']
        self.num_internal_coords  = config['qbmm']['num_internal_coords'] 
        self.num_quadrature_nodes = config['qbmm']['num_quadrature_nodes']
        self.method               = config['qbmm']['method']
        self.adaptive             = config['qbmm']['adaptive']
        self.indices              = config['qbmm']['indices']

        iret = self.set_inversion( config )
        if iret == 1:
            print('qbmm_mgr: init: Configuration failed')
            return

        # Report config
        print('qbmm_mgr: init: Configuration options ready:')
        print('\t governing_dynamics  = %s' % self.governing_dynamics)
        print('\t num_internal_coords = %i' % self.num_internal_coords)
        print('\t method              = %s' % self.method)
        # Report method-specific config
        if self.method == 'qmom':
            print( '\t adaptive            = %s' % str( self.adaptive ) ) 
        if self.method == 'cqmom':
            print( '\t permutation         = %i' % self.permutation )
        if self.method == 'hyqmom' or self.method == 'chyqmom':
            print( '\t max_skewness        = %i' % self.max_skewness )

        self.moment_indices()
            
        return

    def set_inversion(self, config):
        """
        This function sets the inversion procedure based on config options.
        """        
        qbmm_config = config['qbmm']
        
        if self.num_internal_coords == 1:
            #
            self.moment_invert = self.moment_invert_1D
            #
            if self.method == 'qmom':
                #
                self.inversion_algorithm = wheeler
                self.adaptive      = False
                if 'adaptive' in qbmm_config:
                    self.adaptive = qbmm_config['adaptive']
                self.inversion_option = self.adaptive
                #
            elif self.method == 'hyqmom':
                #
                self.inversion_algorithm = hyperbolic
                self.max_skewness  = 30
                if 'max_skewness' in qbmm_config:
                    self.max_skewness = qbmm_config['max_skewness']
                self.inversion_option = self.max_skewness
                #
            else:
                message = 'qbmm_mgr: set_inversion: Error: No method %s for num_internal_coords = 1'
                print( message % self.method )
                return(1)
            #
        elif self.num_internal_coords == 2 or self.num_internal_coords == 3:
            #
            self.moment_invert = self.moment_invert_2PD
            #
            if self.method == 'cqmom':
                #
                self.moment_invert = conditional
                self.inversion_algorithm = conditional
                self.indices = qbmm_config['indices']
                self.permutation   = 12
                if 'permutation' in qbmm_config:
                    self.permutation = qbmm_config['permutation']
                self.inversion_option = self.permutation
                #
            elif self.method == 'chyqmom':
                #
                self.moment_invert = conditional_hyperbolic
                self.inversion_algorithm = conditional_hyperbolic
                self.indices = qbmm_config['indices']
                self.max_skewness  = 30
                if 'max_skewness' in qbmm_config:
                    self.max_skewness = qbmm_config['max_skewness']
                if 'permutation' in qbmm_config:
                    self.permutation = qbmm_config['permutation']
                self.inversion_option = self.max_skewness
                self.inversion_option = self.permutation
                #
        else:
            message = 'qbmm_mgr: set_inversion: Error: dimensionality %i unsupported'
            print( message % self.num_internal_coords )
            return(1)

        return(0)
    
    def moment_indices(self):
        """
        This function sets moment indices according to 
        dimensionality (num_coords and num_nodes) and method.
        """
        if self.num_internal_coords == 1:
            #
            if self.method == 'qmom':
                self.indices = np.arange( 2 * self.num_quadrature_nodes )
            elif self.method == 'hyqmom':
                self.indices = np.arange( 2 * ( self.num_quadrature_nodes - 1 ) + 1 ) # Spencer: is this general?
            #
        elif self.num_internal_coords > 1: 
            #
            if self.method == 'chyqmom' :
                self.indices = np.array( [ [0,0], [1,0], [0,1], [2,0], [1,1], [0,2] ] )
                message  = 'qbmm_mgr: moment_indices: Warning: Moment indices hardcoded for num_coords(2)'
                message += 'and num_nodes(2), requested num_coords(%i) and num_nodes(%i)'
                print( messgae % ( self.num_internal_coords, self.num_quadrature_nodes ) )
            #
        else:
            #
            print('qbmm_mgr: moment_indices: Error: dimensionality %i unsupported' % self.num_internal_coords )

        return 

    def moment_invert_1D(self, moments):
        """
        This function inverts moments in 1D
        """
        return self.inversion_algorithm( moments, self.inversion_option )

    def moment_invert_2PD(self, moments, indices):
        """
        This function inverts moments in ND, with N > 1
        """
        return self.inversion_algorithm( moments, indices, self.inversion_option )

    def quadrature(weights, abscissa, indices):
        """
        This function performs a general cubature for given weights, abscissas and indices
        """
        # [ecg] I think this line is general enough, but must test
        xi_to_idx = np.power( abscissas, indices[None,:] )
        # \sum_j w_j xi_j^i_j
        q = np.dot( weights, xi_to_i )
        return q

    def projection(weights, abscissas, indices):
        """
        This function reconstructs moments from quadrature weights and abscissas
        """
        num_indices = len( self.indices )
        moments = np.zeros( num_indices )
        for i_index in range( num_indices ):
            moments[i_index] = self.quadrature( weights, absicssas, indices[i_index] )                
        return moments

    def compute_rhs(coefficients, exponents, indices, weights, abscissas):
        """
        Compute moment-transport RHS
        """
        print('qbmm: compute_rhs: Warning: Hardcoded')
