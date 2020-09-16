import numpy as np
from sympy import symbols
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
        print( '\t num_moments         = %i' % self.num_moments )
        
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
                self.permutation   = 12
                if 'permutation' in qbmm_config:
                    self.permutation = qbmm_config['permutation']
                self.inversion_option = self.permutation
                #
            elif self.method == 'chyqmom':
                #
                self.moment_invert = conditional_hyperbolic
                self.inversion_algorithm = conditional_hyperbolic
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
        ###
        ### [ecg] For 1D problems, this will return a 'flat' array, e.g.,
        ### [0,1,2,3,...]. However, for generality, this should always
        ### return an array of shape [num_coords, num_moments]. This is
        ### easy to implement, but Wheeler is not conditioned to take in
        ### such arrays as input, so it craps out. If we were to pursue
        ### this, we would need to modify Wheeler at some point.
        ###
        self.num_moments = 0
        #
        if self.num_internal_coords == 1:
            #
            if self.method == 'qmom':
                self.indices = np.arange( 2 * self.num_quadrature_nodes )
            elif self.method == 'hyqmom':
                # Spencer: is this general?
                self.indices = np.arange( 2 * ( self.num_quadrature_nodes - 1 ) + 1 )
            #
            self.num_moments = len( self.indices )
            #
        elif self.num_internal_coords > 1: 
            #
            if self.method == 'chyqmom' :
                self.indices = np.array( [ [0,0], [1,0], [0,1], [2,0], [1,1], [0,2] ] )
                message  = 'qbmm_mgr: moment_indices: Warning: Moment indices hardcoded for num_coords(2)'
                message += 'and num_nodes(2), requested num_coords(%i) and num_nodes(%i)'
                print( messgae % ( self.num_internal_coords, self.num_quadrature_nodes ) )
            #
            self.num_moments = self.indices.shape
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

    def quadrature(self, weights, abscissas, moment_index):
        """
        This function performs a general cubature for given weights, abscissas and indices
        """
        ###
        ### [ecg] For indices of shape [num_coords, num_moments],
        ### this should work (I think):
        ### (1) xi_to_idx = np.power( abscissas, self.indices[None,:] )
        ### However, indices in 1D problems are 'flat' arrays
        ### (see moment_indices, line 103, and the note therein).
        ### This means that moment_index passed from projection
        ### is a scalar, so expression (1) does not work. For now,
        ### this routine only works for 1D problems, and takes in
        ### a scalar for moment_index (weak).
        ###
        xi_to_idx = abscissas ** moment_index
        # \sum_j w_j xi_j^i_j
        q = np.dot( weights, xi_to_idx )
        return q

    def projection(self, weights, abscissas, indices):
        """
        This function reconstructs moments from quadrature weights and abscissas
        """
        num_indices = len( indices )
        moments = np.zeros( num_indices )
        for i_index in range( num_indices ):
            moments[i_index] = self.quadrature( weights, abscissas, indices[ i_index ] )                
        return moments

    def compute_rhs(self, sym_coefficients, sym_exponents, indices, weights, abscissas):
        """
        Compute moment-transport RHS
        """
        c0 = symbols('c0')
        num_indices   = len( self.indices )        
        num_exponents = len( sym_exponents )
        num_coefficients = len( sym_coefficients )
        rhs = np.zeros( num_indices )
        for i_index in range( num_indices ):
            exponents    = [sym_exponents[j].subs(c0, indices[i_index]) \
                    for j in range(num_exponents)]
            coefficients = [sym_coefficients[j].subs(c0, indices[i_index]) \
                    for j in range(num_coefficients)]
            np_exponents    = np.array( exponents )
            np_coefficients = np.array( coefficients )
            projected_moments = self.projection( weights, abscissas, np_exponents )
            rhs[i_index] = np.dot( np_coefficients, projected_moments )            
        return rhs
