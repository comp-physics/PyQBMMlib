import sys
sys.path.append('../utils/')

import numpy as np
import sympy as smp
from inversion import *
from pretty_print_util import *

class qbmm_manager:

    def __init__(self, config):
        """
        Constructor. Takes in a config dictionary.
        """
        self.flow                 = config['qbmm']['flow']
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
        print('\t flow                = %s' % self.flow )
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

        # Determine moment indices
        self.moment_indices()
        # if self.num_internal_coords == 1:
        print( '\t num_moments         = %i' % self.num_moments )
        # else:
        #     i_array_pretty_print( '\t num_moments        ', '', self.num_moments )

        # Determine coefficients & exponents from governing dynamics
        if self.num_internal_coords < 3:
            self.transport_terms() 

        # RHS buffer
        self.rhs = np.zeros( self.num_moments )
        
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
        elif self.num_internal_coords == 2:
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
                self.max_skewness = 30
                self.permutation  = 12
                if 'max_skewness' in qbmm_config:
                    self.max_skewness = qbmm_config['max_skewness']
                if 'permutation' in qbmm_config:
                    self.permutation = qbmm_config['permutation']
                self.inversion_option = self.max_skewness
                self.inversion_option = self.permutation
                #
        elif self.num_internal_coords == 3:
            #
            self.moment_invert = self.moment_invert_2PD
            #
            if self.method == 'chyqmom':
                #
                self.moment_invert = conditional_hyperbolic
                self.inversion_algorithm = conditional_hyperbolic
                self.max_skewness = 30
                if 'max_skewness' in qbmm_config:
                    self.max_skewness = qbmm_config['max_skewness']
                self.inversion_option = self.max_skewness
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
                self.indices = np.arange( 2 * ( self.num_quadrature_nodes - 1 ) + 1 )
            #
            self.num_moments = len( self.indices )
            #
            message = 'qbmm_mgr: moment_indices: '
            f_array_pretty_print( message, 'indices', self.indices )
        elif self.num_internal_coords == 2: 
            #
            if self.method == 'chyqmom':
                if self.num_quadrature_nodes == 4:
                    self.indices = np.array( [ [0,0], [1,0], [0,1], [2,0], [1,1], [0,2] ] )
                elif self.num_quadrature_nodes == 9:
                    self.indices = np.array( [ [0,0], [1,0], [0,1], [2,0], [1,1], [0,2], [3,0], [0,3], [4,0], [0,4] ] )
                else :
                    print( 'qbmm_mgr: moment_indices: Error: incorrect number of quadrature nodes (not 4 or 9), aborting... %i' % self.num_quadrature_nodes )
                    quit()

            #
            self.num_moments = self.indices.shape[0]
            #
        elif self.num_internal_coords == 3: 
            #
            if self.method == 'chyqmom':
                if self.num_quadrature_nodes == 27:
                    self.indices = np.array( [ [0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2], [3,0,0], [0,3,0], [0,0,3], [4,0,0], [0,4,0], [0,0,4] ] )

                else :
                    print( 'qbmm_mgr: moment_indices: Error: incorrect number of quadrature nodes (not 27), aborting... %i' % self.num_quadrature_nodes )
                    quit()
            else :
                print( 'qbmm_mgr: moment_indices: Error: Unsupported method, aborting...' )
                quit()

            self.num_moments = self.indices.shape[0]
        else:
            #
            print('qbmm_mgr: moment_indices: Error: dimensionality %i unsupported' % self.num_internal_coords )
            quit()
        return 

    def transport_terms(self):
        """
        This function determines the RHS in the moments equation
        for a given governing dynamics
        """
        print('qbmm_mgr: transport_terms: Warning: works for up to two internal coordinates only')

        if self.num_internal_coords == 1:
            x = smp.symbols( 'x' )
            l = smp.symbols( 'l', real = True )
            xdot = smp.parse_expr( self.governing_dynamics )
            integrand = xdot * ( x ** ( l - 1 ) )
            self.symbolic_indices = l
        elif self.num_internal_coords == 2:
            x,xdot = smp.symbols( 'x xdot' )
            l,m    = smp.symbols( 'l m', real = True )
            xddot  = smp.parse_expr( self.governing_dynamics )
            integrand = xddot * ( x ** l ) * ( xdot ** ( m - 1 ) )
            self.symbolic_indices = [l,m]

        terms     = smp.powsimp( smp.expand( integrand ) ).args
        num_terms = len( terms )

        # Add constant term for 2+D problems
        total_num_terms = num_terms
        if self.num_internal_coords > 1:
            total_num_terms += 1

        # Initialize exponents and coefficients (weird, but works)
        self.exponents    = [[smp.symbols('a') for i in range(total_num_terms)]
                             for j in range(self.num_internal_coords)]
        self.coefficients = [smp.symbols('a') for i in range(total_num_terms)]

        # Everything is simpler if now transferred into numpy arrays
        self.exponents    = np.array(self.exponents).T
        self.coefficients = np.array(self.coefficients).T

        # Loop over terms
        for i in range( num_terms ):

            self.exponents[i,0] = terms[i].as_coeff_exponent(x)[1]
            if self.num_internal_coords == 1:
                self.coefficients[i] = l * smp.poly( terms[i]).coeffs()[0]
            else:
                self.exponents[i,1] = terms[i].as_coeff_exponent(xdot)[1]
                self.coefficients[i] = m * smp.poly( terms[i] ).coeffs()[0]

        # Add extra constant term if in 2+D
        if self.num_internal_coords > 1:
            self.exponents[ num_terms, 0 ] = l - 1
            self.exponents[ num_terms, 1 ] = m + 1
            self.coefficients[ num_terms ] = l

        message = 'qbmm_mgr: transport_terms: '
        for i in range( total_num_terms ):
            sym_array_pretty_print( message, 'exponents', self.exponents[i,:] )

        message = 'qbmm_mgr: transport_terms: '
        sym_array_pretty_print( message, 'coefficients', self.coefficients )
        
        self.num_coefficients = len( self.coefficients )
        self.num_exponents    = len( self.exponents    )
        
        return
    
    def moment_invert_1D(self, moments):
        """
        This function inverts moments in 1D
        """
        return self.inversion_algorithm( moments, self.inversion_option )

    def moment_invert_2PD(self, moments):
        """
        This function inverts moments in ND, with N > 1
        """
        return self.inversion_algorithm( moments, self.indices, self.inversion_option )

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
        if self.num_internal_coords == 1:
            xi_to_idx = abscissas ** moment_index
            # \sum_j w_j xi_j^i_j
            q = np.dot( weights, xi_to_idx )
        elif self.num_internal_coords == 2:
            q = 0.
            for i in range(len(abscissas[0])):
                q = q + weights[i] * \
                    abscissas[0][i]**moment_index[0] * \
                    abscissas[1][i]**moment_index[1]
        elif self.num_internal_coords == 3:
            q = 0.
            for i in range(len(abscissas[0])):
                q = q + weights[i] * \
                    abscissas[0][i]**moment_index[0] * \
                    abscissas[1][i]**moment_index[1] * \
                    abscissas[2][i]**moment_index[2]
        else:
            print('Quadrature not implemented for ', self.num_internal_coords)
            quit()
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

    def compute_rhs(self, moments, rhs):
        """
        Compute moment-transport RHS
        """
        # Compute abscissas and weights from moments
        if self.num_internal_coords == 1:
            abscissas, weights = self.moment_invert( moments )      
        else:
            abscissas, weights = self.moment_invert( moments,self.indices )      

        # Symbols
        c0 = self.symbolic_indices

        # Loop over moments
        for i_moment in range( self.num_moments ):
            # Evalue RHS terms
            # [ecg] HACK HACK HACK
            if self.num_internal_coords == 1:
                exponents    = [self.exponents[j,0].subs( c0, self.indices[i_moment] ) 
                                for j in range(self.num_exponents)]
                coefficients = [self.coefficients[j].subs( c0, self.indices[i_moment] ) 
                                for j in range(self.num_coefficients)]
            # [SHB] HACK HACK HACK
            elif self.num_internal_coords == 2:
                exponents    = [ [ \
                        self.exponents[j,0].subs( c0[0], self.indices[i_moment][0]).subs( c0[1], self.indices[i_moment][1]), \
                        self.exponents[j,1].subs( c0[0], self.indices[i_moment][0]).subs( c0[1], self.indices[i_moment][1])  \
                        ]
                        for j in range(self.num_exponents)]
                coefficients = [ \
                        self.coefficients[j].subs( c0[0], self.indices[i_moment][0]).subs( c0[1], self.indices[i_moment][1] ) 
                        for j in range(self.num_coefficients)]
            else :
                print('num_internal_coords',self.num_internal_coords,'not supported yet')
                quit()

            # Put them in numpy arrays
            np_exponents    = np.array( exponents )
            np_coefficients = np.array( coefficients )
            # Project back to moments
            rhs_moments = self.projection( weights, abscissas, np_exponents )
            # Compute RHS
            rhs[i_moment] = np.dot( np_coefficients, rhs_moments )            
        #
        projected_moments = self.projection( weights, abscissas, self.indices )
        for i_moment in range( self.num_moments ):
            moments[i_moment] = projected_moments[i_moment]               
        #
        return
