"""

.. module:: qbmm_manager
   :platform: MacOS, Unix
   :synopsis: A useful module indeed.

.. moduleauthor:: SHB <spencer@caltech.edu> and Esteban Cisneros <csnrsgr2@illinois.edu>

"""

import sys
sys.path.append('../utils/')

from inversion import *
from pretty_print_util import *
import numpy as np
import sympy as smp

try:
    import numba
    from nquad import *
except:
    print('Did not find numba! Install it for significant speedups.')
    from quad import *

class qbmm_manager:
    """This class manages the computation of moment-transport RHS. It is meant to be called from within :class:`time_advancer`, with which it interfaces through :func:`compute_rhs`.     
    """

    def __init__(self, config):
        """Constructor

        :param config: Configuration
        :type config: dict
        """
        self.flow                 = config['qbmm']['flow']
        self.governing_dynamics   = config['qbmm']['governing_dynamics']
        self.num_internal_coords  = config['qbmm']['num_internal_coords'] 
        self.num_quadrature_nodes = config['qbmm']['num_quadrature_nodes']
        self.method               = config['qbmm']['method']
        self.adaptive             = config['qbmm']['adaptive']
        # self.poly                 = config['qbmm']['polydisperse']

        # if self.poly:
        #     self.num_poly_nodes = config['qbmm']['num_poly_nodes']
        #     self.poly_symbol    = config['qbmm']['poly_symbol']

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
        if self.method == 'hyqmom' or self.method == 'chyqmom':
            print( '\t max_skewness        = %i' % self.max_skewness )

        # Determine moment indices
        self.moment_indices()
        print( '\t num_moments         = %i' % self.num_moments )

        # Determine coefficients & exponents from governing dynamics
        if self.num_internal_coords < 3:
            self.transport_terms() 

        # RHS buffer
        self.rhs = np.zeros( self.num_moments )

        return

    def set_inversion(self, config):
        """
        This function sets the inversion procedure based on config options

        :param config: Configuration
        :type config: dict
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
            if self.method == 'chyqmom':
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
        This function sets moment indices according to dimensionality (num_coords and num_nodes) and method.
        """
        
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
        #
        # # Todo: append indices for polydisperse direction r0
        # if self.num_internal_coords == 2 and self.poly:
        #     orig_idx = self.indices
        #     self.indices = np.zeros( num_poly_nodes * len(orig_idx) )
        #     for j in range( num_poly_nodes ):
        #         for i in range( len(orig_idx) ):
        #             self.indices[i] = np.append( orig_idx[i], j )
        return 

    def transport_terms(self):
        """
        This function determines the RHS in the moments equation for a given governing dynamics
        """

        if self.num_internal_coords == 1:
            x = smp.symbols( 'x' )
            l = smp.symbols( 'l', real = True )
            xdot = smp.parse_expr( self.governing_dynamics )
            integrand = xdot * ( x ** ( l - 1 ) )
            self.symbolic_indices = l
        elif self.num_internal_coords == 2:
            # if self.poly:
            #     r0 = smp.symbols( self.poly_symbol )
            x,xdot = smp.symbols( 'x xdot' )
            l,m    = smp.symbols( 'l m', real = True )
            xddot  = smp.parse_expr( self.governing_dynamics )
            integrand = xddot * ( x ** l ) * ( xdot ** ( m - 1 ) )
            self.symbolic_indices = [l,m]

        terms     = smp.powsimp( smp.expand( integrand ) ).args
        num_terms = len( terms )

        # Add constant term for 2+D problems
        total_num_terms = num_terms
        if self.num_internal_coords == 2:
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

        # Add extra constant term if in 2D
        if self.num_internal_coords == 2:
            self.exponents[ num_terms, 0 ] = l - 1
            self.exponents[ num_terms, 1 ] = m + 1
            self.coefficients[ num_terms ] = l

        self.num_coefficients = len( self.coefficients )
        self.num_exponents    = len( self.exponents    )

        for i in range( self.num_coefficients ):
            if self.num_internal_coords == 1:
                self.coefficients[i] = smp.lambdify([l],self.coefficients[i])
                for j in range(self.num_internal_coords):
                    self.exponents[i,j] = smp.lambdify([l],self.exponents[i,j])
            elif self.num_internal_coords == 2:
                self.coefficients[i] = smp.lambdify([l,m],self.coefficients[i])
                for j in range(self.num_internal_coords):
                    self.exponents[i,j] = smp.lambdify([l,m],self.exponents[i,j])

        message = 'qbmm_mgr: transport_terms: '
        for i in range( total_num_terms ):
            sym_array_pretty_print( message, 'exponents', self.exponents[i,:] )

        message = 'qbmm_mgr: transport_terms: '
        sym_array_pretty_print( message, 'coefficients', self.coefficients )

        return
    
    def moment_invert_1D(self, moments):
        """
        This function inverts tracked moments into a quadrature rule in 1D

        :param moments: Tracked moments
        :type moments: array like
        :return: quadrature abscissas, weights
        :rtype: array like

        This is never directly invoked. Instead, the user calls

        >>> xi, wts = qbmm_mgr.moment_inver(moments)

        and qbmm_manager automatically selects moment_invert_1D based if :any:`num_internal_coords = 1`
        """
        return self.inversion_algorithm( moments, self.inversion_option )

    def moment_invert_2PD(self, moments):
        """
        This function inverts moments into a quadrature rule in ND > 1

        :param moments: Tracked moments
        :type moments: array like
        :return: quadrature abscissas, weights
        :rtype: array like

        This is never directly invoked. Instead, the user calls

        >>> xi, wts = qbmm_mgr.moment_inver(moments)

        and qbmm_manager automatically selects moment_invert_2PD based if :any:`num_internal_coords > 1`
        """
        return self.inversion_algorithm( moments, self.indices, self.inversion_option )

    def projection(self, weights, abscissas, indices):
        """
        This function reconstructs moments (indices) from quadrature weights and abscissas

        :param weights: Quadrature weights
        :param abscissas: Quadrature abscissas
        :param indices: Full moment set indices
        :type weights: array like
        :type abscissas: array like
        :type weights: array like
        :return: projected moments
        :rtype: array like
        """
        abscissas = np.array(abscissas)
        moments = np.zeros( len(indices) )
        for i_index in range( len(indices) ):
            if self.num_internal_coords == 3:
                moments[i_index] = quadrature_3d(weights, abscissas, indices[i_index], self.num_quadrature_nodes)
            if self.num_internal_coords == 2:
                moments[i_index] = quadrature_2d(weights, abscissas, indices[i_index], self.num_quadrature_nodes)
            elif self.num_internal_coords == 1:
                moments[i_index] = quadrature_1d(weights, abscissas, indices[i_index])
        return moments

    def compute_rhs(self, moments, rhs):
        """
        This function computes moment-transport RHS

        :param moments: Transported moments
        :param rhs: Moments rate-of-change

        """
        # Compute abscissas and weights from moments
        if self.num_internal_coords == 1:
            abscissas, weights = self.moment_invert( moments )      
        else:
            abscissas, weights = self.moment_invert( moments,self.indices )      

        # Loop over moments
        for i_moment in range( self.num_moments ):
            # Evalue RHS terms
            if self.num_internal_coords == 1:
                exponents    = [np.double(self.exponents[j,0](self.indices[i_moment])) 
                                for j in range(self.num_exponents)]
                coefficients = [np.double(self.coefficients[j](self.indices[i_moment]) )
                                for j in range(self.num_coefficients)]
            elif self.num_internal_coords == 2:
                exponents    = [ [ \
                        np.double(self.exponents[j,0](self.indices[i_moment][0],self.indices[i_moment][1])), \
                        np.double(self.exponents[j,1](self.indices[i_moment][0],self.indices[i_moment][1]))  \
                        ] for j in range(self.num_exponents)]
                coefficients = [ \
                        np.double(self.coefficients[j](self.indices[i_moment][0],self.indices[i_moment][1])) 
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
