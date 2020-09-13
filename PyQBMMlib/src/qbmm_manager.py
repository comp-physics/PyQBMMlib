import numpy as np
from inversion import *


class qbmm_manager:

    def __init__(self, config):

        self.governing_dynamics   = config['governing_dynamics']
        self.num_internal_coords  = config['num_internal_coords'] 
        self.num_quadrature_nodes = config['num_quadrature_nodes']
        self.method               = config['method']
        self.adaptive             = config['adaptive']

        iret = self.set_inversion( config )
        if iret == 1:
            print('qbmm_mgr: Configuration failed')
            return

        # Report config
        print('qbmm_mgr: Configuration options ready:')
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
                
        return

    def set_inversion(self, config):

        if self.num_internal_coords == 1:
            #
            self.moment_invert = self.moment_invert_1D
            #
            if self.method == 'qmom':
                #
                self.inversion_algorithm = wheeler
                self.adaptive      = False
                if 'adaptive' in config:
                    self.adaptive = config['adaptive']
                self.inversion_option = self.adaptive
                #
            elif self.method == 'hyqmom':
                #
                self.inversion_algorithm = hyperbolic
                self.max_skewness  = 30
                if 'max_skewness' in config:
                    self.max_skewness = config['max_skewness']
                self.inversion_option = self.max_skewness
                #
            else:
                message = 'qbmm_mgr: Error: No method %s for num_internal_coords = 1'
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
                self.permutation   = 12
                if 'permutation' in config:
                    self.permutation = config['permutation']
                self.inversion_option = self.permutation
                #
            elif self.method == 'chyqmom':
                #
                self.moment_invert = conditional_hyperbolic
                self.max_skewness  = 30
                if 'max_skewness' in config:
                    self.max_skewness = config['max_skewness']
                self.inversion_option = self.max_skewness
                #
        else:
            message = 'qbmm_mgr: Error: dimensionality %i unsupported'
            print( message % self.num_internal_coords )
            return(1)

        return(0)
    
    def moment_index(self):

        return

    def moment_invert_1D(self, moments):

        return self.inversion_algorithm( moments, self.inversion_option )

    def moment_invert_2PD(self, moments, indices):

        return self.inversion_agorithm( moments, indices, self.inversion_option )

    
