import numpy as np
from inversion import *


class qbmm_manager:

    def __init__(self, config):

        self.governing_dynamics   = config['governing_dynamics']
        self.num_internal_coords  = config['num_internal_coords'] 
        self.num_quadrature_nodes = config['num_quadrature_nodes']
        self.method               = config['method']

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
        if self.method == 'cqmom':
            print( '\t permutation        = %i' % self.permutation )
        if self.method == 'hyqmom' or self.method == 'chyqmom':
            print( '\t max_skewness       = %i' % self.max_skewness )
                
        return

    def set_inversion(self, config):

        if self.num_internal_coords == 1:
            #
            if self.method == 'qmom':
                #
                self.moment_invert = classic_wheeler
                #
            elif self.method == 'aqmom':
                #
                self.moment_invert = adaptive_wheeler
                #
            elif self.method == 'hyqmom':
                #
                self.moment_invert = hyperbolic
                self.max_skewness  = 30
                if config.has_key( 'max_skewness' ):
                    self.max_skewness = config['max_skewness']
                #
            else:
                message = 'qbmm_mgr: Error: No method %s for num_internal_coords = 1'
                print( message % self.method )
                return(1)
            #
        elif self.num_internal_coords == 2 or self.num_internal_coords == 3:
            #
            if self.method == 'cqmom':
                #
                self.moment_invert = conditional
                self.permutation   = 12
                if config.has_key( 'permutation' ):
                    self.permutation = config['permutation']
                #
            elif self.method == 'chyqmom':
                #
                self.moment_invert = conditional_hyperbolic
                self.max_skewness  = 30
                if config.has_key( 'max_skewness' ):
                    self.max_skewness = config['max_skewness']
                #
        else:
            message = 'qbmm_mgr: Error: dimensionality %i unsupported'
            print( message % self.num_internal_coords )
            return(1)

        return(0)
    
    def moment_index(self):

        return
