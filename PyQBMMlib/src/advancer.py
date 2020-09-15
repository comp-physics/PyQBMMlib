import numpy as np

class advancer:

    def __init__(self, config):

        self.method    = config['advancer']['method']
        self.time_step = config['advancer']['time_step']
        self.num_steps = config['advancer']['num_steps']

        self.qbmm_mgr = qbmm_manager( config )

        self.num_dim  = qbmm_mgr.num_moments
        self.state    = np.zeros( self.num_dim )
        self.rhs      = np.zeros( self.num_dim )
        
        return

    def initialize_moments(self):

        return
    
    def advance(self):

        # self.rhs += self.qbmm_mgr.evaluate_rhs( self.state )
        
        return

    def run(self):

        return
