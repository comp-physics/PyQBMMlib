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

    def initialize_state(self, init_state):

        self.state = init_state
        
        return
    
    def advance_euler(self):

        # self.rhs += self.qbmm_mgr.evaluate_rhs( self.state )

        print('advancer: advance: Euler time advancer')
        
        self.rhs = qbmm_mgr.compute_rhs( self.state )

        ### state_{n+1} = proj_{n} + rhs_{n}
        self.state += self.time_step * self.rhs
        
        return

    def advance_RK23(self):

        print('advancer: advance: RK23 time advancer')

        # Stage 1: {y_1, k_1 } = f( t_n, y_0 )
        self.stage_state[0] = self.state.copy()
        time = self.t
        self.qbmm_mgr.compute_rhs( self.stage_state[0], self.stage_k[0] )

        # Stage 2: { y_2, k_2 } = f( t_n, y_1 + dt * k_1 )
        self.stage_state[1] = self.stage_state[0] + self.dt * self.stage_k[0]
        time = self.t + self.dt
        self.qbmm_mgr.compute_rhs( self.stage_state, self.stage_k[1] )

        # Stage 3: { y_3, k_3 } = f( t_n + 0.5 * dt, y_ )
        self.stage_state[2] = 0.75 * self.stage_state[0] + 0.25 * ( self.stage_state[1] + self.dt * self.stage_k[1] )

        # Updates
        test_state = 0.5 * ( self.stage_state[0] + ( self.stage_state[1] + self.dt * self.stage_k[1] ) )
        self.state = ( self.stage_state[0] + 2.0 * ( self.stage_state[2] + self.dt * self.stage_k[2] ) ) / 3.0 
        
        self.rk_error = np.linalg.norm( self.state - test_state ) / np.linalg.norm( self.state )
        
        
    def run(self):

        print('advancer: run: Preparing to step')
        
        self.time = 0.0
        
        for i_step in range( self.num_steps ):

            self.advance()

            if i_step % self.num_steps_print:
                self.report_step()

            self.time += self.time_step

        print('advancer: run: stepping ends')
            
        return

    def report_step():

        message = 'advancer: step = %i ... time = %.4E ... state'
        f_array_pretty_print( message, self.state )
        print('advancer: step')
