import sys
sys.path.append('../utils/')
from stats_util import *
from pretty_print_util import *
from qbmm_manager import *
import numpy as np
import csv

class time_advancer:

    def __init__(self, config):

        self.method          = config['advancer']['method']
        self.time_step       = config['advancer']['time_step']
        self.final_time      = config['advancer']['final_time']
        self.error_tol       = config['advancer']['error_tol']
        self.num_steps       = config['advancer']['num_steps']
        self.num_steps_print = config['advancer']['num_steps_print']
        self.num_steps_write = config['advancer']['num_steps_write']
        self.output_dir      = config['advancer']['output_dir']
        self.output_id       = config['advancer']['output_id']
        self.write_to        = config['advancer']['write_to']
        
        self.qbmm_mgr = qbmm_manager( config )

        self.num_dim  = self.qbmm_mgr.num_moments
        self.indices  = self.qbmm_mgr.indices

        self.state    = np.zeros( self.num_dim )
        self.rhs      = np.zeros( self.num_dim )

        self.stage_state = np.zeros( [3, self.num_dim] )
        self.stage_k     = np.zeros( [3, self.num_dim] )
        
        if self.method == 'Euler':
            self.advance = self.advance_euler
        elif self.method == 'RK23':
            self.advance = self.advance_RK23

        print('advancer: init: Configuration options ready')
        print('\t method          = %s'   % self.method)
        print('\t time_step       = %.4E' % self.time_step)
        print('\t final_time      = %.4E' % self.final_time)
        print('\t error_tol       = %.4E' % self.error_tol)
        print('\t num_steps_print = %i'   % self.num_steps_print)
        print('\t num_steps_write = %i'   % self.num_steps_write)
        print('\t output_dir      = %s'   % self.output_dir)
        print('\t output_id       = %s'   % self.output_id)
        print('\t write_to        = %s'   % self.write_to)

        self.max_time_step = 1.0

        self.file_name = self.output_dir + 'qbmm_state_' + self.output_id
        if self.write_to == 'txt':
            self.file_name += '.dat'
            self.write_to_file = self.write_to_txt
        elif self.write_to == 'h5':
            self.file_name += '.h5'
            self.write_to_file = self.write_to_h5
        
        return
    
    def initialize_state(self, init_state):

        self.state = init_state
        
        return

    def initialize_state_gaussian_bivar(self, mu1, mu2, sigma1, sigma2):

        self.state  = raw_gaussian_moments_bivar( self.indices, mu1, mu2, sigma1, sigma2 )
        message = 'advancer: initialize_bigaussian: '
        f_array_pretty_print( message, 'state', self.state )
        return

    def initialize_state_gaussian_univar(self, mu, sigma):

        self.state = raw_gaussian_moments_univar( self.num_dim, mu, sigma)
        message = 'advancer: initialize_gaussian: '
        f_array_pretty_print( message, 'state', self.state )
        return
    
    def advance_euler(self):

        # self.rhs += self.qbmm_mgr.evaluate_rhs( self.state )

        print('advancer: advance: Euler time advancer')
        
        self.rhs = self.qbmm_mgr.compute_rhs( self.state )

        ### state_{n+1} = proj_{n} + rhs_{n}
        self.state += self.time_step * self.rhs
        
        return

    def advance_RK23(self):

        #message = 'advancer: advace_RK3: '
        #f_array_pretty_print( message, 'state', self.state )
        
        # Stage 1: { y_1, k_1 } = f( t_n, y_0 )
        self.stage_state[0] = self.state.copy()
        time = self.time
        self.qbmm_mgr.compute_rhs( self.stage_state[0], self.stage_k[0] )

        # Stage 2: { y_2, k_2 } = f( t_n, y_1 + dt * k_1 )
        self.stage_state[1] = self.stage_state[0] + self.time_step * self.stage_k[0]
        time = self.time + self.time_step
        self.qbmm_mgr.compute_rhs( self.stage_state[1], self.stage_k[1] )

        # Stage 3: { y_3, k_3 } = f( t_n + 0.5 * dt, ... )
        self.stage_state[2] = 0.75 * self.stage_state[0] + 0.25 * ( self.stage_state[1] + self.time_step * self.stage_k[1] )
        self.qbmm_mgr.compute_rhs( self.stage_state[2], self.stage_k[2] )

        # Updates
        test_state = 0.5 * ( self.stage_state[0] + ( self.stage_state[1] + self.time_step * self.stage_k[1] ) )
        self.state = ( self.stage_state[0] + 2.0 * ( self.stage_state[2] + self.time_step * self.stage_k[2] ) ) / 3.0 

        # f_array_pretty_print( message, 'stage_state_0', self.stage_state[0] )
        # f_array_pretty_print( message, 'stage_state_1', self.stage_state[1] )
        # f_array_pretty_print( message, 'stage_state_2', self.stage_state[2] )
        # f_array_pretty_print( message, 'stage_k_0', self.stage_k[0] )
        # f_array_pretty_print( message, 'stage_k_1', self.stage_k[1] )
        # f_array_pretty_print( message, 'stage_k_2', self.stage_k[2] )
        # f_array_pretty_print( message, 'state', self.state )
        
        self.rk_error = np.linalg.norm( self.state - test_state ) / np.linalg.norm( self.state )

    def adapt_time_step(self):

        error_fraction   = np.sqrt( 0.5 * self.error_tol / self.rk_error )
        time_step_factor = min( max( error_fraction, 0.3 ), 2.0 )
        new_time_step    = time_step_factor * self.time_step
        new_time_step    = min( max( 0.9 * new_time_step, self.time_step ), self.max_time_step )
        self.time_step   = new_time_step        
        return
        
    def run(self):

        print('advancer: run: Preparing to step')
        
        self.time = 0.0

        self.report_step(0)
        self.write_step(0)

        i_step = 0
        step   = True
        while step == True:

            self.advance()

            i_step    += 1
            self.time += self.time_step

            if i_step % self.num_steps_print == 0:
                self.report_step(i_step)

            if i_step % self.num_steps_write == 0:
                self.write_step(i_step)
                
            self.adapt_time_step()

            if self.time > self.final_time or i_step >= self.num_steps:
                step = False
                
        print('advancer: run: stepping ends')
            
        return

    def report_step(self, i_step):

        message = 'advancer: step = ' + str(i_step) + ' ... time = ' + '{:.16E}'.format(self.time) + ' ... time_step = ' + '{:.16E}'.format(self.time_step) + ' ... '
        f_array_pretty_print( message, 'state', self.state )

    def write_step(self, i_step):

        message = 'advancer: step = ' + str(i_step) + ' ... Writing to file'
        print(message)
        self.write_to_file( i_step )
        
    def write_to_txt(self, i_step):

        write_flag = 'a'
        if i_step == 0:
            write_flag = 'w'
            
        with open( self.file_name, write_flag ) as file_id:
           csv.writer( file_id, delimiter=' ' ).writerow( self.state )
           
        return
        
    def write_to_h5(self):
        
        print('advancer: write_to_h5: not implemented yet')
        return
