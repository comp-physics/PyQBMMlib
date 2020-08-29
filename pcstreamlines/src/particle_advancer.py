import numpy as np
import scipy as sp
from simulation_domain import *
from initial_conditions_manager import *
from vtk_tools import *

class particle_advancer():

    def __init__(self, config, sim_domain):
        
        # Some flags
        self.domain_set = False
        self.state_set  = False
        self.init_set   = False
        self.config_set = False
        
        # Set domain
        self.set_domain( sim_domain )       
        
        # Initialize numbers
        self.num_particles = config['advancer_options']['num_particles']
        self.num_dim       = self.domain.num_dim
        self.state_dim     = self.num_particles * self.num_dim
        
        # Initialize auxiliary 1-particle buffer
        self.position  = np.zeros( self.num_dim )
        
        # Initialize state buffers
        self.state       = np.zeros( self.state_dim )
        self.state_rhs   = np.zeros( self.state_dim )

        # Initialize interpolation buffers
        self.interp_indices = np.zeros( [ self.num_particles, self.num_dim ] )
        self.interp_weights = np.zeros( [ self.num_particles, self.num_dim ] )
        
        # Initialize RK4 buffers        
        self.stage_state = np.zeros( self.state_dim )
        self.stage_k     = np.zeros( [ 4, self.state_dim ] )
        self.state_set   = True

        # Initialize residence time buffers
        self.time_elapsed    = {}
        self.residence_times = np.zeros( self.num_particles )
        
        # Initialize particles to be in the domain
        self.particle_in = np.ones( self.num_particles, dtype=bool ) 
        
        # Initialize save buffers
        self.particle_trajectories = {}
        self.particle_velocities   = {}
        self.particle_markers = {}
        self.secondary_fields = {}

        secondary_field_names     = self.domain.get_secondary_fields_names()
        self.num_secondary_fields = len( secondary_field_names )
        if self.num_secondary_fields > 0:
            for field_name in secondary_field_names:
                self.secondary_fields[ field_name ] = {}
        
        # Verbosity
        self.ver = config['advancer_options']['verbosity']
        
        # Initialize advancer options
        self.dt              = config['advancer_options']['time_step']
        self.num_steps       = config['advancer_options']['num_steps']
        self.num_steps_save  = config['advancer_options']['num_steps_save']
        self.num_steps_write = 1 #config['advancer_options']['num_steps_write']
        self.num_steps_print = config['advancer_options']['num_steps_print']
        self.residence_time  = config['advancer_options']['residence_time']
        self.break_steps     = config['advancer_options']['break_steps']
        self.write_to        = config['advancer_options']['write_to']
        self.file_id         = config['advancer_options']['file_id']
        self.out_dir         = config['advancer_options']['out_dir']        
        self.config_set      = True

        self.test = False
        if config['advancer_options'].has_key('test'):
            self.test = config['advancer_options']['test']
        
        if self.ver == 1:
            print('advancer: Configuration options ready:')
            print( '\t num_particles   = %i' % self.num_particles )
            print( '\t time_step       = %.4E' % self.dt )            
            print( '\t num_steps       = %i' % self.num_steps )
            print( '\t num_steps_save  = %i' % self.num_steps_save )
            print( '\t num_steps_write = %i' % self.num_steps_write )
            print( '\t num_steps_print = %i' % self.num_steps_print )
            print( '\t residence_time  = %i' % int( self.residence_time ) )
            print( '\t break_steps     = %i' % int( self.break_steps ) )
            print( '\t write_to        = %s' % self.write_to )
            
        if self.write_to == 'txt':
            self.write_trajectories = self.write_to_txt
        elif self.write_to == 'vtk':
            self.write_trajectories = self.write_to_vtk
        elif self.write_to == 'h5':
            self.write_trajectories = self.write_to_h5
        else:
            print('advancer: Error: write_to = %s not recognized, cannot write to file' % self.write_to )
            self.write_trajectories = self.write_to_null

        if self.test == False:
            self.evaluate_rhs = self.evaluate_particle_rhs
        else:
            self.evaluate_rhs = self.evaluate_test_rhs
            
        # Initial conditions manager
        if config.has_key('initial_conditions_options'):
            self.init_conditions_mgr = initial_conditions_manager( self.num_dim, config )
            self.generate_init_conditions = True
        else:
            self.generate_init_conditions = False

        # VTK writer
        self.io_counter = 0
        self.vtk_writer = vtk_writer()
        
        return

    def set_time_step(self, time_step):

        self.dt = time_step
        return

    def set_num_steps(self, num_steps):

        self.num_steps = num_steps
        return
    
    def set_domain(self, sim_domain):
            
        if self.domain_set == True:
            print('advancer: Warning: domain is already set, proceeding anyway')
            self.domain_set = False
            
        self.domain = sim_domain
        self.domain_set = True
        self.subdomain  = self.domain.has_subdomain
        
        return  
    
    def set_initial_conditions(self, particle_positions):
        
        self.state = particle_positions.ravel()
        self.check_positions()

        if self.num_secondary_fields > 0:
            self.domain.compute_interpolation_weights( self.state, self.interp_indices, self.interp_weights )
        
        self.particle_trajectories = {}
        self.particle_velocities   = {}
        for i_particle in range( 0, self.num_particles ):

            # Position and velocity
            self.particle_trajectories[i_particle] = [particle_positions[i_particle].copy()]
            self.particle_velocities[i_particle]   = [np.zeros(self.num_dim).copy()]

            # Secondary fields
            if self.num_secondary_fields > 0:
                indices = self.interp_indices[ i_particle, : ]
                weights = self.interp_weights[ i_particle, : ]
                secondary_fields = self.domain.interpolate_secondary_fields( indices, weights )
                for field_name in secondary_fields:
                    self.secondary_fields[ field_name ][i_particle] = [secondary_fields[field_name]]

            # Particle markers and timers
            self.particle_markers[i_particle] = False
                    
            if self.ver == 1:
                x_0 = self.particle_trajectories[i_particle][0][0]
                y_0 = self.particle_trajectories[i_particle][0][1]
                message  = 'advancer: set_initial_conditions: i_particle = %i ...'
                message += ' x = %.4E ... y = %.4E ...'
                if self.num_dim == 2:
                    print(  message % ( i_particle, x_0, y_0 ) )
                elif self.num_dim == 3:
                    z_0 = self.particle_trajectories[i_particle][0][2]
                    message += ' z = %.4E'
                    print(message % ( i_particle, x_0, y_0, z_0 ) )

        return

    def generate_initial_conditions(self):

        iret = 0
        if self.generate_initial_conditions == False:
            print('advancer: Error: No config for init condition generation was provided, cannot proceed')
            return iret
        self.init_conditions_mgr.generate()
        particle_positions = self.init_conditions_mgr.init_conditions.copy()
        self.set_initial_conditions( particle_positions )

        return
    
    def evaluate_particle_rhs(self, stage, time, state, state_rhs):

        if self.ver == 2:
            print( 'advancer: evaluate_rhs: stage %i' % stage )
        
        self.domain.compute_interpolation_weights( state, self.interp_indices, self.interp_weights )
        
        iret = 0
        
        for i_particle in range( 0, self.num_particles ):

            i_start = i_particle * self.num_dim
            i_end   = ( i_particle + 1 ) * self.num_dim
            
            indices = self.interp_indices[ i_particle, : ]
            weights = self.interp_weights[ i_particle, : ]            

            self.position = state[ i_start:i_end ]
            p_in = self.check_particle_position()
            
            particle_rhs  = np.zeros( self.num_dim )
            if p_in == True:
                particle_rhs = self.domain.interpolate_fields( indices, weights )

            state_rhs[ i_start:i_end ] = particle_rhs
            
        return iret        

    def evaluate_test_rhs(self, stage, time, state, state_rhs):

        if self.ver == 2:
            print( 'advancer: evaluate_rhs: stage %i' % stage )

        iret = 0

        rhs = np.zeros( self.state_dim )
        rhs[0] =  4.0 * time ** 3.0
        rhs[1] =  5.0 * time ** 4.0
        rhs[2] = -state[2]
        if stage < 5:
            self.stage_k[stage-1] = rhs
        else:
            self.state_rhs = rhs
            
        return iret
    
    def advance(self):

        if self.ver == 2:
            print('advancer: advance')

        # k_0 = dt * f( y_0 )
        self.stage_state = self.state.copy()
        time = self.t        
        iret = self.evaluate_rhs( 1, time, self.stage_state, self.stage_k[0] )
        
        # k_1 = dt * f( y_0 + 0.5 * k_0 )
        self.stage_state = self.state + 0.5 * self.dt * self.stage_k[0]
        time = self.t + 0.5 * self.dt
        iret = self.evaluate_rhs( 2, time, self.stage_state, self.stage_k[1] )
        
        # k_2 = dt * f( y_0 + 0.5 * k_1 )
        self.stage_state = self.state + 0.5 * self.dt * self.stage_k[1]
        time = self.t + 0.5 * self.dt
        iret = self.evaluate_rhs( 3, time, self.stage_state, self.stage_k[2] )
        
        # k_3 = dt * f( y_0 + k_2 ) 
        self.stage_state = self.state + self.dt * self.stage_k[2]
        time = self.t + self.dt
        iret = self.evaluate_rhs( 4, time, self.stage_state, self.stage_k[3] )
        
        self.state += self.dt * ( self.stage_k[0] + 2.0 * self.stage_k[1] + 2.0 * self.stage_k[2] + self.stage_k[3] ) / 6.0
        
        return iret
    
    def check_particle_position(self, subdomain = False):

        # Check that particle is inside domain
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        y_min = self.domain.y_min
        y_max = self.domain.y_max
        if self.num_dim == 3:
            z_min = self.domain.z_min
            z_max = self.domain.z_max
                
        particle_x = self.position[0]
        particle_y = self.position[1]
        if self.num_dim == 3:
            particle_z = self.position[2]
              
        x_in = ( particle_x >= x_min ) and ( particle_x <= x_max )
        y_in = ( particle_y >= y_min ) and ( particle_y <= y_max )
        particle_in = x_in * y_in
        
        if self.num_dim == 3:            
            z_in = ( particle_z >= z_min ) and ( particle_z <= z_max )
            particle_in *= z_in

        output = particle_in
        
        # Check that particle is inside subdomain (but only once)
        particle_sub_in = False
        if subdomain == True and self.marker == False:            
            x_min = self.domain.subdomain_x[0]
            x_max = self.domain.subdomain_x[1]
            y_min = self.domain.subdomain_y[0]
            y_max = self.domain.subdomain_y[1]
            if self.num_dim == 3:
                z_min = self.domain.subdomain_z[0]
                z_max = self.domain.subdomain_z[1]
            x_in = ( particle_x >= x_min ) and ( particle_x <= x_max )
            y_in = ( particle_y >= y_min ) and ( particle_y <= y_max )
            particle_sub_in = x_in * y_in
            if self.num_dim == 3:
                z_in = ( particle_z >= z_min ) and ( particle_z <= z_max )
                particle_sub_in *= z_in
            #print particle_x, x_min, particle_sub_in
        if subdomain == True:
            output = [particle_in, particle_sub_in]
                
        return output
    
    def check_positions(self):
        
        ###
        ### Check that particles are still in domain
        ###
        
        # Loop over particles
        for i_particle in range( 0, self.num_particles ):
            
            # Extract particle position
            i_start = i_particle * self.num_dim
            i_end   = ( i_particle + 1 ) * self.num_dim
            self.position = self.state[ i_start:i_end ]
            # Check individual particle
            self.particle_in[i_particle] = self.check_particle_position()
            
        return

    def save_trajectories(self):
        
        ###
        ### Save trajectories
        ###
        
        if self.ver == 2:
            print( 'advancer: Saving trajectories')
        
        # Evaluate RHS
        self.evaluate_rhs( 5, self.t, self.state, self.state_rhs )

        # Loop over particles
        for i_particle in range( 0, self.num_particles ):
            
            # Extract particle position & velocity
            i_start = i_particle * self.num_dim
            i_end   = ( i_particle + 1 ) * self.num_dim
            self.position = self.state[ i_start:i_end ]
            self.velocity = self.state_rhs[ i_start:i_end ]

            indices = self.interp_indices[ i_particle, : ]
            weights = self.interp_weights[ i_particle, : ]
            
            if self.particle_in[i_particle] == True:
                # Append to trajectory
                trajectory = self.particle_trajectories[i_particle]
                trajectory = np.append( trajectory, [self.position], axis = 0 )
                self.particle_trajectories[i_particle] = trajectory
                # Append to velocity
                velocity = self.particle_velocities[i_particle]
                velocity = np.append( velocity, [self.velocity], axis = 0 )
                self.particle_velocities[i_particle] = velocity
                # Compute secondary fields
                if self.num_secondary_fields > 0:
                    secondary_fields = self.domain.interpolate_secondary_fields( indices, weights )
                    for field_name in secondary_fields:
                        field = secondary_fields[ field_name ]
                        field_trajectory = self.secondary_fields[ field_name ][i_particle]
                        field_trajectory = np.append( field_trajectory, [field], axis = 0 )
                        self.secondary_fields[ field_name ][i_particle] = field_trajectory
                # Particle timers
                
        return
            
    def compute_residence_times(self):
        
        for i_particle in range( 0, self.num_particles ):
            self.time_elapsed[i_particle]       = np.array([0.0])
            self.particle_markers[ i_particle ] = False
        
        for i_particle in range( 0, self.num_particles ):
            X = self.particle_trajectories[ i_particle ]
            V = self.particle_velocities[ i_particle ]
            self.num_saved_steps = len( self.particle_velocities[i_particle] )
            for i_step in range( 1, self.num_saved_steps ):
                # Compute time delta
                dX    = X[ i_step ] - X[ i_step - 1 ]
                disp  = np.sqrt( np.dot( dX, dX ) )
                speed = np.sqrt( np.dot( V[i_step],  V[i_step]  ) )
                time_delta = disp / speed
                # If subdomain specified, check that particle is inside
                if self.subdomain == True:
                    self.position = X[i_step]
                    self.marker   = self.particle_markers[ i_particle ]
                    [p_in, p_sub_in] = self.check_particle_position( subdomain = True )
                    # If particle not in subdomain *yet*, do not aggregate                    
                    if self.marker == False and p_sub_in == False:
                        time_delta = 0.0
                    if self.marker == False and p_sub_in == True:
                        self.particle_markers[ i_particle ] = True
                # Append elapsed time
                time  = self.time_elapsed[i_particle]                                
                time  = np.append( time, [ time[-1] + time_delta ], axis = 0 )
                self.time_elapsed[i_particle] = time

            self.residence_times[i_particle] = self.time_elapsed[i_particle][-1]
        return

    def compute_current_residence_times(self):

        for i_particle in range( 0, self.num_particles ):
            # Get position and velocity histories
            X = self.particle_trajectories[ i_particle ]
            V = self.particle_velocities[ i_particle ]
            # Compute time delta
            dX = 0.0
            U  = 1.0
            if len( X ) > 1:
                dX = X[-1] - X[-2]
            if len( V ) > 1:
                U  = V[-1]
            disp  = np.sqrt( np.dot( dX, dX ) )
            speed = np.sqrt( np.dot( U,  U  ) )
            time_delta = disp / speed
            # If subdomain specified, check that particle is inside
            if self.subdomain == True:
                self.position = X[-1]
                self.marker   = self.particle_markers[ i_particle ]
                [p_in, p_sub_in] = self.check_particle_position( subdomain = True )
                # If particle not in subdomain *yet*, do not aggregate                    
                if self.marker == False and p_sub_in == False:
                    time_delta = 0.0
                if self.marker == False and p_sub_in == True:
                    self.particle_markers[ i_particle ] = True
            # Save
            self.residence_times[ i_particle ] += time_delta
            
    def run(self):
        
        ###
        ### Compute streamlines 
        ###
        
        p_in = int(self.particle_in.sum())
        print( 'advancer: Stepping begins ')
        print( 'advancer: Number of particles: %i ' % self.num_particles )
        print( 'advancer: Number of particles inside domain: %i' % p_in )

        self.io_count = 0
        if self.write_to == 'vtk':
            self.write_trajectories()
        
        self.num_steps = int( self.num_steps )
        
        i_step = 0
        self.t = 0.0
        while p_in > 0:
            
            # Advance
            self.advance()            
            i_step += 1
            self.t += self.dt              
            self.check_positions()
            
            # Number of particles still in the domain
            p_in = int( self.particle_in.sum() )
            
            # Report
            if i_step % self.num_steps_print == 0:
                print('advancer: step = %i ... t = %.4E ... particles inside: %i' % ( i_step, self.t, p_in ) )
            
            # Save
            if i_step % self.num_steps_save == 0 and i_step > 0:                                
                self.save_trajectories()
                self.compute_current_residence_times()

            # Write
            if i_step % self.num_steps_write == 0 and self.write_to == 'vtk':
                self.write_trajectories()

            # Steps
            if i_step >= self.num_steps and self.break_steps:
                break

        print( 'advancer: Stepping ends: step = %i ... t = %.4E ... particles inside: %i' % ( i_step, self.t, p_in ) )
        
        if self.residence_time == True:
            self.compute_residence_times()

        #self.write_trajectories()
            
        return    

    def write_to_txt(self, finalize = True, save_trajectory = False, save_velocity = False,
                     save_secondary_fields = False):

        iret = 0

        # assemble file name: prepend write directory to file-name root
        file_root  = self. out_dir + 'streamlines_'

        # assemble file name: append file ids        
        num_id = len( self.file_id )
        for i_id in range( 0, num_id ):
            file_root += self.file_id[ i_id ] + '_'

        # file header 
        header = ''
        if self.num_dim == 2:
            header = 'X Y'
        elif self.num_dim == 3:
            header = 'X Y Z'

        # loop over particles
        for i_particle in range( 0, self.num_particles ):
            num_time = len( self.time_elapsed[ i_particle ] )
            # particle position
            if save_trajectory == True:
                file_name = file_root + 'position_particle_' + str( i_particle ) + '.dat'
                file_data = np.zeros( [ num_time, self.num_dim + 1 ] )
                file_data[:,0]  = self.time_elapsed[ i_particle ]
                file_data[:,1:] = self.particle_trajectories[ i_particle ]
                print('advancer: Saving particle trajectories to output file: %s' % file_name)
                np.savetxt( file_name, file_data, header = '', comments = '' )
            # particle velocity
            if save_velocity == True:
                file_name = file_root + 'velocity_particle_' + str( i_particle ) + '.dat'
                file_data = np.zeros( [ num_time, self.num_dim + 1 ] )
                file_data[:,0]  = self.time_elapsed[ i_particle ]
                file_data[:,1:] = self.particle_velocities[ i_particle ]
                print('advancer: Saving particle velocities to output file: %s' % file_name)
                np.savetxt( file_name, file_data, header = '' , comments = '' )
            if save_secondary_fields == True and self.num_secondary_fields > 0:
                for field_name in self.secondary_fields:
                    file_name = file_root + field_name + '_particle_'  + str( i_particle ) + '.dat'
                    file_data = np.zeros( [ num_time, 2 ] )
                    file_data[:,0] = self.time_elapsed[ i_particle ]
                    file_data[:,1] = self.secondary_fields[ field_name ][ i_particle ]
                    print('advancer: Saving secondary field %s to output file: %s' % ( field_name, file_name ) )
                    np.savetxt( file_name, file_data, header = '', comments = '' )
                          
        if self.residence_time == True:
            file_name = file_root + 'residence_times.dat'
            file_data = np.zeros( [ self.num_particles, self.num_dim + 2 ] )
            file_data[:,0]                = np.linspace( 1.0, float(self.num_particles), self.num_particles )
            file_data[:,1:self.num_dim+1] = self.init_conditions_mgr.init_conditions
            file_data[:,self.num_dim+1]   = self.residence_times
            print('advancer: Saving residence times to output file: %s' % file_name)
            np.savetxt( file_name, file_data, header = '', comments = '' )
            
        return iret

    def write_to_vtk(self, finalize = False, save_trajectory = False, save_velocity = False,
                     save_secondary_fields = False):

        iret = 0
        #
        file_root = self.out_dir + 'streamlines_'

        num_id = len( self.file_id )
        for i_id in range( 0, num_id ):
            file_root += self.file_id[ i_id ]
            if i_id != num_id - 1:
                file_root += '_'
                
        if finalize == True:
            file_name = file_root + '.pvd'
            self.vtk_writer.write_pvd( file_name )
        else:                
            file_name = file_root + '_' + str( self.io_count ) + '.vtu'
            X = self.state[0::self.num_dim]
            Y = self.state[1::self.num_dim]                
            Z = self.state[2::self.num_dim]
            #res_time   = self.time_elapsed[ i_particle ]
            self.vtk_writer.snapshot( file_name, X, Y, Z, residence_time = self.residence_times )        
        
        self.io_count += 1
        return iret
    
    def write_to_h5(self, finalzie = True, save_trajectory = False, save_velocity = False):

        iret = 0
        
        print('advancer: Warning: write_to_h5 does not work yet')
        iret = -1
        
        return iret
    
    def write_to_null(self, finalize = True, save_trajectory = False, save_velocity = False):

        print('advancer: Error: write_to = %s not recognized, cannot write to file' % self.write_to )
        iret = -1
        return iret

    def clear(self):

        self.particle_trajectories = {}
        self.particle_velocities   = {}

        self.state     = np.zeros( self.state_dim )
        self.state_rhs = np.zeros( self.state_dim )

        self.stage_state = np.zeros( self.state_dim )
        self.stage_k     = np.zeros( [ 4, self.state_dim ] )

        return
