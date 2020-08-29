import numpy as np

if __name__ == '__main__':

    print('inflow_particles: this util postprocess dumps from main runs to find inflow particles')

    num_lines     = 23
    num_particles = 201

    nominal_speed = 1300.0
    cavity_length = 0.07
    cavity_time   = cavity_length / nominal_speed
    
    x_inflow = 0.65163
    file_dir = '../outputs/'
    trapped_particles = np.array([])

    for i_line in range( 1, num_lines + 1 ):
    
        for i_particle in range( 2, 198 ):

            file_head = 'streamlines_actii-3D_inert-halfx_freestream_line_' + str( i_line )
            file_tail = '_position_particle_' + str( i_particle ) + '.dat'
            file_name = file_head + file_tail

            data  = np.loadtxt( file_dir + file_name )
            shape = data.shape
            if len( shape ) > 1:
                time = data[:,0]
                xpos = data[:,1]
                ypos = data[:,2]
                zpos = data[:,3]
                tick = time.nonzero()[0]
            else:
                tick = []
                
            if len( tick ) > 0:
                i_right = tick[0]
                i_left  = i_right - 1
                
                x_left  = xpos[ i_left  ]
                x_right = xpos[ i_right ]
                
                y_left  = ypos[ i_left  ]
                y_right = ypos[ i_right ]
                
                z_left  = zpos[ i_left  ]
                z_right = zpos[ i_right ]
                
                y_slope = ( y_right - y_left ) / ( x_right - x_left )
                y_plane = y_left + y_slope * ( x_inflow - x_left )
                
                z_slope = ( z_right - z_left ) / ( x_right - x_left )
                z_plane = z_left + z_slope * ( x_inflow - x_left )
                
                residence_time = time[-1]
                log_res_time   = np.log10( residence_time / cavity_time )
            
                if log_res_time > 1.1:
                    trapped_particles = np.append( trapped_particles, [y_plane,z_plane], axis = 0 )
                    message = 'inflow_particles: Line %i: Particle %i trapped with log res time %.4E'
                    print( message % ( i_line, i_particle, log_res_time ) )

    print trapped_particles
    np.savetxt( file_dir + 'particles_trapped_particles_cavity_plane.dat', trapped_particles )
