import sys
sys.path.append('../src/')
from PC2H5_manager import *
from particle_advancer import *

if __name__ == '__main__':

    tol = 1.0e-1
    print('test_RK4: This driver tests the RK4 integrator against known functions with rel. tol. = %.4E' % tol)
    print('test_RK4: initializing')

    ###
    ### Config
    ###
    time_step  = 1.0
    final_time = 10.0
    num_steps  = 1 #int( final_time / time_step )
    
    config = {}
    config['advancer_options'] = {}
    config['advancer_options']['test'] = True
    config['advancer_options']['verbosity']       = 1
    config['advancer_options']['num_particles']   = 1
    config['advancer_options']['num_steps']       = num_steps
    config['advancer_options']['time_step']       = time_step
    config['advancer_options']['num_steps_save']  = 1
    config['advancer_options']['num_steps_print'] = 100000
    config['advancer_options']['residence_time']  = False
    config['advancer_options']['break_steps']     = True
    config['advancer_options']['write_to']        = 'txt'
    config['advancer_options']['file_id']         = ['test','RK4','1']
    config['advancer_options']['out_dir']         = './'
    
    ###
    ### Create grid (and fields)
    ###
    x_min =  0.0
    x_max =  2.0

    y_min =  0.0
    y_max =  2.0

    z_min =  0.0
    z_max =  2.0

    N_x = 2
    N_y = 2
    N_z = 2

    x = np.linspace( x_min, x_max, N_x )
    y = np.linspace( y_min, y_max, N_y )
    z = np.linspace( z_min, z_max, N_z )

    yy,zz,xx = np.meshgrid( y, z, x )

    grid = {}
    grid['X'] = xx
    grid['Y'] = yy
    grid['Z'] = zz

    fields = {}
    fields['velocity-1'] = xx
    fields['velocity-2'] = yy
    fields['velocity-3'] = zz

    ###
    ### Simulation domain
    ###
    domain = simulation_domain( config, grid, fields, {} )

    ###
    ### Particle advancer
    ###
    advancer = particle_advancer( config, domain )

    ###
    ### Initial conditions
    ###
    num_particles = advancer.num_particles
    particle_positions = np.zeros( [ num_particles, 3 ] )
    particle_positions[:,2] = 1.0

    ###
    ### Test
    ###
    num_solves = 8
    old_error  = np.ones( 3 )

    failed = False
    for i_solve in range( 1, num_solves + 1 ):

        # Set time step
        time_step = 1.0 / ( 2.0 ** float(i_solve ) )
        advancer.set_time_step( time_step )
        # Set initial condition
        advancer.set_initial_conditions( particle_positions.copy() )
        # Advance
        advancer.run()        
        # Get computed solution
        solution_integ = advancer.particle_trajectories[0]
        # Get time instances
        time = np.array([0.0, time_step])
        # Compute real solution
        solution_real = np.zeros( [2, 3] )
        solution_real[:,0] = time ** 4.0
        solution_real[:,1] = time ** 5.0
        solution_real[:,2] = np.exp( -time )
        # Compute error
        new_error = np.abs( solution_integ[1,:] - solution_real[1,:] )
        # Clear advancer
        advancer.clear()
        # Slope
        slopes = np.zeros( 2 )
        orders = np.zeros( 2 )
        slopes = np.log( new_error[1:] / old_error[1:] )
        orders = -slopes / np.log( 2.0 )
        old_error = new_error
        # Report
        message = 'test_RK: solve : %i ... dt = %.4E ... error_1 = %.4E ... error_1 = %.4E ... error_2 = %.4E'
        if i_solve > 1:
            message += ' ... order_1 = %.4E ... order_2 = %.4E'
            print( message % ( i_solve, time_step, new_error[0], new_error[1], new_error[2], orders[0], orders[1] ) )
        else:
            print( message % ( i_solve, time_step, new_error[0], new_error[1], new_error[2] ) )
        # Assess
        order_error = np.abs( orders - 5.0 )
        if i_solve > 1 and new_error[0] != 0.0 and order_error[0] > tol and order_error[1] > tol:
            failed = True
            #break

    if failed == False:
        color = '\033[92m'
        print('test_RK4: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_RK4: ' + '\033[91m' + 'test failed ' + '\033[0m' + '... check your integrator')
