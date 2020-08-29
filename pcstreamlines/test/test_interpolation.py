import sys
sys.path.append('../src/')
from PC2H5_manager import *
from particle_advancer import *

def analytical_field(point):

    F = -3.0 * point[0] #np.sin( 2.0 * np.pi * point[0] )
    G =  2.0 * point[1] #np.cos( 2.5 * np.pi * point[1] )
    H =  0.5 * point[2] #np.sin( 3.0 * np.pi * point[2] )
    U = F + G + H #F * G * H
    return U

if __name__ == '__main__':

    tol = 1.0e-12
    print('test_interpolation: This driver tests 3D interpolation against a linear field with rel. tol. = %.4E' % tol)
    print('test_interpolation: initializing')
    
    ###
    ### Create grid
    ###
    x_min = -1.0
    x_max =  1.0

    y_min =  0.0
    y_max =  3.0

    z_min =  0.0
    z_max =  1.0

    N_x = 100
    N_y = 300
    N_z = 50

    x = np.linspace( x_min, x_max, N_x )
    y = np.linspace( y_min, y_max, N_y )
    z = np.linspace( z_min, z_max, N_z )

    yy,zz,xx = np.meshgrid( y, z, x )

    grid = {}
    grid['X'] = xx
    grid['Y'] = yy
    grid['Z'] = zz

    ###
    ### Create fields
    ###
    grid_coordinates = np.zeros( [ N_x * N_y * N_z, 3 ] )
    grid_coordinates[:,0] = xx.ravel()
    grid_coordinates[:,1] = yy.ravel()
    grid_coordinates[:,2] = zz.ravel()

    F = -3.0 * xx.ravel() #np.sin( 2.0 * np.pi * xx.ravel() )
    G =  2.0 * yy.ravel() #np.cos( 2.5 * np.pi * yy.ravel() )
    H =  0.5 * zz.ravel() #np.sin( 3.0 * np.pi * zz.ravel() )
    U =  F + G + H        #F * G * H
    U = np.reshape( U, [ N_z, N_y, N_x ] )

    fields = {}
    fields['velocity-1'] = U
    fields['velocity-2'] = U
    fields['velocity-3'] = U

    ###
    ### Simulation domain
    ###
    domain  = simulation_domain( {}, grid, fields, {} )    
    
    ###
    ### Create particles
    ###
    num_particles = 10
    xyz_p = np.zeros( [ num_particles, 3 ] )
    xyz_p[:,0] = ( x_max - x_min ) * np.random.random_sample( num_particles ) + x_min
    xyz_p[:,1] = ( y_max - y_min ) * np.random.random_sample( num_particles ) + y_min
    xyz_p[:,2] = ( z_max - z_min ) * np.random.random_sample( num_particles ) + z_min

    points = xyz_p.ravel()

    
    ###
    ### Interpolate
    ###
    all_indices = np.zeros( [ num_particles, 3 ] )
    all_weights = np.zeros( [ num_particles, 3 ] )
    domain.compute_interpolation_weights( points, all_indices, all_weights )

    failed = False
    for i_particle in range( 0, num_particles ):

        indices = all_indices[i_particle]
        weights = all_weights[i_particle]
        [U_interp,_,_] = domain.interpolate_fields( indices, weights )
        U_real = analytical_field( xyz_p[i_particle] )
        error  = np.abs( U_real - U_interp )

        if error > tol:
            failed = True
            
        message  = 'test_interpolation: '
        message += 'particle : %i ... U_real = %+.5E .. U_interp = %+.5E ... error = %.5E [%%]'
        print( message % ( i_particle, U_real, U_interp, error * 100.0 ) )

    if failed == False:
        color = '\033[92m'
        print('test_interpolation: ' + '\033[92m' + 'test passed' + '\033[0m')
    else:
        print('test_interpolation: ' + '\033[91m' + 'test failed ' + '\033[0m' + '... check your interpolation')
