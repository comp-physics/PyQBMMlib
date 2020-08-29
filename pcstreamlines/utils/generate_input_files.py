import yaml
import numpy as np

###
### Note: This util is not a generalized input-file generator.
### It serves a very specific purpose: to generate input files
### to study streamline behavior in the 3D ACT-II simulations.
### 

def generate_default_options(database_name):

    # Dump options
    dump_options  = 'dump_options:\n'
    dump_options += '    database_dir:     "../H5-actii-3D/"'   + '\n'
    dump_options += '    database_name:    ' + database_name    + '\n'
    dump_options += '    geometry_name:    "geom3d-mf"'         + '\n'
    dump_options += '    domain_name:      "downstreamY5"'      + '\n'
    dump_options += '    grid_name:        "Block"'             + '\n'
    dump_options += '    symmetrize:       yes'                 + '\n'
    dump_options += '    secondary_fields: ["mixtureFraction"]' + '\n'
    
    # Advancer options
    advancer_options  = 'advancer_options:\n'
    advancer_options += '    verbosity:       1'             + '\n'
    advancer_options += '    num_particles:   201'           + '\n'
    advancer_options += '    num_steps:       100000'        + '\n'
    advancer_options += '    time_step:       1.0e-6'        + '\n'
    advancer_options += '    num_steps_save:  1'             + '\n'
    advancer_options += '    num_steps_print: 100'           + '\n'
    advancer_options += '    residence_time:  yes'           + '\n'
    advancer_options += '    break_steps:     yes'           + '\n'
    advancer_options += '    write_to:        "txt"'         + '\n'
    advancer_options += '    out_dir:         "../outputs/"' + '\n'

    # Domain options
    domain_options  = 'domain_options:\n'
    domain_options += '    subdomain_x: [0.65163,0.72163]'      + '\n'
    domain_options += '    subdomain_y: [-0.0283245,0.0036755]' + '\n'
    domain_options += '    subdomain_z: [0.0,0.0175]'           + '\n'
     
    # Initial condition options
    initial_condition_options  = 'initial_conditions_options:\n'
    initial_condition_options += '    geometry:  line'          + '\n'
    initial_condition_options += '    extent:    0.016'         + '\n' # 0.0345/2 for reactive case
    initial_condition_options += '    direction: z'             + '\n'

    return dump_options, advancer_options, domain_options, initial_condition_options

if __name__ == '__main__':

    print('input_file_util: this util generates a set of input files')

    # Case ID
    simulation  = 'inert'
    mesh_config = 'halfx'
    case_id     = simulation + '-' + mesh_config
    if case_id == 'inert-halfx':
        database_name = '"MeanFlow_001680000"'
    elif case_id == 'react-halfx':
        database_name = '"MeanFlow_000888000"'
    
    # Coordinates
    y_cavity      = -0.0083245
    cavity_height =  0.02

    num_lines  = 21
    delta_eta  = 0.005
    eta_min    = 0.005
    eta_max    = 0.100
    
    #num_lines  = 51
    #delta_eta  = 0.0025
    #eta_min    = 0.0050
    #eta_max    = 0.0800
    
    eta_coords = np.linspace( eta_min, eta_max, num_lines )
    y_coords   = y_cavity + cavity_height * eta_coords

    # Loop over lines
    for i_line in range( 0, num_lines ):

        dump_options, advancer_options, domain_options, init_options = generate_default_options( database_name )
        
        line_id = '"line_' + str( i_line + 1 ) + '"'
        advancer_options += '    file_id:         ["actii-3D","'+case_id+'","freestream",'+line_id+']' + '\n'

        origin = '[0.63163,' + str( y_coords[i_line] ) + ',0.008]'
        init_options += '    origin:    ' + origin

        config_options  = dump_options     + '\n'
        config_options += domain_options   + '\n'
        config_options += advancer_options + '\n'
        config_options += init_options
        print config_options
        print '------------------------------------------------------'

        file_name = '../inputs/actii_3D_freestream_line_' + str( i_line + 1 ) + '.yaml'
        with open( file_name, 'w' ) as file_handle:
            file_handle.write( config_options )
