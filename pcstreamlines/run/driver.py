import sys
sys.path.append('../src/')
from PC2H5_manager import *
from particle_advancer import *

if __name__ == '__main__':

    print('driver: initializing')


    num_argv = len( sys.argv )
    if num_argv != 2:
        print('driver: Error: must specify input file')
        print('driver: Run command is: python driver input_file.yaml')
        print('driver: Cannot continue')
        sys.exit()
    
    ###
    ### Config
    ###
    config_file = sys.argv[1]
    config_mgr  = configuration_manager( config_file )
    config      = config_mgr.get_config()
    
    ###
    ### Options
    ### 
    grid_dict = {}
    grid_dict['Block'] = 6
    
    ###
    ### Dump manager
    ###
    dump_mgr = pcdump_manager( grid_dict, config )
    dump_mgr.open_database()
    dump_mgr.load_grid()
    dump_mgr.load_grid_fields()

    iret, grid   = dump_mgr.get_grid()
    iret, fields = dump_mgr.get_fields()
    iret, secondary_fields = dump_mgr.get_secondary_fields()
    
    ###
    ### Simulation domain
    ###
    domain = simulation_domain( config, grid, fields, secondary_fields )
    
    ###
    ### Particle advancer
    ###
    advancer = particle_advancer( config, domain )
    
    ###
    ### Run
    ###
    advancer.generate_initial_conditions()
    advancer.run()

    ###
    ### Write to file
    ###
    iret = advancer.write_trajectories( finalize = True, save_trajectory = True, save_secondary_fields = True )
    
