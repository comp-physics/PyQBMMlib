import numpy as np

class initial_conditions_manager():

    ###
    ### This class generates particle initial conditions.
    ### There are four options:
    ### (1) point
    ### (2) line
    ### (3) plane
    ### (4) circle
    ###

    def __init__(self, num_dim, config):

        iret = 1
        self.config_ready = False
        
        ###
        ### Dimensionality
        ### (only non-config variable)
        ###
        self.num_dim = num_dim
        
        ###
        ### Number of particles
        ### (only option from advancer config)
        ### 
        self.num_particles = config['advancer_options']['num_particles']


        ###
        ### Buffers
        ###
        self.init_conditions = np.zeros( [ self.num_particles, self.num_dim ] )
        
        ###
        ### Generator config
        ###

        # (1) Geometry
        self.geometry = config['initial_conditions_options']['geometry']

        if self.geometry == 'point' and self.num_particles != 1:
            message = 'init_conditions_mgr: Error: [point] option only works with one particle (specified %i)'
            print(message % self.num_particles)
            message  = 'init_conditions_mgr: Error: If you need to solve for multiple particles,'
            message += ' [line, plane, circle] options'
            print(message)
            return
                
        # (2) Origin, or base point  (used differently across geometries)        
        self.origin = config['initial_conditions_options']['origin']
        
        # (2) Geometry-dependent options
        if self.geometry == 'line':
            self.extent    = config['initial_conditions_options']['extent']
            self.direction = config['initial_conditions_options']['direction']            
        if self.geometry == 'plane':
            self.extent_X1  = config['initial_conditions_options']['extent_X1']
            self.extent_X2  = config['initial_conditions_options']['extent_X2']
            self.directions = config['initial_conditions_options']['directions']
            self.num_particles_X1 = config['initial_conditions_options']['num_particles_X1']            
        if self.geometry == 'circle':
            self.radius = config['initial_conditions_options']['radius']
            self.num_particles_radius = config['initial_conditions_options']['num_partiles_radius']

        self.config_ready = True
            
        return

    def generate(self):

        iret = 0
        
        if self.config_ready == False:
            print('init_conditions_mgr: Error: Generate: Configuration not set, cannot proceed')
            iret = -1
            return iret

        if self.geometry == 'point':
            iret = self.generate_point()
        elif self.geometry == 'line':
            iret = self.generate_line()
        elif self.geometry == 'plane':
            iret = self.generate_plane()
        elif self.geometry == 'circle':
            iret = self.generate_circle()

        return iret
            
    def generate_point(self):

        iret = 0
        self.init_conditions[0] = np.array( self.origin )
        return iret

    def generate_line(self):

        iret = 0

        if self.direction == 'x':
            idx = 0
        elif self.direction == 'y':
            idx = 1
        elif self.direction == 'z':
            idx = 2
        
        line_min = self.origin[idx] - 0.5 * self.extent
        line_max = self.origin[idx] + 0.5 * self.extent

        line_coords = np.linspace( line_min, line_max, self.num_particles )

        for i_particle in range( 0, self.num_particles ):

            if self.direction == 'x':
                #
                if self.num_dim == 2:
                    self.init_conditions[i_particle,0] = line_coords[i_particle]
                    self.init_conditions[i_particle,1] = self.origin[1]
                elif self.num_dim == 3:
                    self.init_conditions[i_particle,2] = self.origin[2]
                #
            elif self.direction == 'y':
                #
                if self.num_dim == 2:
                    self.init_conditions[i_particle,0] = self.origin[0]
                    self.init_conditions[i_particle,1] = line_coords[i_particle]
                elif self.num_dim == 3:
                    self.init_conditions[i_particle,2] = self.origin[2]
                #
            elif self.direction == 'z':
                #
                self.init_conditions[i_particle,0] = self.origin[0]
                self.init_conditions[i_particle,1] = self.origin[1]
                self.init_conditions[i_particle,2] = line_coords[i_particle]
                #

        return iret
                
    def generate_plane(self):

        iret = 0

        self.num_particles_X2 = self.num_particles // self.num_particles_X1
        if self.num_particles % self.num_particles_X1 != 0:
            messaage = 'init_conditions_mgr: Error: Plane: Cannot generate initial condition ... '
            message += 'num_particles(%i) is not divisible by num_particles_plane(%i)'
            print( message % ( self.num_particles, self.num_particles_X1 ) )
            iret = -1
            return iret        

        if self.directions == 'xy':
            idx_1 = 0
            idx_2 = 1
            idx_3 = 2
        elif self.directions == 'xz':
            idx_1 = 0
            idx_2 = 2
            idx_3 = 1
        elif self.directions == 'yz':
            idx_1 = 1
            idx_2 = 2
            idx_3 = 0

        plane_min_1 = self.origin[idx_1] - 0.5 * self.extent_X1
        plane_max_1 = self.origin[idx_1] + 0.5 * self.extent_X1

        plane_min_2 = self.origin[idx_2] - 0.5 * self.extent_X2
        plane_max_2 = self.origin[idx_2] + 0.5 * self.extent_X2
                
        plane_coords_1 = np.linspace( plane_min_1, plane_max_1, self.num_particles_X1 )
        plane_coords_2 = np.linspace( plane_min_2, plane_max_2, self.num_particles_X2 )

        grid_1, grid_2 = np.meshgrid( plane_coords_1, plane_coords_2 )

        self.init_conditions[:,idx_1] = grid_1.ravel()
        self.init_conditions[:,idx_2] = grid_2.ravel()
        self.init_conditions[:,idx_3] = self.origin[idx_3]
        return iret
    
    def generate_circle(self):

        iret = 0

        num_particles_theta = self.num_particles // self.num_particles_radius
        if self.num_particles % self.num_particles_X1 != 0:
            messaage = 'init_conditions_mgr: Error: Circle: Cannot generate initial condition ... '
            message += 'num_particles(%i) is not divisible by num_particles_radius(%i)'
            print( message % ( self.num_particles, self.num_particles_X1 ) )
            iret = -1
            return iret

        return iret
