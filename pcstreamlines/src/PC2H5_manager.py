from configuration_manager import *
import numpy as np
import h5py

class pcdump_manager():


    def __init__(self, grid_dict, config, ver = 1):

        self.open_db      = False
        self.grid_ready   = False
        self.fields_ready = False
        self.dict_ready   = False
        self.config_ready = False
        
        self.grid_dict  = grid_dict
        self.dict_ready = True
        
        self.database_dir  = config['dump_options']['database_dir']
        self.database_name = config['dump_options']['database_name']
        self.geometry_name = config['dump_options']['geometry_name']
        self.domain_name   = config['dump_options']['domain_name']
        self.grid_name     = config['dump_options']['grid_name']

        self.symmetrize = False
        if config['dump_options'].has_key( 'symmetrize' ):
            self.symmetrize = config['dump_options']['symmetrize']

        self.secondary_fields       = {}
        self.secondary_fields_names = []
        if config['dump_options'].has_key( 'secondary_fields' ):
            self.secondary_fields_names = config['dump_options']['secondary_fields']
        
        self.config_ready  = True

        if ver == 1:
            print('pcdump_mgr: Configuration options ready:')
            print( '\t database_dir  = %s' % self.database_dir )
            print( '\t database_name = %s' % self.database_name )
            print( '\t geometry_name = %s' % self.geometry_name )
            print( '\t domain_name   = %s' % self.domain_name )
            print( '\t grid_name     = %s' % self.grid_name )
            if self.symmetrize == True:
                print( '\t symmetrize    = yes' )
            else:
                print( '\t symmetrize    = no' )
            
        return
    
    def set_config(self, db_dir, db_name, geometry_name, domain_name):

        if self.config_ready == True:
            print('pcdump_mgr: Warning: config is already set, but will be over-written')            
            self.config_ready = False
            
        self.database_dir  = db_dir
        self.database_name = db_name
        self.geometry_name = geometry_name
        self.domain_name   = domain_name
        self.config_ready  = True
        
        return
        
    def open_database(self):

        iret = 0
        self.open_db  = False

        if self.config_ready == False:
            print('pcdump_mgr: Warning: config not set, cannot proceed')
            iret = -1
            return iret
        
        filename = self.database_dir + self.database_name + ".h5"
        database = h5py.File( filename, 'r' )

        PlasCom2   = database[ u'PlasCom2' ]
        geometry   = PlasCom2[ u'Geometry' ]
        simulation = PlasCom2[ u'Simulation' ]

        self.grids  = geometry[ self.geometry_name ]
        self.domain = simulation[ self.domain_name ]

        self.open_db = True
        return iret

    def load_grid(self):

        iret = 0
        self.grid_ready = False
        
        self.group_name = u'' + self.grid_name
        self.grid_id    = u'grid'

        grid_id = self.grid_dict[ self.grid_name ]
        self.grid_id += str( grid_id )

        self.num_dim = len( self.grids[ self.group_name ] )
        
        self.grid = {}
        if self.num_dim == 2:
            self.grid['X'] = self.grids[ self.group_name ][ u'X' ][:,:]
            self.grid['Y'] = self.grids[ self.group_name ][ u'Y' ][:,:]
        elif self.num_dim == 3:
            self.grid['X'] = self.grids[ self.group_name ][ u'X' ][:,:,:]
            self.grid['Y'] = self.grids[ self.group_name ][ u'Y' ][:,:,:]
            self.grid['Z'] = self.grids[ self.group_name ][ u'Z' ][:,:,:]

        # Symmetrize grids in 3D
        if self.num_dim == 3 and self.symmetrize == True:
            N_z = self.grid['Z'].shape[0]
            k_start = N_z//2
            k_end   = N_z
            self.grid['X'] = self.grid['X'][ k_start:k_end,:,: ]
            self.grid['Y'] = self.grid['Y'][ k_start:k_end,:,: ]
            self.grid['Z'] = self.grid['Z'][ k_start:k_end,:,: ]
            
        self.grid_ready = True
        
        return iret
        
    def load_grid_fields(self):

        iret = 0
        self.fields_ready = False

        if self.grid_ready == False:
            print('pcdump_mgr: Warning: Grid has not been loaded yet, cannot proceed')
            iret = -1
            return iret
        
        self.data = self.domain[ self.grid_id ]

        # Set fields
        self.fields = {}
        if self.num_dim == 2:
            self.fields[ 'velocity-1' ] = self.data[ 'velocity-1' ][:,:]
            self.fields[ 'velocity-2' ] = self.data[ 'velocity-2' ][:,:]
        if self.num_dim == 3:
            self.fields[ 'velocity-1' ] = self.data[ 'velocity-1' ][:,:,:]
            self.fields[ 'velocity-2' ] = self.data[ 'velocity-2' ][:,:,:]
            self.fields[ 'velocity-3' ] = self.data[ 'velocity-3' ][:,:,:]

        # Set secondary fields
        self.secondary_fields = {}
        if len( self.secondary_fields_names ) > 0:
            for field_name in self.secondary_fields_names:            
                if self.num_dim == 2:
                    self.secondary_fields[ field_name ] = self.data[ field_name ][:,:]
                if self.num_dim == 3:
                    self.secondary_fields[ field_name ] = self.data[ field_name ][:,:,:]
            
        # Symmetrize fields in 3D
        if self.num_dim == 3 and self.symmetrize == True:
            # Z indices
            N_z = self.fields['velocity-1'].shape[0]
            k_start = 0
            k_mid   = N_z//2
            k_end   = N_z
            # Coefficients
            coeff = [ 1.0, 1.0, -1.0 ]
            # Symmetrize fields
            for i_field in range( 0, self.num_dim ):
                field_name = 'velocity-' + str( i_field + 1 )
                field_aft  = np.flip( self.data[ field_name ][ k_start:k_mid,:,: ], axis = 0 )
                field_fore = self.data[ field_name ][ k_mid:k_end,:,: ]
                field_symm = 0.5 * ( coeff[ i_field ] * field_aft + field_fore )
                self.fields[ field_name ] = field_symm            
            # Symmetrize secondary fields
            for field_name in self.secondary_fields_names:                
                field_aft  = np.flip( self.data[ field_name ][ k_start:k_mid,:,: ], axis = 0 )
                field_fore = self.data[ field_name ][ k_mid:k_end,:,: ]
                field_symm = 0.5 * ( field_aft + field_fore )
                self.secondary_fields[ field_name ] = field_symm
            
        self.fields_ready = True

        return iret

    def get_grid(self):

        iret = 0
        if self.grid_ready == False:
            print('pcdump_mgr: Warning: Grid has not been loaded yet, cannot proceed')
            iret = -1
            return iret,[]
        
        return iret, self.grid
    
    def get_fields(self, Favre = False):

        iret = 0
        if self.fields_ready == False:
            print('pcdump_mgr: Warning: Data has not been loaded yet, cannot proceed')
            iret = -1
            return iret,[]
        
        output = self.fields
        if Favre == True:
            output = self.Favre_fields

        return iret,output

    def get_secondary_fields(self):

        iret = 0
        if self.fields_ready == False:
            print('pcdump_mgr: Warning: Data has not been loaded yet, cannot proceed')
            iret = -1
            return iret,[]
        
        output = self.secondary_fields
        return iret,output
