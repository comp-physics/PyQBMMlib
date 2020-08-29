import numpy as np
import scipy as sp
import scipy.interpolate as sp_interp
from configuration_manager import *

class simulation_domain():
    
    def __init__(self, config, grid, fields, secondary_fields):
        
        self.grid_set   = False
        self.fields_set = False
        self.set_grid( grid )
        self.set_grid_fields( fields )
        self.set_secondary_grid_fields( secondary_fields )

        self.has_subdomain = False
        if config.has_key('domain_options'):
            self.set_subdomain( config )
            
        return
    
    def set_grid(self, grid):
        
        if self.grid_set == True:
            print( 'domain: Warning: grid is already set, proceeding anyway' )
            self.grid_set = False
        
        self.num_dim = len( grid )
        
        ###
        ### Dimension-specific functions
        ###
        if self.num_dim == 2:
            self.interpolate_fields = self.interpolate_fields_2D
            self.interpolate_secondary_fields = self.interpolate_secondary_fields_2D
        elif self.num_dim == 3:
            self.interpolate_fields = self.interpolate_fields_3D
            self.interpolate_secondary_fields = self.interpolate_secondary_fields_3D
        
        ###
        ### Grid
        ###
        x = grid[ 'X' ]
        y = grid[ 'Y' ]
        if self.num_dim == 3:
            z = grid[ 'Z' ]

        ###
        ### Grid extents
        ###
        self.x_min = np.min( x )
        self.x_max = np.max( x )
        self.y_min = np.min( y )
        self.y_max = np.max( y )
        if self.num_dim == 3:
            self.z_min = np.min( z )
            self.z_max = np.max( z )
            
        if self.num_dim == 2:
            self.N_x = x.shape[1]
            self.N_y = y.shape[0]
        if self.num_dim == 3:
            self.N_x = x.shape[2]
            self.N_y = y.shape[1]
            self.N_z = z.shape[0]
                
        num_points = self.N_x * self.N_y
        if self.num_dim == 3:
            num_points *= self.N_z
                    
        if self.num_dim == 2:
            self.dx = np.gradient( x, axis = 1 )[0,0]
            self.dy = np.gradient( y, axis = 0 )[0,0]
        if self.num_dim == 3:
            self.dx = np.gradient( x, axis = 2 )[0,0,0]
            self.dy = np.gradient( y, axis = 1 )[0,0,0]
            self.dz = np.gradient( z, axis = 0 )[0,0,0]
        
        self.grid_set = True

        print('domain: Grid ready:')
        print( '\t num_dim = %i' % self.num_dim )
        if self.num_dim == 2:
            print( '\t N_x = %i ... N_y = %i' % ( self.N_x, self.N_y ) )
            print( '\t dx = %.4E ... dy = %.4E' % ( self.dx, self.dy ) )
        elif self.num_dim == 3:
            print( '\t N_x = %i ... N_y = %i ... N_z = %i' % ( self.N_x, self.N_y, self.N_z ) )
            print( '\t dx = %.4E ... dy = %.4E ... dz = %.4E' % ( self.dx, self.dy, self.dy ) )
        
        return
    
    def set_grid_fields(self, grid_fields):
        
        iret = 0
        if self.fields_set == True:
            print( 'domain: Warning: fields are already set, proceeding anyway' )
        if self.grid_set == False:
            print( 'domain: Error: grid is not set yet' )
            iret = -1
            
        self.U = grid_fields[ 'velocity-1' ]
        self.V = grid_fields[ 'velocity-2' ]
        if self.num_dim == 3:
            self.W = grid_fields[ 'velocity-3' ]
            
        self.fields_set = True
        return iret 

    def set_secondary_grid_fields(self, secondary_fields):

        iret = 0
        # Secondary fields are kept in dictionary form for flexibility
        self.secondary_grid_fields = secondary_fields
        return iret
    
    def set_subdomain(self, config):

        self.has_subdomain = False
        self.subdomain_x = config['domain_options']['subdomain_x']
        self.subdomain_y = config['domain_options']['subdomain_y']
        if self.num_dim == 3:
            self.subdomain_z = config['domain_options']['subdomain_z']
        self.has_subdomain = True

        print('domain: Subdomain ready')
        print('\t x_min = %.4E ... x_max = %.4E' % ( self.subdomain_x[0], self.subdomain_x[1] ) )
        print('\t y_min = %.4E ... y_max = %.4E' % ( self.subdomain_y[0], self.subdomain_y[1] ) )
        if self.num_dim == 3:
            print('\t z_min = %.4E ... z_max = %.4E' % ( self.subdomain_z[0], self.subdomain_z[1] ) )
            
        return

    def get_secondary_fields_names(self):

        return self.secondary_grid_fields.keys()
    
    def compute_interpolation_weights(self, points, indices, weights):

        ###
        ### Note: points = [ x_1, y_1, z_1, x_2, y_2, z_2, ..., x_N, y_N, z_N ]
        ###     
        
        ###
        ### Prepare indices and weights
        ###
        iret = 0
        
        # Extract coordinates        
        N   = len( points )
        x_p = points[0:N:self.num_dim]
        y_p = points[1:N:self.num_dim]
        if self.num_dim == 3:
            z_p = points[2:N:self.num_dim]

        # Compute interpolation indices
        indices[:,0] = np.floor( ( x_p - self.x_min ) / self.dx )
        indices[:,1] = np.floor( ( y_p - self.y_min ) / self.dy )
        if self.num_dim == 3:
            indices[:,2] = np.floor( ( z_p - self.z_min ) / self.dz )

        # Compute interpolation weights
        weights[:,0] = 1.0 + ( indices[:,0] - ( x_p - self.x_min ) / self.dx )
        weights[:,1] = 1.0 + ( indices[:,1] - ( y_p - self.y_min ) / self.dy )
        if self.num_dim == 3:
            weights[:,2] = 1.0 + ( indices[:,2] - ( z_p - self.z_min ) / self.dz )

        return iret

    def interpolate_fields_2D(self, indices, weights):
        
        interp_U = self.interpolate_single_field_2D( indices, weights, self.U )
        interp_V = self.interpolate_single_field_2D( indices, weights, self.V )
        output = [interp_U,interp_V]        
        return output

    def interpolate_fields_3D(self, indices, weights):

        interp_U = self.interpolate_single_field_3D( indices, weights, self.U )
        interp_V = self.interpolate_single_field_3D( indices, weights, self.V )
        interp_W = self.interpolate_single_field_3D( indices, weights, self.W )
        output = [interp_U,interp_V,interp_W]
        return output

    def interpolate_secondary_fields_2D(self, indices, weights):
        
        # As opposed to interpolate_fields_2D, this returns a dictionary
        # This is ok because no computation is done on these fields
        num_fields    = len( self.secondary_grid_fields )
        interp_fields = {}
        if num_fields > 0:
            for field_name in self.secondary_grid_fields:
                field  = self.secondary_grid_fields[ field_name ]
                interp = self.interpolate_single_field_2D( indices, weights, field )
                interp_fields[ field_name ] = interp
        return interp_fields
    
    def interpolate_secondary_fields_3D(self, indices, weights):

        # As opposed to interpolate_fields_3D, this returns a dictionary
        # This is ok because no computation is done on these fields
        interp_fields = {}
        for field_name in self.secondary_grid_fields:
            field  = self.secondary_grid_fields[ field_name ]
            interp = self.interpolate_single_field_3D( indices, weights, field )
            interp_fields[ field_name ] = interp
        return interp_fields
        
    def interpolate_single_field_2D(self, indices, weights, field):

        # Unpack indices and weights
        i,j     = indices
        w_x,w_y = weights

        # Indices are stored as floats, convert to int
        i = int( i )
        j = int( j )
        
        # Interpolate in x
        f_1 = w_x * field[ j,  i ] + ( 1.0 - w_x ) * field[ j,  i+1 ]
        f_2 = w_x * field[ j+1,i ] + ( 1.0 - w_x ) * field[ j+1,i+1 ]

        # Interpolate in y
        f = w_y * f_1 + ( 1.0 - w_y ) * f_2

        return f
    
    def interpolate_single_field_3D(self, indices, weights, field):

        # Unpack indices and weights
        i,j,k       = indices
        w_x,w_y,w_z = weights

        # Indices are stored as floats, convert to int
        i = int( i )
        j = int( j )
        k = int( k )
        
        # Interpolate in x
        f_1 = w_x * field[ k,  j,  i ] + ( 1.0 - w_x ) * field[ k,  j,  i+1 ]
        f_2 = w_x * field[ k,  j+1,i ] + ( 1.0 - w_x ) * field[ k,  j+1,i+1 ]
        f_3 = w_x * field[ k+1,j,  i ] + ( 1.0 - w_x ) * field[ k+1,j,  i+1 ]
        f_4 = w_x * field[ k+1,j+1,i ] + ( 1.0 - w_x ) * field[ k+1,j+1,i+1 ]
        # Interpolate in y
        f_a = w_y * f_1 + ( 1.0 - w_y ) * f_2
        f_b = w_y * f_3 + ( 1.0 - w_y ) * f_4
        # Interpolate in z
        f = w_z * f_a + ( 1.0 - w_z ) * f_b
        
        return f                
        
        
