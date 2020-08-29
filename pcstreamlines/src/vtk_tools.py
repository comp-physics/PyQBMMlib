import xml.dom.minidom

class vtk_writer:

    def __init__(self):

        self.file_names = []
        return

    def positions_to_string(self, X, Y, Z):

        num_particles = len( X )
        string  = ''
        for i_particle in range( 0, num_particles ):
            string += repr( X[ i_particle ] ) + ' '
            string += repr( Y[ i_particle ] ) + ' '
            string += repr( Z[ i_particle ] ) + ' ' 

        return string

    def array_to_string(self, array):

        num_particles = len( array )
        string = ''
        for i_particle in range( 0, num_particles ):
            string += repr( array[ i_particle ] )
        return string
    
    def snapshot(self, file_name, X, Y, Z, secondary_fields = [], residence_time = []):

        num_particles = len( X )

        ###
        ### Document and root element
        document = xml.dom.minidom.Document()
        root_element = document.createElementNS( "VTK", "VTKFile" )
        root_element.setAttribute( "type", "UnstructuredGrid" )
        root_element.setAttribute( "version", "0.1" )
        root_element.setAttribute( "byte_order", "LittleEndian" )
        document.appendChild( root_element )

        # Unstructured grid element
        unstructured_grid = document.createElementNS( "VTK", "UnstructuredGrid" )
        root_element.appendChild( unstructured_grid )

        # Piece 0
        piece = document.createElementNS( "VTK", "Piece" )
        piece.setAttribute( "NumberOfPoints", str( num_particles ) )
        piece.setAttribute( "NumberOfCells", "0" )
        unstructured_grid.appendChild( piece )

        ###
        ### Points
        points = document.createElementNS( "VTK", "Points" )
        piece.appendChild( points )

        # Point location data
        point_coords = document.createElementNS( "VTK", "DataArray" )
        point_coords.setAttribute( "type", "Float32" )
        point_coords.setAttribute( "format", "ascii" )
        point_coords.setAttribute( "NumberOfComponents", "3" ) # [ecg]  generalize to num_dim
        points.appendChild( point_coords )

        string = self.positions_to_string( X, Y, Z )
        point_coords_data = document.createTextNode( string )
        point_coords.appendChild( point_coords_data )

        ###
        ### Cells
        cells = document.createElementNS( "VTK", "Cells" )
        piece.appendChild( cells )

        # Cell locations
        cell_connectivity = document.createElementNS( "VTK", "DataArray" )
        cell_connectivity.setAttribute("type", "Int32")
        cell_connectivity.setAttribute("Name", "connectivity")
        cell_connectivity.setAttribute("format", "ascii")        
        cells.appendChild(cell_connectivity)

        # Cell location data
        connectivity = document.createTextNode( "0" )
        cell_connectivity.appendChild( connectivity )

        cell_offsets = document.createElementNS( "VTK", "DataArray" )
        cell_offsets.setAttribute( "type", "Int32" )
        cell_offsets.setAttribute( "Name", "offsets" )
        cell_offsets.setAttribute( "format", "ascii" )                
        cells.appendChild( cell_offsets )    
        offsets = document.createTextNode( "0" )
        cell_offsets.appendChild( offsets )

        cell_types = document.createElementNS( "VTK", "DataArray" )
        cell_types.setAttribute( "type", "UInt8" )
        cell_types.setAttribute( "Name", "types" )
        cell_types.setAttribute( "format", "ascii" )                
        cells.appendChild( cell_types )
        types = document.createTextNode( "1" )  
        cell_types.appendChild( types )

        ###
        ### Data at points
        point_data = document.createElementNS( "VTK", "PointData" )
        piece.appendChild( point_data )

        # Points
        point_coords_2 = document.createElementNS( "VTK", "DataArray" )
        point_coords_2.setAttribute( "Name", "Points" )
        point_coords_2.setAttribute( "NumberOfComponents", "3" )
        point_coords_2.setAttribute( "type", "Float32" )
        point_coords_2.setAttribute( "format", "ascii" )
        point_data.appendChild( point_coords_2 )        

        point_coords_string = self.positions_to_string( X, Y, Z )
        point_coords_2_data = document.createTextNode( point_coords_string )
        point_coords_2.appendChild( point_coords_2_data )

        # Residence time:
        if len( residence_time ) > 0:
            res_time_node = document.createElementNS( "VTK", "DataArray" )
            res_time_node.setAttribute( "Name", "log_residence_time" )
            res_time_node.setAttribute( "type", "Float32" )
            res_time_node.setAttribute( "format", "ascii" )
            point_data.appendChild( res_time_node )

            string = self.array_to_string( residence_time )
            res_time_data = document.createTextNode( string )
            res_time_node.appendChild( res_time_data )
            

        ###
        ### Cell data
        cell_data = document.createElementNS( "VTK", "CellData" )
        piece.appendChild( cell_data )

        output_file = open( file_name, 'w' )
        document.writexml( output_file, newl = '\n' )
        output_file.close()
        self.file_names.append( file_name )

    def write_pvd(self, file_name):

        output_file = open( file_name, 'w' )

        pvd = xml.dom.minidom.Document()
        pvd_root = pvd.createElementNS( "VTK", "VTKFile" )
        pvd_root.setAttribute( "type", "Collection" )
        pvd_root.setAttribute( "version", "0.1" )
        pvd_root.setAttribute( "byte_order", "LittleEndian" )
        pvd.appendChild( pvd_root )

        collection = pvd.createElementNS( "VTK", "Collection" )
        pvd_root.appendChild( collection )

        num_snapshots = len( self.file_names )
        for i_snapshot in range( num_snapshots ):
            dataSet = pvd.createElementNS( "VTK", "DataSet" )
            dataSet.setAttribute( "timestep", str( i_snapshot ) )
            dataSet.setAttribute( "group", "" )
            dataSet.setAttribute( "part", "0" )
            dataSet.setAttribute( "file", str( self.file_names[ i_snapshot ] ) )
            collection.appendChild( dataSet )
