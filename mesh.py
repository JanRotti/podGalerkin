import numpy as np
import math
from cell import cell
from node import node


class exp_mesh:
    
    global mesh

    def __init__(self,mesh,mid_point):
        
        print_string="Custom mesh initialization started"
        print(f'{print_string:{"-"}^80}')
        
        self.mesh = mesh                # handling meshio instance of mesh
        self.mid_point = mid_point      # set mid point for shift

        # set length scales
        self.n = len(self.mesh.points)          # length of points array
        self.N = len(self.mesh.cells[0][1])     # length of cells array
        
        # providing coordinate changes
        self.points = np.empty_like(self.mesh.points)
        self.points_cylinder = np.empty_like(self.mesh.points)
        self.compute_coord_transform()
        
        ## cylinder grid values
        self.thetas = np.unique(self.points_cylinder[:,1])
        self.rads = np.unique(self.points_cylinder[:,0])

        ## define custom cells instances in self.cells
        self.cells = [cell(node_list,i) for i,node_list in enumerate(mesh.cells[0][1])]  
        
        # compute cell volumes
        self.compute_cell_volumes()
        
        # compute cell centers
        self.compute_cell_centers()
        
        # used for derivative computation 
        self.dthe = np.mean(np.diff(np.unique(self.points_cylinder[:,1])))
        
        ## define custom node instances in self.nodes
        self.nodes = [node(x,y,i) for i,(x,y) in enumerate(self.points)]
        
        # set cylinder coordinates
        for i,nod in enumerate(self.nodes):
            nod.set_cyl(self.points_cylinder[i,0],self.points_cylinder[i,1])
        
        # set node neighbors
        self.compute_node_neighbors()

        # compute nodal volumes
        self.compute_node_volume_participation()

        print_string = "Custom mesh initialized successfully!"
        print(f'{print_string:{"-"}^80}')

    def compute_coord_transform(self):
        
        # iterate mesh points
        for i,(x,y) in enumerate(self.mesh.points):
            
            # set shifted cartesian coordinates
            self.points[i][0] = x - self.mid_point[0]
            self.points[i][1] = y - self.mid_point[1]

            # set cylinder coordinates
            self.points_cylinder[i][0] = np.sqrt((self.points[i][0])*(self.points[i][0])+(self.points[i][1])*(self.points[i][1]))
            self.points_cylinder[i][1] = np.arctan2(self.points[i][1],self.points[i][0]) if np.arctan2(self.points[i][1],self.points[i][0]) >= 0 else np.arctan2(self.points[i][1],self.points[i][0])+2*np.pi
            
            # setting theta with 2*pi to 0
            if math.isclose(2*np.pi,self.points_cylinder[i][1]):
                self.points_cylinder[i][1] = 0 

        # rounding cylinder coordinates due to numerical errors        
        self.points_cylinder = np.around(self.points_cylinder,12)
        print_string="Shifted and Cylinder coordinates calculated and stored!"
        print(f'{print_string:{"-"}^80}')
        
    def compute_cell_volumes(self):
        ## computing cell volumes with cell subroutine
        
        for cell in self.cells:
            cell.compute_volume(self)

        print_string="Cell volumes computed in cell structures!"
        print(f'{print_string:{"-"}^80}')

    def compute_cell_centers(self):
        for cell in self.cells:
            cell.compute_center(self)

        print_string="Cell centers computed in cell structures!"
        print(f'{print_string:{"-"}^80}')

    def compute_node_neighbors(self):
        
        # iterate over radius levels
        for i,rad in enumerate(self.rads):
            
            # find indexes on same radius level
            same_rad = self.points_cylinder[:,0]==rad
            same_rad_index_list = np.where(same_rad==True)[0]
           
            # find indexes of next radius level
            if rad != self.rads[-1]:
                next_rad = self.points_cylinder[:,0]==self.rads[i+1]
                next_rad_index_list = np.where(next_rad==True)[0]            
            
            # iterate over thetas with stencil left - middle - up
            for j,theta in enumerate(self.thetas):
                
                # find indexes of same theta level
                same_theta = self.points_cylinder[:,1]==theta
                same_theta_index_list = np.where(same_theta==True)[0]
                index = int(np.intersect1d(same_theta_index_list,same_rad_index_list))
                
                # find indexes of next theta level - left node
                next_theta = self.points_cylinder[:,1]== (self.thetas[j+1] if (theta!=self.thetas[-1]) else self.thetas[0])
                next_theta_index_list = np.where(next_theta==True)[0]
                left = int(np.intersect1d(next_theta_index_list,same_rad_index_list))
                
                # iterate over radius levels
                if rad != self.rads[-1]:

                    # find upper node
                    up = int(np.intersect1d(same_theta_index_list,next_rad_index_list))
                    
                    # set up and bottom of nodes
                    self.nodes[index].set_u(up)
                    self.nodes[up].set_b(index)
                
                # set left and right neighbor nodes
                self.nodes[index].set_l(left)
                self.nodes[left].set_r(index) 
        
        print_string="Node neighbors assigned!"
        print(f'{print_string:{"-"}^80}')
                
    def compute_cell_values_from_node_data(self,data):
        ## by node value averaging
        # initialize cell data array
        cell_data = np.empty(self.N)

        # iterate over cells
        for cell in self.cells:

            # use cell function to compute cell value
            cell_data[cell.index]=cell.compute_cell_value(data[cell.nodes])
        
        # return cell based data array
        return cell_data

    def compute_cell_values_by_interpolation(self,data):
        ## by interpolation
        # initialize cell data array
        cell_data = np.empty(self.N)

        # iterate over cells
        for cell in self.cells:

            # compute cell interpolation
            cell.cell_interpolation(data,self)

            # use cell function to compute cell value
            cell_data[cell.index]=cell.get_interpolated_center_value()
        
        # return cell based data array
        return cell_data
        
    def compute_cell_interpolations(self,data):
        for cell in self.cells:
            cell.cell_interpolation(data,self)

    def compute_node_volume_participation(self):
        for cell in self.cells:

            # retrieve cell nodes
            node_list = cell.nodes 
            
            # attribute cell volume to nodes
            for node_index in node_list:
                self.nodes[node_index].dv += cell.volume/len(node_list)

        print_string="Nodal volume participation computed!"
        print(f'{print_string:{"-"}^80}')


