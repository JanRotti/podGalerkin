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
            self.points_cylinder[i][1] = np.arctan2(y,self.points[i][0]) if np.arctan2(y,self.points[i][0]) >= 0 else np.arctan2(y,self.points[i][0])+2*np.pi
            
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

    def compute_cell_values_from_node_data2(self,data):
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

    def compute_cell_derivatives(self):
        for cell in self.cells:
            cell.dx = cell.int_coeffs[0] * (cell.center[1])+ cell.int_coeffs[1]
            cell.dy = cell.int_coeffs[0] * (cell.center[0])+ cell.int_coeffs[2]

    def compute_node_volume_participation(self):
        for cell in self.cells:

            # retrieve cell nodes
            node_list = cell.nodes 
            
            # attribute cell volume to nodes
            for node_index in node_list:
                self.nodes[node_index].dv += cell.volume/len(node_list)

        print_string="Nodal volume participation computed!"
        print(f'{print_string:{"-"}^80}')

    def compute_finite_spatial_derivatives(self,data,second=False):
        ## finite difference method
        
        # initialize spatial derivatives
        derivs_x = np.empty(len(self.nodes))
        derivs_y = np.empty(len(self.nodes))
        if second:
            sec_derivs_x = np.empty(len(self.nodes))
            sec_derivs_y = np.empty(len(self.nodes))
        
        # node based derivative calculation - node iteration
        for node in self.nodes:
            
            # get node based indexes - right,left,up,bottom based on 1 proximity stencil
            i = node.index 
            r = node.get_r()
            l = node.get_l()
            u = node.get_u() if node.get_u() else None  # special boundary case
            b = node.get_b() if node.get_b() else None  # special boundary case
            
            # compute cylinder derivative dtheta with special case jump point
            theta_l = self.nodes[l].theta if (self.nodes[l].theta!=0) else 2*np.pi
            theta_r = self.nodes[r].theta if (self.nodes[i].theta!=0) else self.nodes[r].theta-2*np.pi
            dtheta = (data[l]-data[r])/(theta_l-theta_r) # central difference scheme
            
            # compute cylinder derivative dr with boundary case treatment
            rad_u = self.nodes[u].rad if (u) else None
            rad_b = self.nodes[b].rad if (b) else None
            if(u==None):
                dr = 0      # farfield condition
                #dr = (data[i]-data[b])/(node.rad-rad_b) # backwards difference scheme
            elif(b==None):
                dr = 0      # no slip wall condition
            else:
                dr = (data[u]-data[b])/(rad_u-rad_b) # central difference scheme
            
            # compute transform derivatives
            drdx = node.x/node.rad
            dthetadx = -node.y/(node.rad*node.rad)
            drdy = node.y/node.rad
            dthetady = node.x/(node.rad*node.rad)
            
            # compute cartesian derivatives
            derivs_x[i] = np.multiply(dtheta,dthetadx) + np.multiply(dr,drdx)
            derivs_y[i] = np.multiply(dtheta,dthetady) + np.multiply(dr,drdy)
            
            # add derivatives to node
            node.dx = derivs_x[i]
            node.dy = derivs_y[i]

            # compute second derivatives
            if second:

                # compute second cylinder derivative ddtheta - special cases already treated
                ddtheta = (data[l]-2*data[i]+data[r])/((theta_l-node.theta)*(node.theta-theta_r)) # central difference scheme
                
                # compute second cylinder derivative ddr - special cases already treated
                if(u==None):
                    ddr = 0     # farfield condition
                elif(b==None):
                    ddr = 0     # no slip wall condition
                else:
                    # outer derivative by FDS, inner derivatives using BDS
                    ddr = (data[u]*(node.rad-rad_b)+data[b]*(rad_u-node.rad)-data[i]*(rad_u-rad_b))/((rad_u-node.rad)*(rad_u-node.rad)*(node.rad-rad_b))
                
                # compute second transform derivatives
                ddthetadx = (2*node.y*node.x)/(np.power(node.rad,4))
                ddrdx = (node.y*node.y)/(np.power(node.rad,3))
                ddthetady = -(2*node.y*node.x)/(np.power(node.rad,4))
                ddrdy = (node.x*node.x)/(np.power(node.rad,3))
                
                # compute second cartesian derivatives
                sec_derivs_x[i] = np.multiply(ddtheta,ddthetadx) + np.multiply(ddr,ddrdx) + 2 * np.multiply(np.multiply(dtheta,dthetadx),np.multiply(dr,drdx))
                sec_derivs_y[i] = np.multiply(ddtheta,ddthetady) + np.multiply(ddr,ddrdy) + 2 * np.multiply(np.multiply(dtheta,dthetady),np.multiply(dr,drdy))

                # add derivatives to node
                node.ddx = sec_derivs_x[i]
                node.ddy = sec_derivs_y[i]  
        
        # return constructs with first or second order spatial derivatives
        if second:
            return [derivs_x,derivs_y,sec_derivs_x,sec_derivs_y]
        else:
            return [derivs_x,derivs_y]

    def compute_interpolated_derivatives(self,data):
        ## interpolation derivatives
        ## only first order derivatives possible
        
        # initialize spatial derivatives
        derivs_x = np.empty(self.N)
        derivs_y = np.empty(self.N)

        # cell based derivative calculation - cell iteration 
        for cell in self.cells:
            
            # compute interpolation coefficients for bilinear interpolation
            cell.cell_interpolation(data,self.mesh)

            # compute first spatial derivatives
            cell.get_interpolated_first_derivatives()
            dx = cell.dx
            dy = cell.dy
        
            # save derivatives
            derivs_x[cell.index] = dx
            derivs_y[cell.index] = dy

        # return derivative arrays
        return [derivs_x,derivs_y]




