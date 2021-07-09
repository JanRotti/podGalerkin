import numpy as np
from geometry.cell import cell
from geometry.node import node

class rot_sym_mesh:

    ### initializing class attributes
    mesh = None
    mid_point = [0.5,0] # cylinder center in [x,y]

    # length scales
    n = None            # number of nodes
    N = None            # number of cells

    # coordinates
    points = None       # cartesian node coordinates 
    points_pol = None   # polar node coordinates

    # polar potentials
    rad_levels = None   # list of potential radii
    phi_levels = None   # list of potential angles

    # subclass attributes
    nodes = None
    cells = None

    # problem attributes
    volume_weights = None      # skalar product weights in state based space
    gamma = 1.4         # isentropic coefficient
    alpha = 1           # parameter for inner product norm 1 / gamma -> stagnation energy norm; 1 -> stagnation enthalpy norm;


    def __init__(self,mesh):

        self.status("Initializing Mesh")

        self.mesh = mesh    # handing over meshio object

        # set geometric length scales
        self.n = len(self.mesh.points)
        self.N = len(self.mesh.cells[0][1])

        # computing coordinates
        self.points = self.mesh.points - self.mid_point
        self.points_pol = np.empty_like(self.points)
        self.compute_polar_coordinates()

        # get polar potentials - necessary for finite differences
        self.rad_levels = np.unique(self.points_pol[:,0])
        self.phi_levels = np.unique(self.points_pol[:,1])

        # initializing minimal node and cell instances
        self.cells = [cell(nodes,i) for i,nodes in enumerate(mesh.cells[0][1])]
        self.nodes = [node(x,y,i) for i,(x,y) in enumerate(self.points)]

        # computing cell and node attributes
        self.compute_cell_volumes()
        self.compute_cell_centers()
        for nod in self.nodes:
            nod.rad = self.points_pol[nod.i][0]
            nod.phi = self.points_pol[nod.i][1]
        self.compute_node_volume()
        self.compute_volume_weights()
        self.compute_node_neighbors()
        
        self.status("Mesh Initialization Successful!")

    # local status updates for mesh class
    def status(self,print_string):
        print(f'{print_string:{"-"}<80}')

    def compute_polar_coordinates(self):
        # iterate nodes
        for i in range(self.n):
            
            # set cylinder coordinates
            self.points_pol[i][0] = np.sqrt((self.points[i][0])*(self.points[i][0])+(self.points[i][1])*(self.points[i][1]))
            self.points_pol[i][1] = np.arctan2(self.points[i][1],self.points[i][0]) if np.arctan2(self.points[i][1],self.points[i][0]) >= 0 else np.arctan2(self.points[i][1],self.points[i][0])+2*np.pi
            
            # setting phi with 2*pi to 0
            if np.isclose(2*np.pi,self.points_pol[i][1],rtol=1e-09,atol=0.0):
                self.points_pol[i][1] = 0 

        # rounding cylinder coordinates due to numerical errors        
        self.points_pol = np.around(self.points_pol,12)

    def compute_cell_volumes(self):
        ## computing cell volumes with cell subroutine
        for cel in self.cells:
            cel.compute_volume(self)

    def compute_cell_centers(self):
        ## computing cell centers with cell subroutine
        for cel in self.cells:
            cel.compute_center(self)

    def compute_node_neighbors(self):
        # iterate over radius levels
        for i, rad in enumerate(self.rad_levels):
            
            # find indexes on same radius level
            same_rad = self.points_pol[:,0]==rad
            same_rad_index_list = np.where(same_rad==True)[0]
           
            # find indexes of next radius level
            if rad != self.rad_levels[-1]:
                next_rad = self.points_pol[:,0]==self.rad_levels[i+1]
                next_rad_index_list = np.where(next_rad==True)[0]            
            
            # iterate over phis with stencil left - middle - up
            for j,phi in enumerate(self.phi_levels):
                
                # find indexes of same phi level
                same_phi = self.points_pol[:,1]==phi
                same_phi_index_list = np.where(same_phi==True)[0]
                index = int(np.intersect1d(same_phi_index_list,same_rad_index_list))
                
                # find indexes of next phi level - left node
                next_phi = self.points_pol[:,1]== (self.phi_levels[j+1] if (phi!=self.phi_levels[-1]) else self.phi_levels[0])
                next_phi_index_list = np.where(next_phi==True)[0]
                left = int(np.intersect1d(next_phi_index_list,same_rad_index_list))
                
                # iterate over radius levels
                if rad != self.rad_levels[-1]:

                    # find upper node
                    up = int(np.intersect1d(same_phi_index_list,next_rad_index_list))
                    
                    # set up and bottom of nodes
                    self.nodes[index].u = up
                    self.nodes[up].b = index
                
                # set left and right neighbor nodes
                self.nodes[index].l = left
                self.nodes[left].r = index

    def compute_node_volume(self):
        for cel in self.cells:
            node_list = cel.nodes 
            for node_index in node_list:
                self.nodes[node_index].volume += cel.volume/len(node_list)

    def finite_differences(self,data,fd=False,compute_laplacian=False):
        
        # initialize data vectors
        dx = np.empty(self.n)
        dy = np.empty(self.n)
        if compute_laplacian:
            laplacian = np.empty(self.n)
        
        
        for nod in self.nodes:
            
            # 5 point stencil indizes
            i = nod.i
            r = nod.r
            l = nod.l
            u = nod.u # special boundary case
            b = nod.b # special boundary case

            # transform derivatives
            drdx = nod.x/nod.rad
            dphidx = -nod.y/(nod.rad**2)
            drdy = nod.y/nod.rad
            dphidy = nod.x/(nod.rad**2)     

            # temporary radii and phi values
            rad_u = self.nodes[u].rad if (u) else None
            rad_b = self.nodes[b].rad if (b) else None
            phi_l = self.nodes[l].phi if (self.nodes[l].phi!=0) else 2*np.pi
            phi_r = self.nodes[r].phi if (self.nodes[i].phi!=0) else self.nodes[r].phi-2*np.pi
            
            if not (u) or not (b): # boundary conditions
                dr = 0
                dphi = 0
            else:
                if fd: # using forward differences
                    dr   = (data[u] - data[i]) / (rad_u - nod.rad)
                    dphi = (data[l] - data[i]) / (phi_l - nod.phi)
                else: # using central differences
                    dr   = (data[u] - data[b]) / (rad_u - rad_b)
                    dphi = (data[l] - data[r]) / (phi_l - phi_r)

            # cartesian differences
            dx[i] = dphi * dphidx + dr * drdx
            dy[i] = dphi * dphidy + dr * drdy

            if compute_laplacian:
                if not (u) or not (b):
                    ddr = 0
                    ddphi = 0
                else:
                    ddphi = (data[l] - 2*data[i] + data[r]) / ((phi_l - nod.phi) * (nod.phi - phi_r)) 
                    ddr   = (data[u] * (nod.rad - rad_b) + data[b] * (rad_u - nod.rad) - data[i] * (rad_u - rad_b)) / ((rad_u - nod.rad) * (nod.rad - rad_b) * (rad_u - rad_b) / 2)
                
                laplacian[i] = ddr + (1 / nod.rad) * dr + (1 / (nod.rad**2)) * ddphi

        if compute_laplacian:
            return [dx,dy,laplacian]
        else:
            return [dx,dy]

    def compute_volume_weights(self):
        weights = np.empty(self.n)
        for nod in self.nodes:
            weights[nod.i] = nod.volume
        self.volume_weights = np.concatenate([weights,weights,weights])
