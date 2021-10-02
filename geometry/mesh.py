import numpy as np
from geometry.cell import cell
from geometry.node import node

class rotationalSymmetricMesh:

    ### initializing class attributes
    mesh = None
    midPoint = [0.5,0]

    # length scales
    n = None            # number of nodes
    N = None            # number of cells

    # coordinates
    points = None       # cartesian node coordinates 
    pointsPol = None   # polar node coordinates

    # polar potentials
    radLevels = None   # list of potential radii
    phiLevels = None   # list of potential angles

    # subclass attributes
    nodes = None
    cells = None

    # problem attributes
    volumeWeights = None      # skalar product weights in state based space
    gamma = 1.4         # isentropic coefficient
    alpha = 1           # parameter for inner product norm 1 / gamma -> stagnation energy norm; 1 -> stagnation enthalpy norm;


    def __init__(self,mesh):

        self.status("Initializing Mesh")

        self.mesh = mesh    # handing over meshio object

        # number of nodes
        self.n = len(self.mesh.points)
        # number of cells
        self.N = len(self.mesh.cells[0][1]) 

        # node coordinates
        self.points = self.mesh.points - self.midPoint
        self.pointsPol = np.empty_like(self.points)
        self.compute_polar_coordinates()

        # polar potentials
        self.radLevels = np.unique(self.pointsPol[:,0])
        self.phiLevels = np.unique(self.pointsPol[:,1])

        # initializing node and cell instances
        self.cells = [cell(nodes,i) for i,nodes in enumerate(mesh.cells[0][1])]
        self.nodes = [node(x,y,i) for i,(x,y) in enumerate(self.points)]

        ### computing cell and node based attributes
        self.compute_cell_volumes()
        self.compute_cell_centers()
        
        # assert node coordinates
        for nod in self.nodes:
            nod.rad = self.pointsPol[nod.i][0]
            nod.phi = self.pointsPol[nod.i][1]
        
        # compute node volumes
        self.compute_node_volume()
        #compute node based weights
        self.compute_volume_weights()
        # assert node neighbors
        self.compute_node_neighbors()
        
        # status for initialization
        self.status("Mesh Initialization Successful!")

    
    def status(self,print_string):
        # pretty printing
        print(f'{print_string:{"-"}<80}')

    def compute_polar_coordinates(self):
        # iterate nodes
        for i in range(self.n):
            
            # set cylinder coordinates
            self.pointsPol[i][0] = np.sqrt((self.points[i][0])*(self.points[i][0])+(self.points[i][1])*(self.points[i][1]))
            self.pointsPol[i][1] = np.arctan2(self.points[i][1],self.points[i][0]) if np.arctan2(self.points[i][1],self.points[i][0]) >= 0 else np.arctan2(self.points[i][1],self.points[i][0])+2*np.pi
            
            # setting phi with 2*pi to 0
            if np.isclose(2*np.pi,self.pointsPol[i][1],rtol=1e-09,atol=0.0):
                self.pointsPol[i][1] = 0 

        # rounding cylinder coordinates due to numerical errors        
        self.pointsPol = np.around(self.pointsPol,12)

    def compute_cell_volumes(self):
        # computing cell volumes with cell subroutine
        for cel in self.cells:
            cel.compute_volume(self)

    def compute_cell_centers(self):
        # computing cell centers with cell subroutine
        for cel in self.cells:
            cel.compute_center(self)

    def compute_node_neighbors(self):
        # iterate over radius levels
        for i, rad in enumerate(self.radLevels):
            
            # find indexes on same radius level
            sameRadius = self.pointsPol[:,0]==rad
            sameRadiusIndexList = np.where(sameRadius==True)[0]
           
            # find indexes of next radius level
            if rad != self.radLevels[-1]:
                nextRadius = self.pointsPol[:,0]==self.radLevels[i+1]
                nextRadiusIndexList = np.where(nextRadius==True)[0]            
            
            # iterate over phis with stencil left - middle - up
            for j,phi in enumerate(self.phiLevels):
                
                # find indexes of same phi level
                samePhi = self.pointsPol[:,1]==phi
                samePhiIndexList = np.where(samePhi==True)[0]
                index = int(np.intersect1d(samePhiIndexList,sameRadiusIndexList))
                
                # find indexes of next phi level - left node
                nextPhi = self.pointsPol[:,1]== (self.phiLevels[j+1] if (phi!=self.phiLevels[-1]) else self.phiLevels[0])
                nextPhiIndexList = np.where(nextPhi==True)[0]
                left = int(np.intersect1d(nextPhiIndexList,sameRadiusIndexList))
                
                # iterate over radius levels
                if rad != self.radLevels[-1]:

                    # find upper node
                    up = int(np.intersect1d(samePhiIndexList,nextRadiusIndexList))
                    
                    # set up and bottom of nodes
                    self.nodes[index].u = up
                    self.nodes[up].b = index
                
                # set left and right neighbor nodes
                self.nodes[index].l = left
                self.nodes[left].r = index

    def compute_node_volume(self):
        # iterate cells
        for cel in self.cells:
            nodeList = cel.nodes 
            # iterate nodes
            for nodeIndex in nodeList:
                # assert weighted cell volume to cell constituting nodes
                self.nodes[nodeIndex].volume += cel.volume/len(nodeList)

    def finite_differences(self,data,fd=False,computeLaplacian=False):
        
        # initialize data vectors
        dx = np.zeros(self.n)
        dy = np.zeros(self.n)
        if computeLaplacian:
            laplacian = np.zeros(self.n)
        
        # iterate nodes
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
            radiusUp = self.nodes[u].rad if (u) else None
            radiusBottom = self.nodes[b].rad if (b) else None
            phiLeft = self.nodes[l].phi if (self.nodes[l].phi!=0) else 2*np.pi
            phiRight = self.nodes[r].phi if (self.nodes[i].phi!=0) else self.nodes[r].phi-2*np.pi
            
            if not (b): # boundary conditions
                # no slip/penetration wall
                dr = 0
                dphi = 0
            elif not (u):
                # neumann boundary for freestream
                dr = 0
                dphi = (data[l] - data[r]) / (phiLeft - phiRight)
            else:
                if fd: # using forward differences
                    dr   = (data[u] - data[i]) / (radiusUp - nod.rad)
                    dphi = (data[l] - data[i]) / (phiLeft - nod.phi)
                else: # using central differences
                    dr   = (data[u] - data[b]) / (radiusUp - radiusBottom)
                    dphi = (data[l] - data[r]) / (phiLeft - phiRight)

            # cartesian differences
            dx[i] = dphi * dphidx + dr * drdx
            dy[i] = dphi * dphidy + dr * drdy

            if computeLaplacian:
                if not (b): # boundary conditions
                    # no slip/penetration wall
                    ddr = 0
                    ddphi = 0
                elif not (u):
                    # neumann boundary for freestream
                    ddr = 0
                    ddphi = (data[l] - 2*data[i] + data[r]) / ((phiLeft - nod.phi) * (nod.phi - phiRight)) 
                else:
                    # central differences
                    ddphi = (data[l] - 2*data[i] + data[r]) / ((phiLeft - nod.phi) * (nod.phi - phiRight)) 
                    ddr   = (data[u] * (nod.rad - radiusBottom) + data[b] * (radiusUp - nod.rad) - data[i] * (radiusUp - radiusBottom)) / ((radiusUp - nod.rad) * (nod.rad - radiusBottom) * (radiusUp - radiusBottom) / 2)
                
                # laplacian computation
                laplacian[i] = ddr + (1 / nod.rad) * dr + (1 / (nod.rad**2)) * ddphi

        # structured return
        if computeLaplacian:
            return [dx,dy,laplacian]
        else:
            return [dx,dy]

            
    def compute_volume_weights(self):
        weights = np.empty(self.n)
        for nod in self.nodes:
            weights[nod.i] = nod.volume
        self.volumeWeights = np.concatenate([weights,weights,weights])

    def nabla_finite_differences(self,data):
        res = np.zeros(self.n)

        for nod in self.nodes:
            # 5 point stencil indizes
            i = nod.i
            r = nod.r
            l = nod.l
            u = nod.u # special boundary case
            b = nod.b # special boundary case

            # temporary radii and phi values
            radiusUp = self.nodes[u].rad if (u) else None
            radiusBottom = self.nodes[b].rad if (b) else None
            phiLeft = self.nodes[l].phi if (self.nodes[l].phi!=0) else 2*np.pi
            phiRight = self.nodes[r].phi if (self.nodes[i].phi!=0) else self.nodes[r].phi-2*np.pi

            # finite difference computation      
            if not (b): # boundary conditions
                dr = 0
                dphi = 0
            elif not (u):
                dr = 0
                dphi = (data[l] - data[r]) / (phiLeft - phiRight)
            else:
                dr   = (data[u] - data[b]) / (radiusUp - radiusBottom)
                dphi = (data[l] - data[r]) / (phiLeft - phiRight)

            # nabla computation is cylinder coords
            res[i] = dr + 1 / nod.rad * dphi
        
        return res