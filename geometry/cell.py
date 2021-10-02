import numpy as np

class cell: 

    def __init__(self,nodeList,index):
        
        self.i = index
        
        # nodes spanning cell - quadrilateral
        self.nodes = nodeList
        
        # cell attributes
        self.volume = None
        self.center = None
        self.coeffs = None

        # value attributes
        self.value = None
        self.dx = None
        self.dy = None

    def compute_volume(self,mesh):
        
        # indexes of cell nodes
        i1 = self.nodes[0]
        i2 = self.nodes[1]
        i3 = self.nodes[2]
        i4 = self.nodes[3]

        # computation of cell volume with arbitary quadrilateral formula
        self.volume = np.abs(((mesh.points[i1][0]*mesh.points[i2][1]-mesh.points[i2][0]*mesh.points[i1][1])+(mesh.points[i2][0]*mesh.points[i3][1]-mesh.points[i2][1]*mesh.points[i3][0])+(mesh.points[i3][0]*mesh.points[i4][1]-mesh.points[i3][1]*mesh.points[i4][0])+(mesh.points[i4][0]*mesh.points[i1][1]-mesh.points[i4][1]*mesh.points[i1][0]))/2)
        
    def compute_center(self,mesh):
        
        # cell center from node coordinate average
        self.center = np.mean(mesh.points[self.nodes],axis=0)
    
    def compute_cell_value(self,data):
        
        # compute cell center value by weighted average
        self.value = 0.25*np.sum(data[self.nodes])
        
        return self.value