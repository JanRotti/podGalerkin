import numpy as np

class cell: 

    def __init__(self,node_list,index):
        
        self.index = index
        
        # nodes spanning cell - quadrilateral
        self.nodes = node_list
        
        # cell attributes
        self.volume = None
        self.center = None
        self.int_coeffs = None

        # value attributes
        self.value = None
        self.dx = None
        self.dy = None
        self.ddx = None
        self.ddy = None

    def compute_volume(self,mesh):
        
        # indexes of cell nodes
        i1 = self.nodes[0]
        i2 = self.nodes[1]
        i3 = self.nodes[2]
        i4 = self.nodes[3]

        # computation of cell volume with arbitary quadrilateral formula
        self.volume = np.abs(((mesh.points[i1][0]*mesh.points[i2][1]-mesh.points[i2][0]*mesh.points[i1][1])+(mesh.points[i2][0]*mesh.points[i3][1]-mesh.points[i2][1]*mesh.points[i3][0])+(mesh.points[i3][0]*mesh.points[i4][1]-mesh.points[i3][1]*mesh.points[i4][0])+(mesh.points[i4][0]*mesh.points[i1][1]-mesh.points[i4][1]*mesh.points[i1][0]))/2)
        
    def compute_center(self,mesh):

        # cell center computation
        self.center = np.mean(mesh.points[self.nodes],axis=0)
    
    def compute_cell_value(self,value_list):
        ## poorly implemented - needs refinement
        # mean of node values
        self.value = 0.25*np.sum(value_list)
        return self.value

    def cell_interpolation(self,data,mesh):  
        
        # index of nodes spanning cell
        i1,i2,i3,i4 = self.nodes

        # cartesian coordinate list of nodes
        x = mesh.points[self.nodes][:,0]
        y = mesh.points[self.nodes][:,1]

        # construction of bilinear mapping matrix
        A = np.array([[x[0]*y[0],x[0],y[0],1],[x[1]*y[1],x[1],y[1],1],
                    [x[2]*y[2],x[2],y[2],1],[x[3]*y[3],x[3],y[3],1]])
        
        # compute inverse of A
        AI = np.linalg.inv(A)

        # solving the inverse problem c = AI*x
        c = np.matmul(AI,data[[i1,i2,i3,i4]])

        # set internal interpolation coefficients
        self.int_coeffs = c

    def get_interpolated_center_value(self):

        # compute cell value from linear cell interpolation
        self.value = self.center[0]*self.center[1]*self.int_coeffs[0] + self.center[0]*self.int_coeffs[1] + self.center[1]*self.int_coeffs[2] + self.int_coeffs[3]
        return self.value

    def get_interpolated_first_derivatives(self):
        
        # compute derivatives from linear cell interpolation
        self.dx = self.center[1]*self.int_coeffs[0] + self.int_coeffs[1]
        self.dy = self.center[0]*self.int_coeffs[0] + self.int_coeffs[2]
