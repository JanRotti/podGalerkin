class node:
    
    def __init__(self,x,y,index):
        
        self.index = index
        
        # cartesian coords
        self.x = x
        self.y = y
        
        # cylinder coordinates
        self.rad = None
        self.theta = None
        
        # neighbor nodes
        self.r = None
        self.l = None
        self.u = None
        self.b = None
        
    def set_cyl(self,r,theta):
        self.rad = r
        self.theta = theta
   
    def set_r(self,r):
        self.r = r
        
    def set_l(self,l):
        self.l = l
        
    def set_u(self,u):
        self.u = u
        
    def set_b(self,b):
        self.b = b
        
    def get_r(self):
        return self.r
        
    def get_l(self):
        return self.l
        
    def get_u(self):
        return self.u
        
    def get_b(self):
        return self.b