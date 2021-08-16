class node:

    i = None
    x = None
    y = None

    # node value
    value = None

    # cartesian derivatives
    dx = None
    dy = None
    laplacian = None

    # cylinder coordinates
    rad = None
    phi = None
    
    # neighbor nodes
    r = None
    l = None
    u = None
    b = None

    # approximate volume participation
    volume = 0

    def __init__(self,x,y,index):
        
        self.i = index
        
        # cartesian coords
        self.x = x
        self.y = y
        
   