import numpy as np

class derivatives:
    
    def __init__(self,mesh):
        
        self.mesh = mesh
        
        # set lenghts scales 
        self.n = mesh.n                     # length of points array
        self.N = mesh.N                     # length of cells array
        
        # delivering class instances for cells, nodes and coordinates
        self.points_cylinder = mesh.points_cylinder
        self.points = mesh.points
        self.nodes = mesh.nodes
        self.cells = mesh.cells
    
    def compute_finite_spatial_derivatives(self,data,second=False):
        ## finite differnce method
        
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
                    #ddr = ((node.rad-self.nodes[b].rad)*(data[u]-data[i])-(self.nodes[u].rad-node.rad)*(data[i]-data[b]))/(0.5*(self.nodes[u].rad-node.rad)*(node.rad-self.nodes[b].rad)*(self.nodes[u].rad-self.nodes[b].rad))
                
                # compute second transform derivatives
                ddthetadx = (2*node.y*node.x)/(np.power(node.rad,4))
                ddrdx = (node.y*node.y)/(np.power(node.rad,3))
                ddthetady = -(2*node.y*node.x)/(np.power(node.rad,4))
                ddrdy = (node.x*node.x)/(np.power(node.rad,3))
                
                # compute second cartesian derivatives
                sec_derivs_x[i] = np.multiply(ddtheta,ddthetadx) + np.multiply(ddr,ddrdx) + 2 * np.multiply(np.multiply(dtheta,dthetadx),np.multiply(dr,drdx))
                sec_derivs_y[i] = np.multiply(ddtheta,ddthetady) + np.multiply(ddr,ddrdy) + 2 * np.multiply(np.multiply(dtheta,dthetady),np.multiply(dr,drdy))
        
        
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





