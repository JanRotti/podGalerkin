import numpy as np

class isentropic_fluid_problem:

    def __init__(self,mesh):
        self.gamma = 1.4                        # isentropic coefficient
        self.alpha = 1                          # inner product parameter
        self.nu = 1.516e-5                      # kinematic viscosity of fluid
        self.mesh = mesh
        self.n = mesh.n
        self.N = mesh.N
        self.mach_weight = 2*self.alpha/(self.gamma-1)
        
    def L_operator(self,node_data,cell=False):
        ## compute L operator of the problem form \dot{q}=\nu L(q)+ Q(q,q) 
        ## describing the isentropic Navier-Stokes equations

        # data in vectorized form u,v,a -> q (state) 
        u = node_data[:self.n]
        v = node_data[self.n:2*self.n]
        a = node_data[2*self.n:3*self.n]

        # compute node based spatial derivatives 
        ux,uy,uxx,uyy = self.mesh.compute_finite_spatial_derivatives(u,second=True)
        vx,vy,vxx,vyy = self.mesh.compute_finite_spatial_derivatives(v,second=True)
        
        # transform node derivatives to cell derivatives
        if cell:
            uxx = self.mesh.compute_cell_values_from_node_data(uxx)
            uyy = self.mesh.compute_cell_values_from_node_data(uyy)
            vxx = self.mesh.compute_cell_values_from_node_data(vxx)
            vyy = self.mesh.compute_cell_values_from_node_data(vyy)
        
        # return state based operator values
        return np.concatenate((np.add(uxx,uyy),np.add(vxx,vyy),np.zeros(len(uxx))))      
    
    def Q_operator(self,node_data,node_data2,cell=False):
        ## compute Q operator of the problem form \dot{q}=\nu L(q)+ Q(q,q) 
        ## describing the isentropic Navier-Stokes equations
        
        # data in vectorized form u,v,a -> q (state)
        u = node_data[:self.n]
        v = node_data[self.n:2*self.n]
        a = node_data[2*self.n:3*self.n]
        u2 = node_data2[:self.n]
        v2 = node_data2[self.n:2*self.n]
        a2 = node_data2[2*self.n:3*self.n]

        # compute node based spatial derivatives 
        ux,uy = self.mesh.compute_finite_spatial_derivatives(u2,second=False)
        vx,vy = self.mesh.compute_finite_spatial_derivatives(v2,second=False)
        ax,ay = self.mesh.compute_finite_spatial_derivatives(a2,second=False)
        
        # compute node data to cell data
        if cell:
            u = self.mesh.compute_cell_values_from_node_data(u)
            v = self.mesh.compute_cell_values_from_node_data(v)
            a = self.mesh.compute_cell_values_from_node_data(a)
            ux = self.mesh.compute_cell_values_from_node_data(ux)
            vx = self.mesh.compute_cell_values_from_node_data(vx)
            ax = self.mesh.compute_cell_values_from_node_data(ax)
            uy = self.mesh.compute_cell_values_from_node_data(uy)
            vy = self.mesh.compute_cell_values_from_node_data(vy)
            ay = self.mesh.compute_cell_values_from_node_data(ay)

        # compute state based cell values
        q1 = np.multiply(u,ux) + np.multiply(v,uy) + 2/(self.gamma-1) * np.multiply(a,ax)
        q2 = np.multiply(u,vx) + np.multiply(v,vy) + 2/(self.gamma-1) * np.multiply(a,ay)
        q3 = np.multiply(u,ax) + np.multiply(v,ay) + (self.gamma-1)/2 * np.multiply(a,np.add(ux,vy))
        
        # return state based operator values
        return np.concatenate((-q1,-q2,-q3))        

    def skalar_product(self,data,data2):
        ## energy based inner product
        
        integral = 0

        # case node data
        if len(data)/3==self.n and len(data2)/3==self.n:
            for node in self.mesh.nodes:
                i = node.index
                # summation over nodes with corresponding node volume 
                integral += (data[i]*data2[i]+data[2*i]*data2[2*i]+self.mach_weight*data[3*i]*data2[3*i]) * node.dv
            return integral
        # case cell data
        elif len(data)/3==self.N and len(data2)/3==self.N:
            for cell in self.mesh.cells:
                i = cell.index
                # summation over ncells with corresponding cell volumes
                integral += (data[i]*data2[i]+data[2*i]*data2[2*i]+self.mach_weight*data[3*i]*data2[3*i]) * cell.volume
            return integral
        # case invalid dimensions
        else:
            raise ValueError("Invalid Dimension in data vectors!")



class inc_fluid_problem:

    def __init__(self,mesh):
        self.description = "inc"
        self.n = mesh.n                                 # number of points, len of point array
        self.N = mesh.N                                 # number of cells, len of cell array
        self.mesh = mesh                                # set reference object
        self.rho = 1.2886                               # density
        self.mu = 1.716E-5                              # dynamic viscosity
        self.nu = self.mu / self.rho                    # kinematic viscosity
        self.q = None

    def set_data_dict(self,data_dict):
        # state data of shape (n,number of timesteps)
        self.q = np.vstack([data_dict["u"],data_dict["v"],data_dict["p"]]) # with u,v,p

    def first_derivatives(self,cell_data):
        
        ### cell based computation of derivatives
        
        # disassemble data vector
        u = cell_data[:self.N]
        v = cell_data[self.N:2*self.N]
        p = cell_data[2*self.N:3*self.N]
        
        # initialize data arrays
        dudx = np.array(self.N) 
        dudy = np.array(self.N) 
        dvdx = np.array(self.N) 
        dvdy = np.array(self.N) 
        dpdx = np.array(self.N) 
        dpdy = np.array(self.N) 
        
        # compute spatial derivatives
        self.mesh.compute_cell_interpolations(u)
        self.mesh.compute_cell_derivatives()
        for cell in self.mesh.cells:
            dudx[cell.index] = cell.dx
            dudy[cell.index] = cell.dy
        
        self.mesh.compute_cell_interpolations(v)
        self.mesh.compute_cell_derivatives()
        for cell in self.mesh.cells:
            dvdx[cell.index] = cell.dx
            dvdy[cell.index] = cell.dy
        
        self.mesh.compute_cell_interpolations(p)
        self.mesh.compute_cell_derivatives()
        for cell in self.mesh.cells:
            dpdx[cell.index] = cell.dx
            dpdy[cell.index] = cell.dy
        
        # return data based spatial derivatives
        return [dudx,dudy,dvdx,dvdy,dpdx,dpdy]
    
    def inner_product(self,data,data2):
        
        # data assignment
        u = data[:self.N]
        v = data[self.N:2*self.N]
        p = data[2*self.N:3*self.N]
        u2 = data2[:self.N]
        v2 = data2[self.N:2*self.N]
        p2 = data2[2*self.N:3*self.N]

        # form integral
        integral = 0
        for cell in self.mesh.cells:
            i = cell.index
            dv = cell.volume
            integral += (u[i]*u2[i]+v[i]*v2[i]+p[i]*p2[i]) * dv






