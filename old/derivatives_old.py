import numpy as np

### FINITE DIFFERENCE METHOD
def finite_differences(mesh,data,fd=False,second=False):
    # data vector with length n
    for nod in mesh.nodes:
        
        # stencil indizes
        i = nod.index
        r = nod.get_r()
        l = nod.get_l()
        u = nod.get_u() if nod.get_u() else None  # special boundary case
        b = nod.get_b() if nod.get_b() else None  # special boundary case

        # transform derivatives
        drdx = nod.x/nod.rad
        dthetadx = -nod.y/(nod.rad**2)
        drdy = nod.y/nod.rad
        dthetady = nod.x/(nod.rad**2)     
        if second:
            ddthetadx = (2*nod.y*nod.x)/(np.power(nod.rad,4))
            ddrdx = (nod.y*nod.y)/(np.power(nod.rad,3))
            ddthetady = -(2*nod.y*nod.x)/(np.power(nod.rad,4))
            ddrdy = (nod.x*nod.x)/(np.power(nod.rad,3))
        # compute cylindrical derivatives
        rad_u = mesh.nodes[u].rad if (u) else None
        rad_b = mesh.nodes[b].rad if (b) else None
        theta_l = mesh.nodes[l].theta if (mesh.nodes[l].theta!=0) else 2*np.pi
        theta_r = mesh.nodes[r].theta if (mesh.nodes[i].theta!=0) else mesh.nodes[r].theta-2*np.pi
        if not (u) or not (b):
            dr = 0
            dtheta = 0
        else:
            if fd:
                dr = (data[u]-data[i])/(rad_u-nod.rad)
                dtheta = (data[l]-data[i])/(theta_l-nod.theta)
            else:
                dr = (data[u]-data[b])/(rad_u-rad_b)
                dtheta = (data[l]-data[r])/(theta_l-theta_r)

        # transform to cartesian coordinates
        nod.dx = dtheta*dthetadx + dr*drdx
        nod.dy = dtheta*dthetady + dr*drdy

        if second:
            if not (u) or not (b):
                ddr = 0
                ddtheta = 0
            else:
                ddtheta = (data[l]-2*data[i]+data[r])/((theta_l-nod.theta)*(nod.theta-theta_r)) 
                ddr = (data[u]*(nod.rad-rad_b)+data[b]*(rad_u-nod.rad)-data[i]*(rad_u-rad_b))/((rad_u-nod.rad)**2*(nod.rad-rad_b))
            nod.ddx = ddtheta*ddthetadx+ddr*ddrdx
            nod.ddy = ddtheta*ddthetady+ddr*ddrdy

### POLYNOMIAL APPROXIMATION METHOD
def polynomial_derivatives(mesh,data,second=False):
    # data vector with length n
    for nod in mesh.nodes:
        
        # stencil indizes
        i = nod.index
        r = nod.get_r()
        l = nod.get_l()
        u = nod.get_u() if nod.get_u() else None  # special boundary case
        b = nod.get_b() if nod.get_b() else None  # special boundary case  

        # transform derivatives
        drdx = nod.x/nod.rad
        dthetadx = -nod.y/(nod.rad**2)
        drdy = nod.y/nod.rad
        dthetady = nod.x/(nod.rad**2)    

        if second:
            ddthetadx = (2*nod.y*nod.x)/(np.power(nod.rad,4))
            ddrdx = (nod.y*nod.y)/(np.power(nod.rad,3))
            ddthetady = -(2*nod.y*nod.x)/(np.power(nod.rad,4))
            ddrdy = (nod.x*nod.x)/(np.power(nod.rad,3))
        if not (u) or not (b):
            nod.dx = 0
            nod.ddx = 0
            nod.dy = 0
            nod.ddy = 0
        else:
            # polynomial fitting
            a_phi = (data[l]-data[r])/(nod.theta-mesh.nodes[r].theta)-(data[i]-data[r])/(mesh.nodes[l].theta-mesh.nodes[r].theta)
            b_phi = (data[i]-data[r])/(nod.theta-mesh.nodes[r].theta)-(data[l]-data[r])+(data[i]-data[r])*(nod.theta-mesh.nodes[r].theta)/(mesh.nodes[l].theta-mesh.nodes[r].theta)
            c_phi = data[r]
            dpol_phi = 2*a_phi*(nod.theta-mesh.nodes[r].theta) + b_phi
            ddpol_phi = a_phi
    
            a_rad = (data[u]-data[b])/(nod.rad-mesh.nodes[b].rad)-(data[i]-data[b])/(mesh.nodes[u].rad-mesh.nodes[b].rad)
            b_rad = (data[i]-data[b])/(nod.rad-mesh.nodes[b].rad)-(data[l]-data[b])+(data[i]-data[b])*(nod.rad-mesh.nodes[b].rad)/(mesh.nodes[u].rad-mesh.nodes[b].rad)
            c_rad = data[r]
            dpol_rad = 2*a_rad*(nod.rad-mesh.nodes[b].rad) + b_rad
            ddpol_rad = a_rad
            
            # transform to cartesian coordinates
            nod.dx = dpol_phi*dthetadx + dpol_rad*drdx
            nod.dy = dpol_phi*dthetady + dpol_rad*drdy
            if second:
                nod.ddx = ddpol_phi*ddthetadx+ddpol_rad*ddrdx+2*dpol_rad*drdx*dpol_phi*dthetadx
                nod.ddy = ddpol_phi*ddthetady+ddpol_rad*ddrdy+2*dpol_phi*dthetady*dpol_rad*drdy
                
### BILINEAR FACE INTERPOLATION
# for cell centered derivatives
def bilinear_derivatives(mesh,data,second=False):
    for cel in mesh.cells:

        # retrieve cartesian coordinates of cell nodes
        x = mesh.nodes[cel.nodes][:,0]
        y = mesh.nodes[cel.nodes][:,1]

        # construct condition matrix
        A = np.array([[x[0]*y[0],x[0],y[0],1],[x[1]*y[1],x[1],y[1],1],
                    [x[2]*y[2],x[2],y[2],1],[x[3]*y[3],x[3],y[3],1]])

        # compute inverse of A
        AI = np.linalg.inv(A)

        c = np.matmul(AI,data[cel.nodes])
        cel.int_coeffs = c

        cel.dx = c[0] * cel.center[1] + c[1]
        cel.dy = c[0] * cel.center[0] + c[2]


