import numpy as np
from derivatives import *

def Q(mesh,q1,q2,method="fd"):
    gamma = 1.4
    ## methods for node based calculation
    # fd -> finite differences
    # pd -> polynomial approximation
    ## methods for cell based calculation
    # bm -> bilinear face interpolation
    
    d = int(len(q1)/3)
    
    # decomposition into variable vectors
    u1 = q1[:d]
    v1 = q1[d:2*d]
    a1 = q1[2*d:3*d]
    u2 = q2[:d]
    v2 = q2[d:2*d]
    a2 = q2[2*d:3*d]

    # initializing spatial derivatives
    u2x = np.empty(d)
    v2x = np.empty(d)
    a2x = np.empty(d)
    u2y = np.empty(d)
    v2y = np.empty(d)
    a2y = np.empty(d)

    # computation of spatial derivatives
    if len(q1==mesh.n):
        if method=="fd":
            finite_differences(mesh,u2)
        elif method=="pd":
            polynomial_derivatives(mesh,u2)
        else:
            raise ValueError("Invalid for node based derivatives!")
        for nod in mesh.nodes:
            u2x[nod.index]=nod.dx
            u2y[nod.index]=nod.dy
        
        if method=="fd":
            finite_differences(mesh,v2)
        elif method=="pd":
            polynomial_derivatives(mesh,v2)
        for nod in mesh.nodes:
            v2x[nod.index]=nod.dx
            v2y[nod.index]=nod.dy

        if method=="fd":
            finite_differences(mesh,a2)
        elif method=="pd":
            polynomial_derivatives(mesh,a2)    
        for nod in mesh.nodes:
            a2x[nod.index]=nod.dx
            a2y[nod.index]=nod.dy
    elif len(q1==mesh.N):
        if method!="bm":
            raise ValueError("Selected method not applicaple to cell based data! Please select bilinear mapping!")
        bilinear_derivatives(mesh,u2)
        for nod in mesh.nodes:
            u2x[nod.index]=nod.dx
            u2y[nod.index]=nod.dy
        bilinear_derivatives(mesh,v2)
        for nod in mesh.nodes:
            v2x[nod.index]=nod.dx
            v2y[nod.index]=nod.dy
        bilinear_derivatives(mesh,a2)
        for nod in mesh.nodes:
            a2x[nod.index]=nod.dx
            a2y[nod.index]=nod.dy
    else:
        raise ValueError("Invalid dimension for input data!")
    
    # computation of index based operator
    u_tmp = np.multiply(u1,u2x) + np.multiply(v1,u2y) + 2/(gamma - 1)*np.multiply(a1,a2x)
    v_tmp = np.multiply(u1,v2x) + np.multiply(v1,v2y) + 2/(gamma - 1)*np.multiply(a1,a2y)
    a_tmp = np.multiply(u1,a2x) + np.multiply(v1,a2y) + (gamma - 1)/2*np.multiply(a1,np.add(u2x,u2y))
    return -1 * np.concatenate((u_tmp,v_tmp,a_tmp))

def L(mesh,q,method="fd"):
    # fd -> finite differences
    # pd -> polynomial approximation   
    d = int(len(q)/3)
    
    # decomposition into variable vectors
    u = q[:d]
    v = q[d:2*d]
    a = q[2*d:3*d]    

    if len(q)!=mesh.n:
        ValueError("Invalid data dimension!")
    
    # initializing spatial derivatives
    uxx = np.empty(d)
    uyy = np.empty(d)
    vxx = np.empty(d)
    vyy = np.empty(d)
    
    # computing second derivatives
    if method=="fd":
        finite_differences(mesh,u,second=True)
    elif method=="pd":
        polynomial_derivatives(mesh,u,second=True)
    else:
        raise ValueError("Invalid for node based derivatives!")
    for nod in mesh.nodes:
        uxx[nod.index]=nod.ddx
        uyy[nod.index]=nod.ddy
    
    if method=="fd":
        finite_differences(mesh,v,second=True)
    elif method=="pd":
        polynomial_derivatives(mesh,v,second=True)
    for nod in mesh.nodes:
        vxx[nod.index]=nod.ddx
        vyy[nod.index]=nod.ddy

    # computation of index based operator
    u_tmp = np.add(uxx,uyy)
    v_tmp = np.add(vxx,vyy)
    a_tmp = np.zeros(d)
    return np.concatenate((u_tmp,v_tmp,a_tmp))

def inner_product(mesh,q1,q2):
    ## energy based inner product
    integral = 0    
    alpha = 1
    gamma = 1.4
    mach_weight = 2 * alpha / (gamma - 1)
    
    d = int(len(q1)/3)
    f = int(len(q2)/3)

    # case node data
    if d==mesh.n and f==mesh.n:
        for nod in mesh.nodes:
            i = nod.index
            # summation over nodes with corresponding node volume 
            integral += (q1[i]*q2[i]+q1[2*i]*q2[2*i]+mach_weight*q1[3*i]*q2[3*i]) * nod.dv
        return integral
        
    # case cell data
    elif d==mesh.N and f==mesh.N:
        for cel in mesh.cells:
            i = cel.index
            # summation over ncells with corresponding cell volumes
            integral += (q1[i]*q2[i]+q1[2*i]*q2[2*i]+mach_weight*q1[3*i]*q2[3*i]) * cel.volume
        return integral
    # case invalid dimensions
    else:
        raise ValueError("Invalid Dimension in data vectors!")
    return 0