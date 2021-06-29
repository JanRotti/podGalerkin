import numpy as np
from derivatives import *

def Q(mesh,q1,q2,method="fd",output="node"):
    # quadratic navier stokes operator inkompressible
    ## methods for node based calculation
    # fd -> finite differences
    # pd -> polynomial approximation
    ## methods for cell based calculation
    # bm -> bilinear face interpolation
    
    d = int(len(q1)/2)
    
    # decomposition into variable vectors
    u1 = q1[:d]
    v1 = q1[d:2*d]
    u2 = q2[:d]
    v2 = q2[d:2*d]

    # initializing spatial derivatives
    if (output=="cell" and method=="bm"):
        d = int(mesh.N/2)
    
    u2x = np.empty(d)
    v2x = np.empty(d)
    u2y = np.empty(d)
    v2y = np.empty(d)


    # cell based data
    if output=="cell" and len(q1==mesh.n):
        u1 = mesh.compute_cell_values_from_node_data(u1)
        v1 = mesh.compute_cell_values_from_node_data(v1)
        u2_cell = mesh.compute_cell_values_from_node_data(u2)
        v2_cell = mesh.compute_cell_values_from_node_data(v2)

    # computation of spatial derivatives
    if len(q1==mesh.n) and not method=="bm":
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

    elif len(q1==mesh.N):
        if method!="bm":
            raise ValueError("Selected method not applicaple to cell based data! Please select bilinear mapping!")
        bilinear_derivatives(mesh,u2)
        for cel in mesh.nodes:
            u2x[cel.index]=cel.dx
            u2y[cel.index]=cel.dy
        bilinear_derivatives(mesh,v2)
        for nod in mesh.nodes:
            v2x[cel.index]=cel.dx
            v2y[cel.index]=cel.dy
    
    elif len(q1==mesh.n) and (output=="cell" and method=="bm"):
        bilinear_derivatives(mesh,u2_cell)
        for cel in mesh.nodes:
            u2x[cel.index]=cel.dx
            u2y[cel.index]=cel.dy
        bilinear_derivatives(mesh,v2_cell)
        for nod in mesh.nodes:
            v2x[cel.index]=cel.dx
            v2y[cel.index]=cel.dy
    
    else:
        raise ValueError("Invalid dimension for input data!")
    
    if(output=="cell" and method!="bm"):
        u2x = mesh.compute_cell_values_from_node_data(u2x)
        u2y = mesh.compute_cell_values_from_node_data(u2y)
        v2x = mesh.compute_cell_values_from_node_data(v2x)
        v2y = mesh.compute_cell_values_from_node_data(v2y)

    # computation of index based operator
    u_tmp = np.multiply(u1,u2x) + np.multiply(v1,u2y)
    v_tmp = np.multiply(u1,v2x) + np.multiply(v1,v2y)
    return -1 * np.concatenate((u_tmp,v_tmp))


def L(mesh,q,method="fd",output="node"):
    # linear navier stokes operator
    
    # fd -> finite differences
    # pd -> polynomial approximation   
    d = mesh.n
    
    # decomposition into variable vectors
    u = q[:d]
    v = q[d:2*d]

    if len(q)!=mesh.n:
        ValueError("Invalid data dimension!")
    
    # initializing spatial derivatives
    ulap = np.empty(d)
    vlap = np.empty(d)

    # computing second derivatives
    if method=="fd":
        finite_differences(mesh,u,second=True)
    elif method=="pd":
        polynomial_derivatives(mesh,u,second=True)
    else:
        raise ValueError("Invalid for node based derivatives!")
    for nod in mesh.nodes:
        ulap[nod.index]=nod.laplacian
    
    if method=="fd":
        finite_differences(mesh,v,second=True)
    elif method=="pd":
        polynomial_derivatives(mesh,v,second=True)
    for nod in mesh.nodes:
        vlap[nod.index]=nod.laplacian

    if output=="cell":
        ulap = mesh.compute_cell_values_from_node_data(ulap)
        vlap = mesh.compute_cell_values_from_node_data(vlap)
        d = int(mesh.N)

    # computation of index based operator
    u_tmp = ulap
    v_tmp = vlap
    return np.concatenate((u_tmp,v_tmp))


# volume based inner product for incompressible navier stokes with 3 and 2 design variables
def inner_product(mesh,q1,q2):

    integral = 0

    # vector length cases
    d = int(len(q1)/3)
    f = int(len(q2)/3)
    g = int(len(q1)/2)
    h = int(len(q2)/2)

    # case node data 3 design variables
    if d==mesh.n and f==mesh.n:
        for nod in mesh.nodes:
            i = nod.index
            # summation over nodes with corresponding node volume 
            integral += (q1[i]*q2[i]+q1[d+i]*q2[d+i]+q1[2*d+i]*q2[2*d+i]) * nod.dv
        return integral

    # case node data 2 design variables
    elif g==mesh.n and h==mesh.n:
        for nod in mesh.nodes:
            i = nod.index
            # summation over nodes with corresponding node volume 
            integral += (q1[i]*q2[i]+q1[g+i]*q2[g+i]) * nod.dv
        return integral

    # case cell data 3 design variables
    elif d==mesh.N and f==mesh.N:
        for cel in mesh.cells:
            i = cel.index
            # summation over ncells with corresponding cell volumes
            integral += (q1[i]*q2[i]+q1[d+i]*q2[d+i]+q1[2*d+i]*q2[2*d+i]) * cel.volume
        return integral

    # case cell data 2 design variables
    elif g==mesh.N and h==mesh.N:
        for cel in mesh.cells:
            i = cel.index
            # summation over ncells with corresponding cell volumes
            integral += (q1[i]*q2[i]+q1[d+i]*q2[d+i]) * cel.volume
        return integral

    # case invalid dimensions
    else:
        raise ValueError("Invalid Dimension in data vectors!")
