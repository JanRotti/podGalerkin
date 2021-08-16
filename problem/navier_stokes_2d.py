import numpy as np

def curl(mesh,q):
    u = q[:mesh.n]
    v = q[mesh.n:2*mesh.n]

    [_,uy] = mesh.finite_differences(u)
    [vx,_] = mesh.finite_differences(v)

    return (vx - uy)

def laplacian(mesh,q):
    u = q[:mesh.n]
    v = q[mesh.n:2*mesh.n]
    
    # derivatives
    [_, _, ulap] = mesh.finite_differences(u,compute_laplacian=True)
    [_, _, vlap] = mesh.finite_differences(v,compute_laplacian=True)

    return np.concatenate((ulap,vlap))

def diffusion(mesh,q):
    
    # local variable vectors
    u = q[:mesh.n]
    v = q[mesh.n:2*mesh.n]

    # derivatives
    [_, _, ulap] = mesh.finite_differences(u,compute_laplacian=True)
    [_, _, vlap] = mesh.finite_differences(v,compute_laplacian=True)

    return np.concatenate((ulap,vlap))

def convection(mesh,q1,q2):
    n = mesh.n

    # local variable vectors
    u1 = q1[:n]
    v1 = q1[n:2*n]
    u2 = q2[:n]
    v2 = q2[n:2*n]

    # derivatives
    [u2x, u2y] = mesh.finite_differences(u2)
    [v2x, v2y] = mesh.finite_differences(v2)

    # convection
    u_tmp = np.multiply(u1,u2x) + np.multiply(v1,u2y)
    v_tmp = np.multiply(u1,v2x) + np.multiply(v1,v2y) 
    
    return np.concatenate((u_tmp,v_tmp))