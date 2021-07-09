import numpy as np

def skalar_product(d1, d2, w):
    return np.sum(d1*d2*w)

def convection(mesh,q1,q2,gamma=1.4):
    n = mesh.n

    # local variable vectors
    u1 = q1[:n]
    v1 = q1[n:2*n]
    a1 = q1[2*n:3*n]
    u2 = q2[:n]
    v2 = q2[n:2*n]
    a2 = q2[2*n:3*n]

    # derivatives
    [u2x, u2y] = mesh.finite_differences(u2)
    [v2x, v2y] = mesh.finite_differences(v2)
    [a2x, a2y] = mesh.finite_differences(a2)

    # state based convection
    u_tmp = np.multiply(u1,u2x) + np.multiply(v1,u2y) + (2 / (gamma - 1)) * np.multiply(a1,a2x)
    v_tmp = np.multiply(u1,v2x) + np.multiply(v1,v2y) + (2 / (gamma - 1)) * np.multiply(a1,a2y)
    a_tmp = np.multiply(u1,a2x) + np.multiply(v1,a2y) + ((gamma - 1) / 2) * np.multiply(a1,u2x + u2y)
    
    return -1 * np.concatenate((u_tmp,v_tmp,a_tmp))

def diffusion(mesh,q):
    n = mesh.n
    
    # local variable vectors
    u = q[:n]
    v = q[n:2*n]

    # derivatives
    [_, _, ulap] = mesh.finite_differences(u,compute_laplacian=True)
    [_, _, vlap] = mesh.finite_differences(v,compute_laplacian=True)

    # state based diffusion
    u_tmp = ulap
    v_tmp = vlap
    a_tmp = np.zeros(n)
    return np.concatenate((u_tmp,v_tmp,a_tmp))

def curl(mesh,q):
    u = q[:mesh.n]
    v = q[mesh.n:2*mesh.n]

    [_,uy] = mesh.finite_differences(data = u)
    [vx,_] = mesh.finite_differences(data = v)

    return (vx - uy)

def laplacian(mesh,q):
    u = q[:mesh.n]
    v = q[mesh.n:2*mesh.n]
    
    # derivatives
    [_, _, ulap] = mesh.finite_differences(u,compute_laplacian=True)
    [_, _, vlap] = mesh.finite_differences(v,compute_laplacian=True)

    return np.concatenate((ulap,vlap))