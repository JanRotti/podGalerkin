import numpy as np
from tqdm import tqdm
from problem.isentropic_navier_stokes import * 

### GALERKIN PROJECTION
def isentropic_projection(mesh,podModes,qAvg,skalar_product):
    # isentropic navier stokes based coefficients
    dim = podModes.shape[1]
    vecSize = len(qAvg)

    # initialization of coeffiecient structures
    b1 = np.zeros(dim)
    b2 = np.zeros(dim)
    L1 = np.zeros((dim,dim))
    L2 = np.zeros((dim,dim))
    Q = np.array([np.zeros((dim,dim)) for x in range(dim)])

    # computing average operators
    avgDiff = diffusion(mesh,qAvg)
    avgConv = convection(mesh,qAvg,qAvg)

    # temporary convection/diffusion computation
    convTmp1 = np.zeros((dim,vecSize))
    convTmp2 = np.zeros((dim,vecSize))
    convTmp3 = np.zeros((dim,dim,vecSize))
    diffTmp  = np.zeros((dim,vecSize))

    # computation loop for convection/diffusion
    for i in tqdm(range(dim)):
        convTmp1[i] = convection(mesh,qAvg,podModes[:,i])
        convTmp2[i] = convection(mesh,podModes[:,i],qAvg)
        diffTmp[i]  = diffusion(mesh,podModes[:,i])
        for j in range(dim):
            convTmp3[i,j] = convection(mesh,podModes[:,i],podModes[:,j])
    
    # projection via skalar product
    for k in tqdm(range(dim)):
        b1[k] = skalar_product(avgDiff,podModes[:,k])
        b2[k] = skalar_product(avgConv,podModes[:,k])
        for i in range(dim):
            L1[k,i] = skalar_product(diffTmp[i],podModes[:,k])
            L2[k,i] = skalar_product(convTmp1[i] + convTmp2[i],podModes[:,k])
            for j in range(dim):
                Q[k][i,j] = skalar_product(convTmp3[i,j],podModes[:,k])
        
    # output
    print("Projection based Galerkin coefficients in order: b1,b2,L1,L2,Q")
    
    return [b1, b2, L1, L2, Q] 


def isentropic_control_projection(mesh,podModes,qAvg,qCon,skalar_product):
    dim = podModes.shape[1]
    vecSize = len(qAvg)

    # computing control based operators
    Lcon = diffusion(mesh,qCon)
    Qcon = convection(mesh,qCon,qCon)
    QconAvg1 = convection(mesh,qCon,qAvg)
    QconAvg2 = convection(mesh,qAvg,qCon)

    # initialize arrays
    d1 = np.empty(dim)
    d2 = np.empty(dim)
    f = np.empty(dim)
    g = np.empty((dim,dim))
    h = np.empty(dim)

    # compute temporary L and Q operators for projection
    tmp1 = np.empty((dim,vecSize))
    tmp2 = np.empty((dim,vecSize))
    for i in tqdm(range(dim)):
        tmp1[i] = convection(mesh,qCon,podModes[:,i])
        tmp2[i] = convection(mesh,podModes[:,i],qCon)

    # compute finale coefficients
    for k in range(dim):
        d1[k] = skalar_product(Lcon,podModes[:,k])
        d2[k] = skalar_product(QconAvg1 + QconAvg2, podModes[:,k])
        f[k] = skalar_product(Qcon,podModes[:,k])
        h[k] = -skalar_product(qCon,podModes[:,k])
        for i in range(dim):
            g[k,i] = skalar_product(tmp1[i] + tmp2[i],podModes[:,k])

    print("Additional projection based Galerkin coefficients for control in order: d1, d2, f, g, h")
    return [d1, d2, f, g, h]

