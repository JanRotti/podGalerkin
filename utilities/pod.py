import numpy as np

def get_POD(snapshots,skalar_product,maxPOD=10):
    ### METHOD OF SNAPSHOTS
    
    # input dimensions
    T = snapshots.shape[1]
    n = snapshots.shape[0]

    # correlation matrix
    C = np.empty((T,T))
    for i in range(T):
        # utilize symmetry of C
        for j in range(i,T):
            C[i,j] = skalar_product(snapshots[:,i], snapshots[:,j])
            C[j,i] = C[i,j] # symmetry property

    # eigenvalue problem of correlation matrix
    S, V =  np.linalg.eigh(C,UPLO='L')
    # flip due to return structure of bp.linalg.eigh
    S = np.flip(S,0) # make S in descending order
    V = np.flip(V,1) # make V correspondingly

    # construct spatial POD Modes from snapshots
    podModes = np.zeros((n,maxPOD))
    for i in range(maxPOD): # faster than straight matrix multiplication!
        podModes[:,i] = 1 / np.sqrt(S[i]) *  np.matmul(snapshots,V[:,i])

    # computing eigenvalues from snapshots
    S = np.zeros(maxPOD)
    for i in range(maxPOD):
        for j in range(T):
            S[i] += skalar_product(snapshots[:,j], podModes[:,i])**2

    return [podModes, S]

def get_activations(snapshots,podModes,skalar_product,recNum=0):
    ### COMPUTE REFERENCE ACTIVATIONS
    
    # input dimensions
    n = podModes.shape[1]
    T = snapshots.shape[1]

    # if reconstruction dimension is 0 -> use all possible vectors
    if recNum==0:
        recNum = n

    # compute mode activation by projection
    activations = np.zeros((n,T))
    for t in range(T):
        for i in range(recNum):
            activations[i,t] = skalar_product(snapshots[:,t], podModes[:,i])
    
    return activations