import numpy as np

def get_POD(snapshots,skalar_weights,max_POD=10):
    
    # POD by method of snapshots
    T = snapshots.shape[1]

    # construct correlation matrix
    C = np.empty((T,T))
    for i in range(T):
        for j in range(i,T):
            C[i,j] = np.sum(snapshots[:,i] * snapshots[:,j] * skalar_weights)
            C[j,i] = C[i,j]  # C is symmetric by construction

    S, V =  np.linalg.eigh(C,UPLO='L')
    S = np.flip(S,0)
    V = np.flip(V,1)

    # construct spatial POD Modes
    pod_modes = np.zeros((snapshots.shape[0],max_POD))
    for i in range(max_POD):
        pod_modes[:,i] = 1 / np.sqrt(S[i]) *  np.matmul(snapshots,V[:,i])

    # computing eigenvalues
    S = np.zeros(max_POD)
    for i in range(max_POD):
        for j in range(T):
            S[i] += np.sum(snapshots[:,j] * pod_modes[:,i] * skalar_weights)**2

    return [pod_modes, S]

def get_activations(snapshots,pod_modes,skalar_weights,rec_num=0):
    if rec_num==0:
        rec_num = pod_modes.shape[1]
    activations = np.zeros((pod_modes.shape[1],snapshots.shape[1]))
    for t in range(snapshots.shape[1]):
        for i in range(rec_num):
            activations[i,t] = np.sum(snapshots[:,t] * pod_modes[:,i] * skalar_weights)
    return activations