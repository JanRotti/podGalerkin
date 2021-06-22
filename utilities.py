import numpy as np
import math
import os
import csv
import meshio
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

def mydist(xm,ym,x,y):
    return math.sqrt((x-xm)*(x-xm)+(y-ym)*(y-ym))

# axes plotting for fixed mesh with central cylinder
def plot_cylinder(x,y,data,cmap='cividis',levels=100,ax=None,clip=True,shifted=False):
    if ax==None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    # set max and min
    if clip:
        vmax = 0.05
        vmin = -0.05
        data = np.clip(data,vmin,vmax)
    
    # plot central cylinder
    if shifted:
        center = (0,0)
    else:
        center = (0.5,0)
    cylinder = plt.Circle(center,0.5, color="grey")
    cylinder_border = plt.Circle(center,0.5,color="black", fill=False)

    ax.add_patch(cylinder)
    ax.add_patch(cylinder_border)
    
    # add grid
    grid_x, grid_y = np.mgrid[-2:10:1200j, -5:5:1000j]
    points = np.stack((x,y),axis=1)
    
    # form interpolated grid
    grid = griddata(points, data, (grid_x, grid_y))
    
    # set cylinder area values to 0
    r = 50
    if shifted:
        xm = 200
        ym = 500
        for i in range(150,250,1):
            for j in range(450,550,1):
                if mydist(xm,ym,i,j)<r:
                    grid[i,j] = 0
    else:
        xm = 250
        ym = 500        
        for i in range(200,300,1):
            for j in range(450,550,1):
                if mydist(xm,ym,i,j)<r:
                    grid[i,j] = 0
    
    # add contourf plot
    cntr = ax.contourf(grid_x,grid_y,grid,levels,cmap=cmap)
    if levels<=40:
        for line, level in zip(cntr.collections, cntr.levels):
            if level < 0:
                f = line.get_fc()
                if not np.array_equal(f,np.array([[1,1,1,1]])):
                    line.set_linestyle('dotted')
                    line.set_edgecolor('black')
                    line.set_lw(0.1)
            else:
                f = line.get_fc()
                if not np.array_equal(f,np.array([[1,1,1,1]])):
                    line.set_linestyle('dotted')
                    line.set_edgecolor('black')
                    line.set_lw(0.1)
                
    # add colorbar
    plt.colorbar(cntr,ax=ax)

    # set axis options
    ax.set_xlim([-2,10])
    ax.set_ylim([-5,5])
    ax.axis('off')

# reading flow data from files
def get_flow_data(dir_name,delay=2000,subsampling=2,max_size=500):
    file_list = os.listdir(dir_name)
    files = len(file_list) - delay
    filenames = "restart_flow_"
    gamma = 1.4
    alpha = 1

    try:
        with open(dir_name+file_list[0]) as f:
            data_iter = csv.reader(f,delimiter=",")
            N = sum(1 for _ in data_iter) - 1
    except IOError:
        print("Files not accessible")
    
    # compute array size
    if files > max_size:
        if (files)//subsampling > max_size:
            array_size = max_size
        else:
            array_size = files//subsampling
    else:
        array_size = files

    # initializing data matrix
    rho_arr = np.empty([N,array_size])     # density
    p_arr = np.empty([N,array_size])       # pressure
    vort_arr = np.empty([N,array_size])    # vorticity
    e_arr = np.empty([N,array_size])       # energy
    u_arr = np.empty([N,array_size])       # velocity in x
    v_arr = np.empty([N,array_size])       # velocity in y
    a_arr = np.empty([N,array_size])       # speed of sound

    for i in tqdm(range(array_size)):  
        num = i * subsampling + delay
        fname = dir_name + filenames + "{:05d}".format(num) + ".csv"
        # reading file routine
        [pid,x,y,rho,mx,my,p,e,vort] = read_data_from_csv(fname)
        rho_arr[:,i] = rho
        p_arr[:,i]   = p
        vort_arr[:,i]= vort  
        e_arr[:,i]= e 
        u_arr[:,i] = np.divide(mx,rho)
        v_arr[:,i] = np.divide(my,rho)
        a_arr[:,i] = np.sqrt((gamma-1)*e*gamma)

    return {"u":u_arr,"v":v_arr,"a":a_arr,"e":e_arr,"p":p_arr,"rho":rho_arr,"vort":vort_arr}

def get_x_y(dir_name):
    [pid,x,y,rho,mx,my,p,e,vort] = read_data_from_csv(dir_name+os.listdir(dir_name)[0])
    return (x,y)

# get single file data readings
def read_data_from_csv(fname):
    try:
        with open(fname) as f:
            data_iter = csv.reader(f,delimiter=",")
            data = [row[:] for row in data_iter] # skip first iteration due to headers
        data = np.asarray(data[1:],dtype='float64')
        pid = data[:,0]     # id
        X = data[:,1]       # x-coordinate
        Y = data[:,2]       # y-coordinate
        rho = data[:,3]     # density
        mx = data[:,4]      # momentum in x
        my = data[:,5]      # momentum in y 
        p = data[:,7]       # pressure
        e = data[:,-2]      # energy
        vort = data[:,-1]   # vorticity
        return [pid,X,Y,rho,mx,my,p,e,vort]
    except IOError:   
        print("File not accessible")

# plotting a set number of eigenvalues in U onto the x,y grid
def plot_eigenflows(x,y,U,N,num):
    # construct subplot structure
    fig,ax = plt.subplots(num,1,figsize=(15,num*10))
    fig.tight_layout(pad=3.0)

    for i in range(num):
        data = np.sqrt(np.multiply(U[:N,i],U[:N,i])+np.multiply(U[N:2*N,i],U[N:2*N,i]))
        plot_cylinder(x,y,data,ax=ax[i],cmap='cividis')

# returns a colormap (viridis or custom log-wbr) 
def get_cmap(custom=True):
    # return matplotlib ListedColormap
    if custom:
        CC = np.array([
    [0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],
    [0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],
    [0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],
    [0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],
    [0,1,1,1],[0,1,1,1],
    [0,0.888888895511627,1,1],[0,0.777777791023254,1,1],
    [0,0.666666686534882,1,1],[0,0.555555582046509,1,1],
    [0,0.444444447755814,1,1],[0,0.333333343267441,1,1],
    [0,0.222222223877907,1,1],[0,0.111111111938953,1,1],
    [0,0,1,1],
    [0.200000002980232,0.200000002980232,1,1],
    [0.400000005960465,0.400000005960465,1,1],
    [0.600000023841858,0.600000023841858,1,1],
    [0.800000011920929,0.800000011920929,1,1],
    [1,1,1,1],[1,1,1,1],
    [1,0.800000011920929,0.800000011920929,1],
    [1,0.600000023841858,0.600000023841858,1],
    [1,0.400000005960465,0.400000005960465,1],
    [1,0.200000002980232,0.200000002980232,1],
    [1,0,0,1],[1,0.100000001490116,0,1],
    [1,0.200000002980232,0,1],
    [1,0.300000011920929,0,1],
    [1,0.400000005960465,0,1],
    [1,0.500000000000000,0,1],
    [1,0.600000023841858,0,1],
    [1,0.699999988079071,0,1],
    [1,0.800000011920929,0,1],
    [1,0.899999976158142,0,1],
    [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,1],
    [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,1],
    [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,1],
    [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,1],
    [1,1,0,1]]
    )
        return ListedColormap(CC)
    else:
        return 'cividis'

# read incompressbile flow data from files
def read_inc_data_from_csv(fname):
    try:
        with open(fname) as f:
            data_iter = csv.reader(f,delimiter=",")
            data = [row[:] for row in data_iter] # skip first iteration due to headers
        data = np.asarray(data[1:],dtype='float64')
        pid = data[:,0]     # id
        X = data[:,1]       # x-coordinate
        Y = data[:,2]       # y-coordinate
        p = data[:,3]       # pressure
        u = data[:,4]      # momentum in x
        v = data[:,5]      # momentum in y 
        return [pid,X,Y,p,u,v]
    except IOError:   
        print("File not accessible")     

# returns incompressible flow data as full order arrays
def get_inc_flow_data(dir_name,delay=2000,subsampling=2,max_size=500):
    file_list = os.listdir(dir_name)
    files = len(file_list) - delay
    filenames = "restart_flow_"

    try:
        with open(dir_name+file_list[0]) as f:
            data_iter = csv.reader(f,delimiter=",")
            N = sum(1 for _ in data_iter) - 1
    except IOError:
        print("Files not accessible")
    
    # compute array size
    if files > max_size:
        if (files)//subsampling > max_size:
            array_size = max_size
        else:
            array_size = files//subsampling
    else:
        array_size = files

    # initializing data matrix
    p_arr = np.empty([N,array_size])       # pressure
    u_arr = np.empty([N,array_size])       # velocity in x
    v_arr = np.empty([N,array_size])       # velocity in y

    for i in tqdm(range(array_size)):  
        num = i * subsampling + delay
        fname = dir_name + filenames + "{:05d}".format(num) + ".csv"
        # reading file routine
        [pid,X,Y,p,u,v] = read_inc_data_from_csv(fname)
        p_arr[:,i] = p
        u_arr[:,i] = u
        v_arr[:,i] = v

    return {"X":X,"Y":Y,"u":u_arr,"v":v_arr,"p":p_arr}


      