# dependencies
import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# reading simulation data from csv files
def read_csv_data(data_dir,delay=1000,subsampling=1,max_size=500,data_list=[]):
    ## Variables
    # file_name_structure   -> naming scheme of csv files
    # delay                 -> first file to read in
    # subsampling           -> spacing between files to read
    # max_size              -> maximum time dimension for array
    # data_list             -> specify data headers to read
    
    # reading simulation data from csv files
    file_list = os.listdir(data_dir)
    num_files = len(file_list) # difference of delay and filelist length
    if num_files<=0:
        raise ValueError("Invalid delay or empty directory!")

    # get number of data entries
    try:
        with open(data_dir+file_list[0]) as f:
            data_iter = csv.reader(f,delimiter=",")
            N = sum(1 for _ in data_iter) - 1 # number of data entries excluding headers
            f.close()
    except IOError:
        print("File not accessible")

    # get headers
    try:
        with open(data_dir+file_list[0]) as f:
            data_iter = csv.reader(f,delimiter=",")
            for row in data_iter:
                headers = row
                break
            f.close()
    except IOError:
        print("File not accessible") 

    # compute array size for initialization
    if (num_files-delay)//subsampling > max_size:
        array_size = max_size
    else:
        array_size = (num_files-delay)//subsampling

    # initialize data dictionary
    data = {}
    for header in headers:
        if len(data_list)==0:
            data[header] = np.empty((N,array_size))
        elif header in data_list:
            data[header] = np.empty((N,array_size))

    # reading file data from csv files
    for i in tqdm(range(array_size)):
        fname = data_dir + file_list[i*subsampling+delay]
        # reading routine
        with open(fname) as f:
            data_iter = csv.reader(f,delimiter=",")
            for j,row in enumerate(data_iter):
                if j!=0:
                    for k,header in enumerate(headers):
                        if len(data_list)==0:
                            data[header][j-1,i] = row[k]
                        elif header in data_list:
                            data[header][j-1,i] = row[k]

    return data

# plotting cylinder data
def plot_cylinder_data(x,y,data,levels=100,cmap='cividis',ax=None,zoom=False,resolution=1000,cbar=True):
    # construct plotting axis if necessary
    if ax==None:
        if zoom:
            fig,ax = plt.subplots(1,1,figsize=(18,6))
        else:
            fig,ax = plt.subplots(1,1,figsize=(14,12))

    far = np.round(np.max(y),1)

    # plot central cylinder and farfield
    cylinder = plt.Circle((0,0),radius=0.5,color="black")
    farfield = plt.Circle((0,0),radius=far,color="magenta",fill=False)

    ax.add_patch(cylinder)
    ax.add_patch(farfield)

    # set axis limits
    if zoom:
        xlim_l = -far*0.2
        xlim_r = far*0.8
        ylim_r = far/5
        ylim_l = -far/5
    else:
        xlim_l = -far
        xlim_r = far
        ylim_r = far
        ylim_l = -far
        
    ax.set_xlim([xlim_l,xlim_r])
    ax.set_ylim([ylim_l,ylim_r])

    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # distance function
    def dist(xm,ym,x,y):
        return np.sqrt((x-xm)*(x-xm)+(y-ym)*(y-ym))

    # add equidistant grid points
    gridx = np.arange(xlim_l,xlim_r,(xlim_r-xlim_l)/resolution)
    gridy = np.arange(ylim_l,ylim_r,(ylim_r-ylim_l)/resolution)
    grid_x,grid_y = np.meshgrid(gridx, gridy)
    dx = (xlim_r - xlim_l)/resolution
    dy = (ylim_r - ylim_l)/resolution
    points = np.stack((x,y),axis=1)

    # interpolate data
    grid = griddata(points, data, (grid_x, grid_y))

    # set boundary conditions
    r = 0.5
    far = far
    if zoom:
        xm = int(resolution*0.2)
        ym = int(resolution/2)
    else:
        xm = int(resolution/2)
        ym = int(resolution/2)

    for i in range(xm-int(r/dx),xm+int(r/dx),1):
        for j in range(ym-int(r/dy),ym+int(r/dy),1):
            if dist(xm,ym,i,j)<(r/dx):
                grid[j,i] = 0

    for i in range(resolution):
        for j in range(resolution):
            if dist(xm,ym,i,j)>(far/dx):
                grid[i,j]=0

    cntr = ax.contourf(grid_x,grid_y,grid,levels,cmap=cmap)
    
    # add colorbar
    if cbar:
        plt.colorbar(cntr,ax=ax);

# pretty printing a string - ouput for a centered padded string with "--"
def print_padded(print_string):
    print(f'{print_string:{"-"}<80}')

# plotting pod mode activation in time
def plot_activations(coeffs,num,dt):
    t = np.linspace(0,(coeffs.shape[1]-1)*dt,coeffs.shape[1])

    rows = int(num//2)
    fig, ax = plt.subplots(rows,1,figsize=(10,2*8))
    fig.tight_layout(pad=3.0)
    for i in range(rows):
        ax[i].plot(t,coeffs[2*i,:])
        ax[i].plot(t,coeffs[2*i + 1,:])
        ax[i].title.set_text("Eigenflow " + str(2*i) + " and " + str(2*i+1))

def plot_group_activations(coeffs,num,dt):
    t = np.linspace(0,(coeffs.shape[1]-1)*dt,coeffs.shape[1])

    rows = int(num//2)
    fig, ax = plt.subplots(rows,1,figsize=(10,2*8))
    fig.tight_layout(pad=3.0)
    for i in range(rows):
        ax[i].plot(t,coeffs[2*i,:])
        ax[i].plot(t,coeffs[2*i + 1,:])
        ax[i].title.set_text("Eigenflow " + str(i) + " and " + str(i+1))










