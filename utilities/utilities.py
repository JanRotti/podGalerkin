# dependencies
import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# reading simulation data from csv files
def read_csv_data(dataDirectory,delay=1000,subsampling=1,maxSize=500,dataList=[]):
    ## Variables
    # file_name_structure   -> naming scheme of csv files
    # delay                 -> first file to read in
    # subsampling           -> spacing between files to read
    # maxSize               -> maximum time dimension for array
    # dataList              -> specify data headers to read
    
    # reading simulation data from csv files
    fileList = os.listdir(dataDirectory)
    
    # extract iteration numbers
    numList = list(map(lambda sub:int(''.join([i for i in sub if i.isnumeric()])), fileList))
    
    # create sorted argument list
    indexSort = np.argsort(np.array(numList))
    
    # filelist length
    fileNumber = len(fileList) 
    if fileNumber<=0:
        raise ValueError("Invalid delay or empty directory!")

    # get number of data entries
    try:
        with open(dataDirectory+fileList[0]) as f:
            dataIterator = csv.reader(f,delimiter=",")
            N = sum(1 for _ in dataIterator) - 1 # number of data entries excluding headers
            f.close()
    except IOError:
        print("File not accessible")

    # get headers
    try:
        with open(dataDirectory+fileList[0]) as f:
            dataIterator = csv.reader(f,delimiter=",")
            for row in dataIterator:
                headers = row
                break
            f.close()
    except IOError:
        print("File not accessible") 

    # compute array size for initialization
    if (fileNumber-delay)//subsampling > maxSize:
        array_size = maxSize
    else:
        array_size = (fileNumber-delay)//subsampling

    # initialize data dictionary
    data = {}
    for header in headers:
        if len(dataList)==0:
            data[header] = np.empty((N,array_size))
        elif header in dataList:
            data[header] = np.empty((N,array_size))

    # reading file data from csv files
    for i in tqdm(range(array_size)):
        fname = dataDirectory + fileList[indexSort[i*subsampling+delay]]
        # reading routine
        with open(fname) as f:
            dataIterator = csv.reader(f,delimiter=",")
            for j,row in enumerate(dataIterator):
                if j!=0:
                    for k,header in enumerate(headers):
                        if len(dataList)==0:
                            data[header][j-1,i] = row[k]
                        elif header in dataList:
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
    # number of subplot rows
    rows = int(num//2)
    # create subplot structure
    fig, ax = plt.subplots(rows,1,figsize=(10,2*8))
    # reduced subplot spacing
    fig.tight_layout(pad=3.0)
    # iterate rows
    for i in range(rows):
        ax[i].plot(t,coeffs[2*i,:])
        ax[i].plot(t,coeffs[2*i + 1,:])
        ax[i].title.set_text("Eigenflow " + str(2*i+1) + " and " + str(2*i+2))












