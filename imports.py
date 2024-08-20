#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:59:24 2023

@author: rileywilde
"""


#I just found that we can dump all the imports in another file!!!! This is so cool. 
# Just do: 
#from imports import *
#everything imported here is in turn imported in the called function. 
 
import numpy as np
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

import cv2 as cv #one of my least favorite python packages
import skimage.io as io
import argparse

from numpy import concatenate as cat
from skimage import measure #do we ever even use this?
from skimage.filters import gaussian
from skimage.transform import rotate
from scipy.ndimage import convolve
from scipy.signal import find_peaks

#from scipy.ndimage.measurements import label
from scipy.ndimage import label
from matplotlib import patches

from ast import literal_eval #one of my new favorite functions


##DELETE THIS PRIOR TO SHARING
import paramiko
from scp import SCPClient
import torch
import torch.nn as nn

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def draw_boxes(im,row,col,title=''):
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.imshow(im)
    
    for j in range(row.shape[0]):
        rect = patches.Rectangle((col[j,0],row[j,0]), col[j,1]-col[j,0], row[j,1]-row[j,0],  linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(title)
    plt.show()


#%% for EX3:
def read_voxel_size(calibration_fname):
    str2lookfor = 'Optimum voxel size' #first chunk of line with microns measurement 
                
    with open(calibration_fname, 'r') as file:
        text = file.read()
            
    q = text.split('\n')
        
    for s in q:
        if s[0:len(str2lookfor)]==str2lookfor:
            break
        
    if s[0:len(str2lookfor)]!=str2lookfor:
        print('string to look for not found... uh oh. please change it to continue.')
        raise SystemExit(0)
        
    words = s.split(' ')
        
    for i in range(len(words)):
        if words[i] == 'microns':
            i = i-1
            break
        
    dx = float(words[i])*(10**-3) #microns to mm
    return dx


#%% for EX2:
def ang_rot(im,plot=True,title=''):
    
    incr = .1
    stop = 1
    angs = cat((np.arange(-10,-stop,incr), np.arange(stop,10,incr)))

    icol,irow = np.meshgrid(np.arange(im.shape[0]),np.arange(im.shape[1]))
    mask = (icol-im.shape[1]/2)**2 + (irow-im.shape[0]/2)**2 >= (np.min(im.shape)/2.25)**2

    notmask=mask!=1

    nim = im.copy()-im.min()
    nim = nim/nim.max()

    nim[mask] = 0 
    nim = gaussian(nim, sigma = 3)
    '''
    astart = -45
    nim = rotate(nim, astart, preserve_range=True)
    '''
    f1 = np.array([[-1,1],[-1,1]])
    f2 = np.array([[1,-1],[-1,1]])

    vh_sums = []
    di_sums = []
    #plt.figure()
    for a in angs:
        
        imi = rotate(rotate(nim, a+27.2, preserve_range=True, order = 1), -27.2,preserve_range=True, order = 1)

        vert = convolve(imi,f1.T)
        hor = convolve(imi,f1)
        diag = convolve(imi,f2)
        '''
        plt.imshow( cat( (cat((2*imi,np.abs(vert)),1),cat((np.abs(hor),np.abs(diag)),1)),0), vmin = 0, vmax = 2)
        plt.show()'''
        #plt.plot()
        #time.sleep(.1)

        vh_sum = (np.sum(vert[notmask]**2)+np.sum(hor[notmask]**2))
        di_sum = np.sum(diag[notmask]**2)

        vh_sums.append(vh_sum)
        di_sums.append(di_sum)

    vh_sums = np.array(vh_sums)
    di_sums = np.array(di_sums)
    
    
    x = angs
    y = vh_sums/di_sums
    
    coeff = np.polyfit(angs,y,2)
    
    yfit = coeff[0]*x**2+coeff[1]*x+coeff[2]

    ang_out = -coeff[1]/(2*coeff[0])#angs[yfit.argmax()]

    if plot:
        plt.figure()
        plt.plot(angs, y)
        plt.plot(angs,yfit)
        plt.title(title)
        plt.xlabel('rotation')
        plt.ylabel('detail signal')
        
    return ang_out

def auto_seg(signal,n_seg):
    
    t = signal.max()
    
    m = .99

    n_groups=0
    while n_groups!=n_seg:
        
        t = m*t
        
        I = signal>t
        L = label(I)
        n_groups = L[0].max()
        
    return I

                    
    
    #for i in range(len(trng)):
    return   
    
def read_hyperparameters(fnumber):
    fname = 'ct'+str(fnumber)+'_params.txt'

    with open(fname, 'r') as file:
        text = file.read()
            
    q = text.split('\n')
    
    vout = []
    for i in range(8): #can include additional vars after
        x = q[i]
        vout.append(literal_eval(x.split(' ')[1]))
        
    return vout
    

def get_parameter(fnumber,paramname):
    fname = 'ct'+str(fnumber)+'_params.txt'

    with open(fname, 'r') as file:
        text = file.read()
            
    q = text.split('\n')
    
    output = None
    for i in range(len(q)): #can include additional vars after
        x = q[i]
        line = x.split(' ')
        
        if (line[0]==paramname) | (line[0]==paramname+'='):
            output = literal_eval(line[1])
    return output


def get_vertseg_if_there(fnumber):
    str2lookfor = 'VERTSEG' #first chunk of line with microns measurement 
    
    fname = 'ct'+str(fnumber)+'_params.txt'
            
    with open(fname, 'r') as file:
        text = file.read()
            
    q = text.split('\n')
        
    for s in q:
        if s[0:len(str2lookfor)]==str2lookfor:
            break
        
    if s[0:len(str2lookfor)]!=str2lookfor:
        print('no vertical segmentation found. Please do this time.')
        return None
    
    
    words = s.split(' ')
    
    if len(words)<2:
        print('no vertical segmentation found. Please do this time.')
        return None
    
    if len(words[1])<1:
        print('no vertical segmentation found. Please verify file or do manually.')
        return None
        
    return literal_eval(words[1])


def update_param(fnumber,str2lookfor,ex):
    #replace entry if exists,
    #else append entry to end of file
    
    exstr = str(ex)
    eout = ''
    for c in exstr:#remove spaces
        if c!=' ':
            eout=eout+c
    exstr = eout
        
    
    fname = 'ct'+str(fnumber)+'_params.txt'
            
    with open(fname, 'r') as file:
        text = file.read()
            
    lst = text.split(str2lookfor)
    
    if len(lst)==2: # str exists, replace it
        pre,post = lst
    
        rbd = post.find('\n') #interestingly, \n is regarded as a single character!
        
        
        if post[rbd]=='\n': #i.e. actually found
            tail = post[rbd:]
        else:
            tail = '' #i.e. end of file
        
        text = pre+str2lookfor+'= '+exstr+tail
    else: 
        text = text+'\n'+str2lookfor+exstr
        

    with open(fname, 'w') as file:
        file.write(text)



def id_cardboard(si,frac_thresh = .75, t1=37000,t2=39000,t3=41000):
    x= np.sum((si>t1)*(si<t2),0).astype(int) - np.sum(si>t3,0).astype(int)
    #x[x<0] = 0
    t4 = np.floor(frac_thresh*x.max())
    x[x<t4] = 0
    
    return x
    
    
    


from scipy import ndimage


def im_grad(im):
    sx = ndimage.sobel(im,axis=0,mode='constant')
    sy = ndimage.sobel(im,axis=1,mode='constant')
    
    grad = (sx**2 + sy**2)**.5
    return grad
    

def autorot2(si,frac_thresh = .75, t1=37000, t2=39000, t3=41000,title=''):
    # t4 ~~~ width of si.... hence frac_thresh
    
    x = np.sum((si>t1)*(si<t2),0).astype(int) - np.sum(si>t3,0).astype(int)
    
    x[x<0]= 0
    
    t4 = np.floor(frac_thresh*x.max())
    
    x_thresh = (x>t4).astype(float)
    
    a = ang_rot(x_thresh,title=title)
    xx = rotate(x,a,preserve_range=True)
    
    xx[xx<t4] = 0
    
    
    return a,xx
    

def extract_max_subgraph(pts,tri):
    
    E = np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]) , 0 )
    
    #order so second entry > first:
    I = E[:,0]>E[:,1]
    E[I,:] = E[I,-1::-1]
    
    #no self-loops:
    E = E[ E[:,0]!=E[:,1],:]
    
    #so we can extract unique entries for a graph!
    E = np.unique(E,axis=0)
    E = np.concatenate( (E,E[:,-1::-1]),0) #so graph works...
    
    
    n_pts = pts.shape[0]
    
    A = csr_matrix((np.ones(E.shape[0]), (E[:,0],E[:,1])), shape=(n_pts, n_pts))
    
    nseg, labs = connected_components(csgraph=A, directed=False, return_labels=True)
    
    counts = []
    for i in range(nseg):
        counts.append(np.sum(labs==i))
    counts = np.array(counts)
    

    pt_ind2keep =  np.where(labs == counts.argmax())[0]
    
    newind = np.arange(pt_ind2keep.shape[0])
    
    old2new = -1*np.ones(n_pts)
    old2new[pt_ind2keep] = newind 
    
    newtri = old2new[tri]
    newtri = newtri[ np.sum(newtri<0,1) ==0, :]
    
    
    newpts = pts[pt_ind2keep,:]
    
    return newpts, newtri




# %% NEW DICOM:::

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import scipy.ndimage as ndimage
import scipy.stats as stats
from skimage import measure
from skimage.transform import rescale
import os, multiprocessing
import sys
from joblib import Parallel, delayed
#from . import trimesh as tm
from amaazetools.dicom import *
import pandas as pd
import amaazetools.trimesh as tm

def surfacing_subproc(filename,directory,iso_level,write_gif=False):
    
    print('Loading '+filename+'...')
    M = np.load(os.path.join(directory,filename))
    I = M['I']; dx = M['dx']; dz = M['dz']
    
    #Rescale image to account for different dx/dz dimensions
    J = rescale(I.astype(float),(dz/dx,1,1),mode='constant')
    
    try: 
        verts,faces,normals,values = measure.marching_cubes(J,iso_level)
        mesh = tm.mesh(dx*verts,faces) #Multiplication by dx fixes units
    
        #Reverse orientation of triangles (marching_cubes returns inward normals)
        mesh.flip_normals()
    
        #Write to ply file
        mesh_filename = os.path.join(directory,filename[:-4]+'_iso%d'%iso_level)
        print('Saving mesh to '+mesh_filename+'...')
        mesh.to_ply(mesh_filename+'.ply')
    
        if write_gif:
            mesh.to_gif(mesh_filename+'.gif')
        return '0'
    except Exception as error:
        print('surfacing error with ', filename, ': ', error)
        return filename



def surface_bones_parallel(directory, iso=2500, write_gif=False,error_fname='./surfacing_errors.csv',ncores='all'):
    """ parallelized implementation of surface_bones with also surfacing error support.
        Processes all npz files in directory creating surface and saving to a ply file.

        Parameters
        ----------
        directory : str
            Directory to work within.
        iso : float (optional), default is 2500
            Iso level to be used for surfacing.
        write_gif : bool (optional), default=False
            Whether to output rotating gifs for each object. Requires mayavi, which can be hard to install.
        error_fname

        Returns
        -------
        None
    """
    
    ddd = os.listdir(directory)
    
    fnames = []
    for f in ddd:
        if f.endswith('.npz'):
            fnames.append(f)
    
    num_cores =multiprocessing.cpu_count()
    errs = Parallel(n_jobs=num_cores)(delayed(surfacing_subproc)(f,directory,iso,write_gif) for f in fnames)
    
    errs = np.array(errs)
    errs = errs[errs!='0']
    
    if len(errs)==0:
        print('no errors, not saving an error csv.')
    else:
        print('there were ' + str(len(errs)) +' errors. saving CSV to ',error_fname)              
        pd.DataFrame(errs).to_csv(error_fname,header=False, index=False)



'''
class triplegauss(nn.Module):
    def __init__(self):
        super(triplegauss, self).__init__()
        
        self.mu = nn.Parameter(torch.rand(1,3))
        self.si = nn.Parameter(torch.rand(1,3))
        self.ht = nn.Parameter(torch.rand(1,3))
        
        
    def forward(self,x):
        
        if len(x.shape)==1:
            x = x[:,None]
    
        y = torch.sum(self.ht*torch.exp((-1*(x-self.mu)**2)/(self.si**2)),1)
        
        return y
    
    def g(self,x,k):
        if len(x.shape)==2:
            x=x[:,0]
        y = self.ht[0,k]*torch.exp((-1*(x-self.mu[0,k])**2)/(self.si[0,k]**2))
        return y


def t1t2t3(vol):
    
    vmin = vol.min()
    nvol = vol-vmin
    nvmax = nvol.max()
    nvol = nvol/nvmax

    y,x = np.histogram(nvol.flatten(),bins=500)
    
    
    x = (x[0:-1] + x[1:])/2
    
    
    #normalize counts so tr goes faster:
    y = y-y.min()
    y = y/y.max()
    
    x = torch.tensor(x); y = torch.tensor(y)
    
    x = x[:,None]
    
    
    model = triplegauss()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.01)
    
    for i in range(10000):
        loss = torch.sum(torch.abs(model(x)-y))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model.eval()
    plt.figure();
    plt.plot(x,y); 
    plt.plot(x,model(x).detach())
    
    for k in range(0,3):
        plt.plot(x,model.g(x,k).detach())
'''
                                      

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    