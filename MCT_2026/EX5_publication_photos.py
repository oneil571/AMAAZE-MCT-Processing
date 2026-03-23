#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:53:11 2023

@author: rileywilde
"""

from imports import *

from amaazetools import dicom

import multiprocessing
from joblib import Parallel, delayed

import amaazetools.trimesh as tm
 
from joblib import Parallel, delayed
import multiprocessing

from mayavi import mlab


mlab.options.offscreen = True
    
    
def subproc(fname,outdir):
    
    try:
        print(fname)
        pts,tri = tm.read_ply(fname)
        
        fname = fname.split('/')[-1]
        fname = fname.split('.')[0]

        
        pts = pts-pts.mean(0)
        
        
        vals,vecs = scipy.linalg.eig(pts.T@pts)
        ind = np.isreal(vals)
        vals = np.real(vals[ind])
        vecs = np.real(vecs[:,ind])
        
        
        a = np.arange(3);    
        reorder = np.array([a[(a!=vals.argmax()) * (a!=vals.argmin())].item(), vals.argmin(), vals.argmax()])
    
        vecs = vecs[:,reorder]
        #ensure orientation is fine:
        if np.linalg.det(vecs)==-1:
            vecs[:,-1] = -1*vecs[:,-1] 
            
        p2 = pts@vecs
        p2 = p2-p2.min(0)    
        
        #m2 = tm.mesh(p2,tri)
        
        cb = (201/255, 167/255, 126/255)
        #cb = (.8,.8,.8)
        
        mlab.figure()
        mfig = mlab.triangular_mesh(p2[:,0],p2[:,1],p2[:,2],tri,color = cb)
        mfig.scene.background = (0,0,0)
        #ax1 = mlab.axes( color=(1,1,1))#, nb_labels=0 )
        
        mlab.view(azimuth=90, elevation=90)
        mlab.savefig(filename=os.path.join(outdir,fname+'1.png'))
        
        mlab.view(azimuth=180, elevation=90)
        mlab.savefig(filename=os.path.join(outdir,fname+'2.png'))
        
        mlab.view(azimuth=270, elevation=90)
        mlab.savefig(filename=os.path.join(outdir,fname+'3.png'))
        
        mlab.view(azimuth=0, elevation=90)
        mlab.savefig(filename=os.path.join(outdir,fname+'4.png'))
        
        mlab.close()
    
    except Exception as error:
        print('surfacing error with ', fname, ': ', error)
        
    
    




for q in [1]:
    
    outdir = './scanphotos'
    if os.path.exists(outdir)==False:
        os.mkdir(outdir)
    
    for scan_num in range(1,11):
        fder = './CT_'+str(scan_num)
        ddd = os.listdir(fder)
        
        ddd = [os.path.join(fder,d) for d in ddd]
        if scan_num==1:
            paths = ddd
        else:
            paths = paths+ddd

    
    num_cores =multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(subproc)(f,outdir) for f in paths)


       
   
