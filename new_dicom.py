
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