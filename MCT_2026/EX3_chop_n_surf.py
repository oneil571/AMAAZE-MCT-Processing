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
from npy_append_array import NpyAppendArray
import timeit

'''
Hopefully this is the last update to the surfacing workflow ever. ever. ever.
I'm making it so that it loops only once through reading the images.
Writes much more though.
Will this be faster than
'''



def parse_option():
    parser = argparse.ArgumentParser('inputs')
    parser.add_argument('--folder', type=str, 
                        help='what scan are we doing? enter folder path.')  
    parser.add_argument('--iso', type=int, 
                        help='isolevel for surfacing')  
    parser.add_argument('--meshsubfolder', type=str,  default ='Meshes',
                        help='where are the meshes to go?')  
    parser.add_argument('--slicepath', type=str, default='Slices',
                        help='path to slices')
    parser.add_argument('--transpose', type=int, default=0,
                        help='0 if not transposed')
    parser.add_argument('--res', type=str, default='auto',
                        help='input number or leave')
    parser.add_argument('--calibration_name', type=str, default='controls.txt',
                        help='o.g. controls file in folder')
    parser.add_argument('--controls_name', type=str, default='Calibration Report [Calibration].rtf',
                        help='if an NSI scan, input this with --res auto to automatically extract resolution')
    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_option()

    
    start = timeit.default_timer()


    PADDING = 50

    folder = opt.folder
    isolevel = opt.iso

    controls_fname =opt.controls
    calibration_fname = opt.calibration_name

    outpath = opt.meshsubfolder

    for qq in [1]:

        os.chdir(folder)

        #have this in the CT_2 folder...
        scan_num = os.getcwd().split(os.sep)[-1].split('_')[-1]

        slicepath = opt.slicepath

        csvpath = 'CT'+scan_num+'.csv' #results of split.py

        calibration_fname = calibration_fname



        if os.path.exists(outpath)==False:
            os.mkdir(outpath)

        #controls:
        saveddata = np.load('ct'+str(scan_num)+'_new.npz')
        #vol = saveddata['vol']

        fnames = np.sort(os.listdir(slicepath))


        # %% STEP 0: find voxel size.
        if opt.res=='auto':
            dx = read_voxel_size(calibration_fname) #moved this to imports func
        else:
            dx = float(opt.res)

        # %% step 2: split into voxelized grids and save them

        info = pd.read_csv(csvpath,header=None).to_numpy()
        print('WARNING: REMOVING SPACES FROM FILE NAMES')
        info[:,0]=np.char.replace(info[:,0].astype(str), " ", "")

        tier_ranges,tier_ids = np.unique(info[:,1:3].astype(int),axis=0,return_inverse=True)



        rowrng1 = saveddata['rowrng']
        colrng1 = saveddata['colrng']
        ang2rot = saveddata['ang']
        origsz = saveddata['origsz']
        rem = saveddata['remainder']


        def subproc(i,j,infot,zrng,im):
            infoj = infot[j,:]
            fname = infoj[0]+'.npy'
            rowrng2 = infoj[3:5]
            colrng2 = infoj[5:7]

            with NpyAppendArray(os.path.join(outpath,fname), delete_if_exists=(i==zrng[0])) as npaa:
                npaa.append( (im[rowrng2[0]:rowrng2[1],colrng2[0]:colrng2[1]].T)[None,:,:] )
            return 


        num_cores = int(multiprocessing.cpu_count()*.85)
        #Parallel(n_jobs=4)(delayed(subprocess)(fnames, sd,outpath,slicepath, dx, infoi) for infoi in info)
                                            #^function #inputs to func

        for t in range(tier_ids.max()+1):

            infot = info[tier_ids==t,:]
            zrng = tier_ranges[t]

            ang2rot2 = infot[0,7] #these are all identical for a given tier

            for i in range(zrng[0],min(zrng[1],len(fnames))):
                
                im = rotate(io.imread(os.path.join(slicepath,fnames[i])), ang2rot, preserve_range=True)

                im = rotate( im[rowrng1[0]:rowrng1[1],colrng1[0]:colrng1[1]].copy(), ang2rot2,preserve_range=True )

                if i==zrng[0]:
                    infot[:,3] = np.maximum(infot[:,3] -PADDING,0)
                    infot[:,5] = np.maximum(infot[:,5] -PADDING,0)
                    infot[:,4] = np.minimum(infot[:,4] +PADDING,im.shape[0])
                    infot[:,6] = np.minimum(infot[:,6] +PADDING,im.shape[1])

                
                Parallel(n_jobs=num_cores)(delayed(subproc)(i,j,infot,zrng,im) for j in range(infot.shape[0]))

                '''
                for j in range(infot.shape[0]):
                    infoj = infot[j,:]
                    fname = infoj[0]+'.npy'
                    rowrng2 = infoj[3:5]
                    colrng2 = infoj[5:7]

                    with NpyAppendArray(os.path.join(outpath,fname), delete_if_exists=(i==zrng[0])) as npaa:
                        npaa.append( (im[rowrng2[0]:rowrng2[1],colrng2[0]:colrng2[1]].T)[None,:,:] )
                '''


    multiprocessing.active_children()  # trigger cleanup
    os.sync()  # force write buffers to disk
    time.sleep(2)          

    #surface + dicom overview
    for i in range(info.shape[0]):
        fname = os.path.join(outpath,info[i,0])
        IMAGES = np.load(fname+'.npy', mmap_mode="r")
        overview = dicom.bone_overview(IMAGES)
        plt.imsave(os.path.join(fname+'.png'),overview, cmap='gray')

        np.savez_compressed(fname,I=IMAGES, dx=dx,dz=dx)#automatically adds .npz
        #os.remove(fname+'.npy')
        #print('finished ',fname, ' size: ', IMAGES.shape)

    del IMAGES


    stop = timeit.default_timer()
    print('subvol extraction runtime for' + opt.folder+':', stop - start)

    #print('starting DICOM surfacing')
    tsurf0  = timeit.default_timer()
    dicom.surface_bones_parallel(outpath, iso=isolevel, write_gif=False,ncores=min(20,multiprocessing.cpu_count()))
    tsurf1  = timeit.default_timer()
    print('subvol extraction runtime for' + opt.folder+':', tsurf1 - tsurf0)

    with open("EX3_RUNTIMES.txt", "a") as f:
        f.write(opt.folder+' B EXTRACT '+str(stop - start)+'\n')
        f.write(opt.folder+' B SURF '+str(tsurf1 - tsurf0)+'\n')

        #surface w. marching cubes

        #save voxels & triangulation to outpath/

    '''
    ####################################################################### end main
    '''

def surfacing_subproc(filename,directory,iso_level,write_gif=False):
    
    try: 
        print('Loading '+filename+'...')
        M = np.load(os.path.join(directory,filename))
        #I = M['I']; dx = M['dx']; dz = M['dz']
        I = M; dx = 0.079; dz = 0.079
        
        #Rescale image to account for different dx/dz dimensions
        J = rescale(I.astype(float),(dz/dx,1,1),mode='constant')
    except Exception as error:
        print('LOADING ERROR with ', filename, ': ', error)

    try:
        verts,faces,normals,values = tm.marching_cubes(J,iso_level)
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



def surface_bones_parallel(directory, iso=2500, write_gif=False,error_fname='surfacing_errors.csv',ncores='all'):
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
    
        
    if isinstance(ncores,int):
        num_cores = ncores
    else: 
        num_cores =multiprocessing.cpu_count()
        
    errs = Parallel(n_jobs=num_cores)(delayed(surfacing_subproc)(f,directory,iso,write_gif) for f in fnames)
    
    errs = np.array(errs)
    errs = errs[errs!='0']
    
    if len(errs)==0:
        print('no errors, not saving an error csv.')
    else:
        print('there were ' + str(len(errs)) +' errors. saving CSV to ',error_fname)              
        pd.DataFrame(errs).to_csv(error_fname,header=False, index=False)