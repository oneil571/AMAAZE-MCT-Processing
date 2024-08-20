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
This is faster!!
'''



def parse_option():
    parser = argparse.ArgumentParser('inputs')
    parser.add_argument('--folder', type=str, 
                        help='what scan are we doing? enter folder path.')  
    parser.add_argument('--iso', type=int, 
                        help='isolevel for surfacing')  
    parser.add_argument('--meshsubfolder', type=str,  default ='./Meshes',
                        help='where are the meshes to go?')  
    parser.add_argument('--nworkers', type=int,  default =0,
                        help='how many cores to use for parallelization. 0 = use almost all (85%)')
    parser.add_argument('--padding',type=int,default=50,
                        help = 'how much to pad extracted bounding boxes')
    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_option()

    
    start = timeit.default_timer()


    PADDING = opt.padding
    folder = opt.folder
    isolevel = opt.iso

    controls_fname ='./controls.txt'
    calibration_fname = './Calibration Report [Calibration].rtf'

    outpath = opt.meshsubfolder

    for qq in [1]:

        os.chdir(folder)

        #have this in the CT_2 folder...
        scan_num = os.getcwd().split('/')[-1].split('_')[-1]

        slicepath = './Slices/AMAAZE_'+scan_num+' Y Slices'

        csvpath = './CT'+scan_num+'.csv' #results of split.py

        calibration_fname = calibration_fname



        if os.path.exists(outpath)==False:
            os.mkdir(outpath)

        #controls:
        saveddata = np.load('./ct'+str(scan_num)+'_new.npz')
        #vol = saveddata['vol']

        fnames = np.sort(os.listdir(slicepath))


        # %% STEP 0: find voxel size.

        dx = read_voxel_size(calibration_fname) #moved this to imports func


        # %% step 2: split into voxelized grids and save them

        info = pd.read_csv(csvpath,header=None).to_numpy()

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

        if opt.nworkers==0:
            num_cores = int(multiprocessing.cpu_count()*.85)
        else:
            num_cores = opt.nworkers
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
                    

    #surface + dicom overview
    for i in range(info.shape[0]):
        fname = os.path.join(outpath,info[i,0])
        IMAGES = np.load(fname+'.npy', mmap_mode="r")
        overview = dicom.bone_overview(IMAGES)
        plt.imsave(os.path.join(fname+'.png'),overview, cmap='gray')

        np.savez_compressed(fname,I=IMAGES, dx=dx,dz=dx)#automatically adds .npz
        os.remove(fname+'.npy')
        print('finished ',fname, ' size: ', IMAGES.shape)

    del IMAGES


    stop = timeit.default_timer()
    print('subvol extraction runtime b: ', stop - start)

    print('starting DICOM surfacing')
    dicom.surface_bones_parallel(outpath, iso=isolevel, write_gif=False,ncores=min(20,multiprocessing.cpu_count()))

        #surface w. marching cubes

        #save voxels & triangulation to outpath/

    '''
    ####################################################################### end main
    '''

