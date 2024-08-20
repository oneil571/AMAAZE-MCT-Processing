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

import timeit
start = timeit.default_timer()


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


PADDING = opt.padding
folder = opt.folder
isolevel = opt.iso
controls_fname ='./controls.txt'
calibration_fname = './Calibration Report [Calibration].rtf'



for qq in [1]:



    os.chdir(folder)

    #have this in the CT_2 folder...
    scan_num = os.getcwd().split('/')[-1].split('_')[-1]

    slicepath = './Slices/AMAAZE_'+scan_num+' Y Slices'

    csvpath = './CT'+scan_num+'.csv' #results of split.py

    calibration_fname = calibration_fname


    outpath = './Meshes'
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



    rowrng1 = saveddata['rowrng']
    colrng1 = saveddata['colrng']
    ang2rot = saveddata['ang']
    origsz = saveddata['origsz']
    rem = saveddata['remainder']


    #num_cores = multiprocessing.cpu_count()
    #Parallel(n_jobs=4)(delayed(subprocess)(fnames, sd,outpath,slicepath, dx, infoi) for infoi in info)
                                        #^function #inputs to func
    for infoi in info:
        fname = infoi[0]
        zrng = infoi[1:3] #images to sift through

        #things for inner image rotation:
        rowrng2 = infoi[3:5]
        colrng2 = infoi[5:7]
        ang2rot2 = infoi[7]

        print('now starting ',fname)
        IMAGES = []

        for i in range(zrng[0],min(zrng[1],len(fnames))):
            #print((i - zrng[0])/(zrng[1]-zrng[0]))
            im = rotate(io.imread(os.path.join(slicepath,fnames[i])), ang2rot, preserve_range=True)
            #DO NOT USE CV.IMREAD - IT WILL NOT RETURN THE REAL VALUES!!!!!
            im = rotate( im[rowrng1[0]:rowrng1[1],colrng1[0]:colrng1[1]].copy(), ang2rot2,preserve_range=True )

            if i ==zrng[0]:
                rowrng2[0] = max(rowrng2[0] -PADDING,0)
                colrng2[0] = max(colrng2[0] -PADDING,0)
                rowrng2[1] = min(rowrng2[1] +PADDING,im.shape[0])
                colrng2[1] = min(colrng2[1] +PADDING,im.shape[1])

            IMAGES.append( im[rowrng2[0]:rowrng2[1],colrng2[0]:colrng2[1]] )

        IMAGES = np.array(IMAGES).transpose(2,1,0)
        overview = dicom.bone_overview(IMAGES)

        plt.imsave(os.path.join(outpath,fname+'.png'),overview, cmap='gray')
        np.savez_compressed(os.path.join(outpath,fname),I=IMAGES, dx=dx,dz=dx)
        print('finished ',fname, ' size: ', IMAGES.shape)

    #step 3: surface
    #print('reminder: run DICOM surfacing')

    IMAGES = []
    stop = timeit.default_timer()
    print('subvol extraction runtime a:', stop - start)

    print('starting DICOM surfacing')
    dicom.surface_bones_parallel(outpath, iso=isolevel, write_gif=False)

    #surface w. marching cubes

    #save voxels & triangulation to outpath/

'''
####################################################################### end main
'''

