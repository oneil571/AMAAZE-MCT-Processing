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
    parser.add_argument('--meshsubfolder', type=str,  default ='./Meshes',
                        help='where are the meshes to go?')  

    
    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_option()

    
    start = timeit.default_timer()


    PADDING = 50

    folder = opt.folder
    isolevel = opt.iso

    os.chdir(opt.folder)

    controls_fname ='./controls.txt'
    calibration_fname = './Calibration Report [Calibration].rtf'

    outpath = opt.meshsubfolder
    
    print('starting DICOM surfacing')
    dicom.surface_bones_parallel(outpath, iso=isolevel, write_gif=False,ncores=min(20,multiprocessing.cpu_count()))

        #surface w. marching cubes

        #save voxels & triangulation to outpath/

    '''
    ####################################################################### end main
    '''

