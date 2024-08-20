#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:53:11 2023

@author: rileywilde
"""

import os 
import numpy as np
import cv2 as cv
from skimage import measure
from skimage.transform import rotate
import matplotlib.pyplot as plt
from ast import literal_eval

import skimage.io as io

import argparse

def parse_option():
    parser = argparse.ArgumentParser('inputs')
    parser.add_argument('--folder', type=str, default='',
                        help='what scan are we doing? enter folder path.')
    parser.add_argument('--zwindow', type=int, default=10,
                        help='how many slices to average in subsampling')  
    parser.add_argument('--controls_fname', type=str, default='./controls.txt',
                        help='txt file where the cropping information is kept.') 
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_option()

    os.chdir(opt.folder)

    scan_num = os.getcwd().split('/')[-1].split('_')[-1]
    slicepath = './Slices/AMAAZE_'+scan_num+' Y Slices'

    fnames = np.sort(os.listdir(slicepath)) 

    n_slices = len(fnames)

    #read cropping controls in from txt file:
    controls_fname = './controls.txt'
    
    with open(controls_fname, 'r') as file:
        text = file.read()
        
    q = text.split('\n')
    
    angstr = 'ang2rot';    rowstr = 'rowrng';    colstr = 'colrng';
    for s in q:
        if s[0:len(angstr)]==angstr:
            ang2rot = literal_eval(s.split(' ')[1])
        if s[0:len(rowstr)]==rowstr:
            rowrng = literal_eval(s.split(' ')[1])
        if s[0:len(colstr)]==colstr:
            colrng = literal_eval(s.split(' ')[1])

    zwindow = opt.zwindow
    subsampled = []
    for i in range(n_slices):
        #print(i/n_slices,fnames[i])
        im = io.imread(os.path.join(slicepath,fnames[i])).astype(int)
        #DO NOT USE CV.IMREAD - IT WILL NOT RETURN THE REAL VALUES!!!!! 
        #print(i/n_slices,fnames[i])#,im.max())
        if i%zwindow==0: #first
            imstack = im
        else: #middle:end
            imstack = imstack+im

        if i%zwindow==(zwindow-1): #last
            m = imstack.min()
            imstack = rotate(imstack,ang2rot,preserve_range=True,cval=m)
            print(i/n_slices,fnames[i],m)
            imstack = imstack[rowrng[0]:rowrng[1],colrng[0]:colrng[1]].copy()/zwindow
        
            imstack = cv.resize(imstack, (225,225)).copy()
        
            subsampled.append(imstack)

    rem = 0
    if i%zwindow!=(zwindow-1): #fix the end if 'last' cond didn't happen
        imstack = rotate(imstack,ang2rot,preserve_range=True,cval=imstack.min())
        imstack = imstack[rowrng[0]:rowrng[1],colrng[0]:colrng[1]].copy()/((i%zwindow) +1)

        imstack = cv.resize(imstack, (225,225)).copy()
        subsampled.append(imstack)
        rem = (i%zwindow) +1
    np.savez('ct'+scan_num+'_new.npz',vol = subsampled,rowrng=rowrng,colrng=colrng,ang=ang2rot,origsz =im.shape,remainder=rem)
#end main()
    
