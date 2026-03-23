#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:29:18 2024

@author: rileywilde
"""

from imports import *

import argparse




ct2='~/sofia_scan/CT_2/Slices/TeethScan2_79microns_Test_3_dddd.tif'
#ct1='~/sofia_scan/CT_1/Slices/TeethScan_1_dddd.tif'

parser = argparse.ArgumentParser()
parser.add_argument("--slicepath", type=str, default=ct2) #'Slices/TeethScan2_79microns Test 3_0351.tif'
parser.add_argument("--USE_SSH", type=int, default=1)
parser.add_argument("--transpose", type=int, default=0)

args = parser.parse_args()


ang2rot = -6.5
rowrng = [300,1700] #[250,1650] #
colrng = [250,1550] #[150,1500] #



slicenum=100


# %% GET RID OF THIS PRIOR TO SHARING:::
    




if args.USE_SSH: 
    print('warning: REMOVE your SSH details in line 25')
    ssh = createSSHClient('calder.math.umn.edu',22,'riley', 'Soybean#!444')
    scp = SCPClient(ssh.get_transport())

#show identified corners:
imnum2get = f"{slicenum:04d}"

slicepath = args.slicepath.replace('_dddd','_'+imnum2get)
    
if args.USE_SSH:
    scp.get(slicepath)
    
I = plt.imread(slicepath.split('/')[-1])
    

# %%



#imdisp = rotate(rotate(plt.imread(im2get),ang2rot)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]],-angi)

if args.transpose:
    imdisp = rotate(I,ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]].T
else:
    imdisp = rotate(I,ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]]
    

plt.figure()
plt.imshow(imdisp)
plt.show(block=False)