#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:29:18 2024

@author: rileywilde
"""

from imports import *


# %% GET RID OF THIS PRIOR TO SHARING:::
    
USE_SSH = False


scan_num = 1


slicepath = './amaazescans/CT_'+str(scan_num)+'/Slices/AMAAZE_'+str(scan_num)+' Y Slices/'


if USE_SSH: 
    print('warning: enter your SSH details in line 25')
    ssh = createSSHClient('server.adress',22,'username', 'password')
    scp = SCPClient(ssh.get_transport())

#show identified corners:
imnum2get = str(2500)
if len(imnum2get)<4:
    for j in range(4-len(imnum2get)):
        imnum2get = '0'+imnum2get
    



im2get = 'AMAAZE_'+str(scan_num)+' Y_'+imnum2get+'.tif'
    
if USE_SSH:
    scp.get(os.path.join(slicepath,im2get))
    
I = plt.imread(im2get)
    

# %%

ang2rot = 151.5 #degrees
rowrng = [0,3001]
colrng = [0,2747]

#imdisp = rotate(rotate(plt.imread(im2get),ang2rot)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]],-angi)
imdisp = rotate(plt.imread(im2get),ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]].T
    
    
plt.imshow(imdisp)