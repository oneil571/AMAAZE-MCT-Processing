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

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


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
    
    
    
def subproc(fname,opt):
    dust_cutoff = 1000
    holetol = 0
    
    
    M = tm.load_ply(os.path.join(opt.meshsubfolder,fname))

    holes = M.detect_holes()

    ncomp,labs,counts = M.con_comp(returncounts=True)
    print(fname, str(len(counts))+ ' components')

    labels_in_consideration = np.where(counts>dust_cutoff)[0]
    if labels_in_consideration.shape[0]>0: #we have things that ain't dust
        counts_in_consideration = counts[labels_in_consideration]

        holes_in_each = []
        for i in labels_in_consideration:
            ex = labs ==i
            holes_in_each.append(holes[ex].sum())
            #Mi = M.extract_subtri(ex)
        holes_in_each = np.array(holes_in_each)

        l = holes_in_each<=holetol

        labels_holeless = labels_in_consideration[l]
        counts_holeless = counts_in_consideration[l]

        if len(counts_holeless)==0:
            M = M.extract_subtri(labs==counts.argmax())
        else:
            M = M.extract_subtri(labs==labels_holeless[counts_holeless.argmax()])        
    else: #it seems we only have dust
        print(fname+ ': dust cutoff failed, still extracting largest component')
        M = M.extract_subtri(labs==counts.argmax())

    M.to_ply(os.path.join(opt.newmeshsubfolder,fname))
    return 
    
    
    
    

        
    
    





def parse_option():
    parser = argparse.ArgumentParser('inputs')
    parser.add_argument('--folder', type=str, 
                        help='what scan are we doing? enter folder path.')  
    parser.add_argument('--meshsubfolder', type=str,  default ='./Meshes',
                        help='where are the meshes that need to be cleaned?')  
    parser.add_argument('--newmeshsubfolder', type=str,  default ='./Clean_Meshes',
                        help='where are the new meshes going?.') 
    
    
    return parser.parse_args()


def main():
    opt = parse_option()
    
    os.chdir(opt.folder)
    
    mesh_folder = opt.meshsubfolder
    out_folder = opt.newmeshsubfolder
    
    if os.path.exists(out_folder)==False:
        os.mkdir(out_folder)
    
    ddd = os.listdir(mesh_folder)
    
    fnames = []
    for f in ddd:
        if f.endswith('.ply'):
            fnames.append(f)


    
    num_cores =int(multiprocessing.cpu_count()/4)
    Parallel(n_jobs=num_cores)(delayed(subproc)(f,opt) for f in fnames)

    
if __name__ == '__main__':
    main()
            
       
   
