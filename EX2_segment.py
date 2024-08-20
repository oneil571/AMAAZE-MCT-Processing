#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:53:20 2024

@author: rileywilde
"""

from imports import *


# %% GET RID OF THIS PRIOR TO SHARING:::

USE_SSH = False

if USE_SSH:
    print('please configure SSH and enter your password in line 18')
    ssh = createSSHClient('server.address',22,'username', 'password')
    scp = SCPClient(ssh.get_transport())

REDO_TIERS=True
REDO_ISO = False
PLOTTING = True
SCPPLOT = True

scan_num = 4#int(os.getcwd().split('/')[-1].split('_')[-1])



AUTO_ROT,AUTO_SEG,INPUT_ROWS,INPUT_COLS,TIER_THRESH,INVERT,THRESH0,THRESH1 = read_hyperparameters(scan_num)

THRESH0 = 6000
THRESH1 = 6000

slicepath = './amaazescans/CT_'+str(scan_num)+'/Slices/AMAAZE_'+str(scan_num)+' Y Slices/'


saveddata = np.load('./ct'+str(scan_num)+'_new.npz')
vol = saveddata['vol']
rowrng = saveddata['rowrng']
colrng = saveddata['colrng']
ang2rot = saveddata['ang']
origsz = saveddata['origsz']
rem = saveddata['remainder']

rowsz = rowrng[1]-rowrng[0]
colsz = colrng[1]-colrng[0]

if INVERT:
    vol = vol.max()-vol


x = pd.read_csv('./all_mct_scan.csv').fillna(0)
x = x.to_numpy()

x = x[x[:,0]==scan_num,1:].copy()

n_tiers = x[:,0].max()
dim1 = x[:,1].max()
dim2 = x.shape[1]-2

scan_layout = np.zeros((n_tiers,dim1,dim2),object)
for i in range(n_tiers):
    scan_layout[i,:,:] = x[x[:,0]==i+1,2:]

mask = scan_layout!=0 #cells to extract from

tier_mask = np.sum(np.sum(mask,1),1)>0


# %% step 1: segment vertically
#n_tiers = 2



q = -np.mean(np.mean(vol,1),1)
sig = q.copy()
q = q-q.min()

thresh = 3e7/(vol.shape[1]*vol.shape[2])
q[q<thresh] = 0

#vert_pks = find_peaks(q>.85*q.max())[0]

vert_pks = find_peaks(q,width=10)[0]
vert_pks = np.concatenate((np.array([0]),vert_pks))
vert_pks = np.concatenate((vert_pks,np.array([len(q)])) )


plt.figure()
plt.plot(np.arange(len(q)),sig)
plt.xlabel('slice height (z)'); plt.ylabel('(-) average tier density'); plt.title('tier segmentation')
plt.show()

if (TIER_THRESH ==None) | (REDO_TIERS==True):
    print('identified vertical peaks:')
    print(vert_pks)
    yn_vertseg = input('are these ok? enter y/n. # tiers is %1d, # nonempty tiers is %2d \n' % (n_tiers, np.sum(tier_mask)))

    if yn_vertseg =='n':
        ex = input('here is the raw data. enter correct peaks to splice along, separated by commas only:')
        ex= np.array(ex.split(',')).astype(int)

        #this isn't quite proper, but:
        ex[ex==-1] = len(q) #only cuz (-) indexing will throw future segmentation


    else:
        ex = vert_pks
    update_param(scan_num,'TIER_THRESH',ex.tolist())
else:
    print('using saved vertical peaks')
    ex = np.array(TIER_THRESH)







ranges = []
for i in range(len(ex)-1):
    ranges.append([ex[i],ex[i+1]])


ranges = ranges[-1::-1] #fix to be in same order as tiers in spreadsheet

tiers = np.arange(n_tiers)[tier_mask]
ranges = [ranges[i] for i in tiers]

SLICES = []
for i in range(len(ranges)):
    SLICES.append(vol[ranges[i][0]:ranges[i][1],:,:])

I = [x.mean(0) for x in SLICES]


#use this if extracting isolevels by user input... t1,t2,t3 for autothresh

plt.figure(figsize=(10,4))
plt.hist(vol.flatten(),bins=500)
plt.title('voxel value histogram'); plt.ylabel('frequency'); plt.xlabel('voxel value')

plt.show()

if REDO_ISO:
    t1t2t3 = input('enter t1, t2, t3 separated by commas only: ')
    t1t2t3= np.array(t1t2t3.split(',')).astype(int)
    update_param(scan_num,'ISO_THRESHOLDS',t1t2t3.tolist())
else:
    t1t2t3 = get_parameter(scan_num,'ISO_THRESHOLDS')



EXTRACTS = []


for i in range(len(tiers)):

    tier = tiers[i]; #Im = I[i].T
    si = SLICES[i].transpose((0,2,1))


    if AUTO_ROT:
        ''' same params go into id_cardboard:'''
        angi,Im = autorot2(si,t1=t1t2t3[0],t2=t1t2t3[1],t3=t1t2t3[2],title='rotation for tier '+str(i+1))
        #Im =   rotate(Im,angi,preserve_range=True)

    else:
        ''' same params go into autorot2:'''
        Im = id_cardboard(si,t1=t1t2t3[0],t2=t1t2t3[1],t3=t1t2t3[2])
        angi = 0

    plt.figure()
    plt.title('detected dividers for tier '+str(i+1))
    plt.imshow(Im.T)
    plt.axis('off')

    angi = -angi #so as to operate on unmirrored CT scans....
    #angi is no longer used within this code apart from passing to EX3...


    maski = mask[i,:,:]
    layouti = scan_layout[i,:,:]

    rowi,coli = np.where(maski)


    Im2 = Im.copy()


    Isum0 = np.sum(Im,0)
    Isum1 = np.sum(Im,1)


    if AUTO_SEG==True:
        x0 = auto_seg(Isum0,4)
        x1 = auto_seg(Isum1,7)

    else:
        thresh0 = Im.shape[0]*THRESH0/225
        thresh1 = Im.shape[1]*THRESH1/225

        x0 = (Isum0>thresh0).astype(int)
        x1 = (Isum1>thresh1).astype(int)



    if x0[0]==1:
        x0[0]=0


    i0firsts = np.where(x0[0:-1]<x0[1:])[0]
    i0lasts  = np.where(x0[1:]<x0[0:-1])[0]
    i1firsts = np.where(x1[0:-1]<x1[1:])[0]
    i1lasts  = np.where(x1[1:]<x1[0:-1])[0]






    if INPUT_ROWS:
        plt.figure()
        plt.imshow(Im)

        plt.figure()
        plt.plot(Isum1,scalex=5)
        plt.xticks(np.arange(0,Isum1.shape[0],20))
        plt.show()
        print(find_peaks(Isum1,width=3)[0])
        i1m = np.array(input('enter peaks, separated only by single spaces: \n').split(' ')).astype(int)
        update_param(scan_num,'R'+str(i),i1m)
    else:
        i1m = np.floor((i1firsts+i1lasts)/2).astype(int)


    if INPUT_COLS:
        plt.figure()
        plt.plot(Isum0,scalex=5)
        plt.xticks(np.arange(0,Isum1.shape[0],20))
        plt.show()
        print(find_peaks(Isum0,width=3)[0])

        i0m = np.array(input('enter peaks, separated only by single spaces: \n').split(' ')).astype(int)

        update_param(scan_num,'C'+str(i),i0m)

    else:
        i0m = np.floor((i0firsts+i0lasts)/2).astype(int)


    plt.figure()
    plt.plot(np.arange(len(Isum0)),Isum0,linewidth=2)
    #plt.plot(np.arange(len(Isum0)),x0*Isum0.max())
    ma = 1.1*Isum0.max()
    for qq in i0lasts:
        plt.plot([qq,qq], [0,ma],'r')

    plt.title('row segmentation for tier '+str(i+1))
    plt.xlabel('row'); plt.ylabel('sum along columns')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(Isum1)),Isum1,linewidth=2)
    #plt.plot(np.arange(len(Isum1)),x1*Isum1.max())
    ma = 1.1*Isum1.max()
    for qq in i1lasts:
        plt.plot([qq,qq], [0,ma],'r')

    plt.title('column segmentation for tier '+str(i+1))
    plt.xlabel('column'); plt.ylabel('sum along rows')
    plt.show()



    print(i,'n_row', len(i1m),'n_col', len(i0m))

    spacing = 3+np.min([np.min(i1m[1:]-i1m[0:-1]),np.min(i0m[1:]-i0m[0:-1])])





    fullboarder = True
    if fullboarder==True:
        colstart = 0
        colend = Im.shape[1]-1
        rowstart = 0
        rowend = Im.shape[0]-1
    else:
        colstart = np.max((0,i0m[0]-spacing))
        colend   = np.min((Im.shape[1]-1,i0m[-1]+spacing))
        rowstart = np.max((0,i1m[0]-spacing))
        rowend   = np.min((Im.shape[0]-1,i1m[-1]+spacing))

    col = np.array( [colstart]+ i0m.tolist() +[colend])
    row = np.array( [rowstart]+ i1m.tolist() +[rowend])


    rowcolrng = np.vstack((row[rowi],row[rowi+1],col[coli],col[coli+1])).T
    namesi = layouti[maski,None]




    #show identified corners:
    imnum2get = str(int(np.mean(ranges[i]))*10)
    if len(imnum2get)<4:
        for j in range(4-len(imnum2get)):
            imnum2get = '0'+imnum2get

    if USE_SSH & SCPPLOT:
        im2get = 'AMAAZE_'+str(scan_num)+' Y_'+imnum2get+'.tif'

        scp.get(os.path.join(slicepath,im2get))

        plt.imread(im2get)

        #imdisp = rotate(rotate(plt.imread(im2get),ang2rot)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]],-angi)
        imdisp = rotate(plt.imread(im2get),ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]]

        imdisp = rotate(imdisp, angi)

    drawrow = ( (rowsz/vol.shape[1])*(rowcolrng[:,2:4]) ).astype(int)
    drawcol = ( (colsz/vol.shape[2])*(rowcolrng[:,0:2]) ).astype(int)

    EXTRACTS.append( np.concatenate((namesi,[[10*ranges[i][0],10*ranges[i][1]]]*rowcolrng.shape[0], drawrow, drawcol, len(namesi)*[[angi]], len(namesi)*[rowrng.tolist()], len(namesi)*[colrng.tolist()] ),1))
    #np.concatenate((namesi , drawrow, drawcol, [ranges[i]]*rowcolrng.shape[0], len(namesi)*[[angi]] ),1) )


    #draw_boxes(Im,rowcolrng[:,0:2],rowcolrng[:,2:4])
    if USE_SSH & SCPPLOT:
        draw_boxes(imdisp,drawrow,drawcol,title = 'segmentation for tier '+str(i+1))




E = np.concatenate(EXTRACTS)
#E2 = np.concatenate([E[:,0,None], 10*E[:,5:7], rowrng[0] + np.floor(rowsz/currentsz[0]*(currentsz[1] - E[:,4:2:-1])).astype(int), colrng[0]+ np.floor(colsz/currentsz[1]*(currentsz[0] - E[:,2:0:-1])).astype(int), E[:,7,None] ],1)
#E2 = np.concatenate([E[:,0,None], 10*E[:,5:7], drawrow, drawcol, E[:,7,None], E.shape[0]*[rowrng.tolist()],E.shape[0]*[colrng.tolist()] ],1)
                                                #row                                            #col


pd.DataFrame(E).to_csv('./CT'+str(scan_num)+'.csv', header=False, index=False)


