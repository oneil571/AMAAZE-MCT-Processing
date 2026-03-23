#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:53:20 2024

@author: rileywilde
"""
import pdb
from imports import *



def parse_option():
    parser = argparse.ArgumentParser('inputs')
    parser.add_argument('--folder', type=str, required=True,
                        help='what scan are we doing? enter folder path.')
    parser.add_argument('--local_slice_path', type=str, default='Slices',
                        help='local path to slices, if it exists. Will not be used if not local. SSH is also compatible.')
    parser.add_argument('--slicepath_ssh', type=str, default='',
                        help='where remote files are')
    parser.add_argument('--slicepath_local', type=str, default='',
                        help='where local files are')
    parser.add_argument('--slicename', type=str, default='TeethScan2_79microns_Test_3_dddd.tif',
                        help='where .tif images are kept.')
    parser.add_argument('--transpose', type=int, default=0,
                        help='transpose loaded images')
    parser.add_argument('--csv', type=str, default='../sofia_scan_template_7-21.csv',
                        help='transpose loaded images')  
    parser.add_argument('--nrow', type=int, default=4,
                        help='num rows')  
    parser.add_argument('--ncol', type=int, default=4,
                        help='num cols')  
    parser.add_argument('--use_ssh', type=int, default=0,
                            help='1 to grab slices from remote host, 0 to not.')  


                

    return parser.parse_args()

#if __name__ == '__main__':
for qqqq in [1]:

    opt = parse_option()
    # %% GET RID OF THIS PRIOR TO SHARING:::
    if opt.use_ssh:
        print('please configure SSH and enter your password in line 48')
        ssh = createSSHClient('host',22,'user', 'no')
        scp = SCPClient(ssh.get_transport())

    os.chdir(opt.folder)
    scan_num = int(os.getcwd().split(os.sep)[-1].split('_')[-1])

    slicepath_ssh = opt.slicepath_ssh
    slicepath_local = opt.slicepath_local


    REDO_TIERS=True
    REDO_ISO = True
    PLOTTING = True
    LOCAL_SLICES = os.path.exists(slicepath_local)


    AUTO_ROT,AUTO_SEG,INPUT_ROWS,INPUT_COLS,TIER_THRESH,INVERT,THRESH0,THRESH1 = read_hyperparameters(scan_num)


    saveddata = np.load('ct'+str(scan_num)+'_new.npz')
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

    x = pd.read_csv(opt.csv).fillna(0)
    x = x.to_numpy()

    x = x[x[:,0]==scan_num,1:].copy()
    x = x[:,:opt.ncol+2]

    if opt.transpose==0:
        arr_sorted = x[np.lexsort((x[:,1], x[:,0]))]  # sort by t, then m

        T = int(x[:,0].max())
        data = arr_sorted[:, 2:].reshape(T, opt.nrow, opt.ncol)

        data_T = np.transpose(data, (0, 2, 1))

        t_idx = np.repeat(np.arange(1, T+1), opt.ncol)
        col_idx = np.tile(np.arange(1, opt.ncol+1), T)

        x = np.column_stack([
            t_idx,
            col_idx,
            data_T.reshape(T*opt.ncol, opt.nrow)
        ])
        ncol = opt.ncol
        nrow = opt.nrow

        opt.ncol = nrow
        opt.nrow = ncol


    n_tiers = int(x[:,0].max())
    dim1 = opt.nrow #int(x[:,1].max())
    dim2 = opt.ncol #x.shape[1]-2

    scan_layout = np.zeros((n_tiers,dim1,dim2),object)
    for i in range(n_tiers):
        scan_layout[i,:,:] = x[x[:,0]==i+1,2:]

    mask = scan_layout!=0 #cells to extract from

    tier_mask = np.sum(np.sum(mask,1),1)>0


    # %% step 1: segment vertically

    q = -np.mean(np.mean(vol,1),1)
    sig = q.copy()
    q = q-q.min()

    thresh = 3e7/(vol.shape[1]*vol.shape[2])
    q[q<thresh] = 0

    #vert_pks = find_peaks(q>.85*q.max())[0]

    vert_pks = find_peaks(q,width=10)[0]
    vert_pks = np.concatenate((np.array([0]),vert_pks))
    vert_pks = np.concatenate((vert_pks,np.array([len(q)])) )


    fig = plt.figure()
    plt.plot(np.arange(len(q)),sig)
    plt.xlabel('slice height (z)'); plt.ylabel('(-) average tier density'); plt.title('tier segmentation')
    for vvv in vert_pks:
        plt.axvline(x=vvv, color='green', linestyle='--', linewidth=2)

    if (TIER_THRESH ==None) | (REDO_TIERS==True):
        print('green are identified vertical peaks - are they ok? \n if not, please click new peaks. \n # tiers is %1d, # nonempty tiers is %2d \n' % (n_tiers, np.sum(tier_mask)))
        
        clicked_x = []  # store clicked x-values

        def onclick(event):
            if event.inaxes:  # make sure click is inside the axes
                x_click = event.xdata
                clicked_x.append(x_click)
                # Draw vertical line
                event.inaxes.axvline(x_click, color='r', linestyle='--')
                plt.draw()
                print(f"Clicked x = {x_click:.2f}")

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show(block=False)        
        
        a = input('please press enter once done (no clicks = use suggested values) \n')
        yn_vertseg = clicked_x #input('are these ok? enter y/n. # tiers is %1d, # nonempty tiers is %2d \n' % (n_tiers, np.sum(tier_mask)))

        if len(yn_vertseg) > 0: #'n':
            #ex = yn_vertseg #input('here is the raw data. enter correct peaks to splice along, separated by commas only:')
            #ex= np.array(ex.split(',')).astype(int)
            ex = np.array(yn_vertseg).astype(int)
            ex[ex<0]=0
            ex[ex>len(q)] = len(q)
            
            #this isn't quite proper, but:
            #ex[ex==-1] = len(q) #only cuz (-) indexing will throw future segmentation
            print("new vertical peaks", ex)
        else:
            ex = vert_pks
            print("using ", ex)
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

    fig = plt.figure(figsize=(10,4))
    plt.hist(vol.flatten(),bins=500)
    plt.title('voxel value histogram'); plt.ylabel('frequency'); plt.xlabel('voxel value')
    plt.yscale('log')

    


    if REDO_ISO:
        clicked_x = []  # store clicked x-values
        def onclick(event):
            if event.inaxes:  # make sure click is inside the axes
                x_click = event.xdata
                clicked_x.append(x_click)
                # Draw vertical line
                event.inaxes.axvline(x_click, color='r', linestyle='--')
                plt.draw()
                print(f"Clicked x = {x_click:.2f}")

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show(block=False)        
        
        a = input('Please click 3 density values (start & end of divider range, start of obj range). Press enter once done \n')
        t1t2t3 = clicked_x #input('are these ok? enter y/n. # tiers is %1d, # nonempty tiers is %2d \n' % (n_tiers, np.sum(tier_mask)))

        t1t2t3 = np.array(t1t2t3).astype(int)

        #t1t2t3 = input('enter t1, t2, t3 separated by commas only: ')
        #t1t2t3= np.array(t1t2t3.split(',')).astype(int)
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
            Im = id_cardboard(si,t1=t1t2t3[0],t2=t1t2t3[1],t3=t1t2t3[2]) #nov8: had to drop frac thresh, .25 
            angi = 0

        plt.figure()
        plt.title('detected dividers for tier '+str(i+1))

        if opt.transpose:
            plt.imshow(Im.T)
        else:
            plt.imshow(Im)
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
            x0 = auto_seg(Isum0,opt.nrow-1)
            x1 = auto_seg(Isum1,opt.ncol-1)

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
            plt.show(block=False)
            print(find_peaks(Isum1,width=3)[0])
            i1m = np.array(input('enter peaks, separated only by single spaces: \n').split(' ')).astype(int)
            update_param(scan_num,'R'+str(i),i1m)
        else:
            i1m = np.floor((i1firsts+i1lasts)/2).astype(int)


        if INPUT_COLS:
            plt.figure()
            plt.plot(Isum0,scalex=5)
            plt.xticks(np.arange(0,Isum1.shape[0],20))
            plt.show(block=False)
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
        plt.show(block=False)

        plt.figure()
        plt.plot(np.arange(len(Isum1)),Isum1,linewidth=2)
        #plt.plot(np.arange(len(Isum1)),x1*Isum1.max())
        ma = 1.1*Isum1.max()
        for qq in i1lasts:
            plt.plot([qq,qq], [0,ma],'r')

        plt.title('column segmentation for tier '+str(i+1))
        plt.xlabel('column'); plt.ylabel('sum along rows')
        plt.show(block=False)



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
        imnum2get = int(np.mean(ranges[i]))*10
        im2get = opt.slicename.replace('dddd',f"{imnum2get:04d}")

        if opt.use_ssh & PLOTTING:

            scp.get(os.path.join(slicepath_ssh,im2get))

            imdisp = rotate(plt.imread(im2get),ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]]

            imdisp = rotate(imdisp, angi)

        elif PLOTTING & LOCAL_SLICES:
            imdisp = rotate(plt.imread(os.path.join(slicepath_local,im2get)),ang2rot,preserve_range=True)[rowrng[0]:rowrng[1],colrng[0]:colrng[1]]
            imdisp = rotate(imdisp, angi)

        #if opt.transpose: #how code was originally
        drawrow = ( (rowsz/vol.shape[1])*(rowcolrng[:,2:4]) ).astype(int)
        drawcol = ( (colsz/vol.shape[2])*(rowcolrng[:,0:2]) ).astype(int)
        #else: #experimental
        #    drawcol = ( (rowsz/vol.shape[1])*(rowcolrng[:,2:4]) ).astype(int)
        #   drawrow = ( (colsz/vol.shape[2])*(rowcolrng[:,0:2]) ).astype(int)

        EXTRACTS.append( np.concatenate((namesi,[[10*ranges[i][0],10*ranges[i][1]]]*rowcolrng.shape[0], drawrow, drawcol, len(namesi)*[[angi]], len(namesi)*[rowrng.tolist()], len(namesi)*[colrng.tolist()] ),1))
        #np.concatenate((namesi , drawrow, drawcol, [ranges[i]]*rowcolrng.shape[0], len(namesi)*[[angi]] ),1) )


        #draw_boxes(Im,rowcolrng[:,0:2],rowcolrng[:,2:4])
        if (opt.use_ssh | LOCAL_SLICES) & PLOTTING :
            draw_boxes(imdisp,drawrow,drawcol,title = 'segmentation for tier '+str(i+1))



    E = np.concatenate(EXTRACTS)
    #E2 = np.concatenate([E[:,0,None], 10*E[:,5:7], rowrng[0] + np.floor(rowsz/currentsz[0]*(currentsz[1] - E[:,4:2:-1])).astype(int), colrng[0]+ np.floor(colsz/currentsz[1]*(currentsz[0] - E[:,2:0:-1])).astype(int), E[:,7,None] ],1)
    #E2 = np.concatenate([E[:,0,None], 10*E[:,5:7], drawrow, drawcol, E[:,7,None], E.shape[0]*[rowrng.tolist()],E.shape[0]*[colrng.tolist()] ],1)
                                                    #row                                            #col


    pd.DataFrame(E).to_csv('CT'+str(scan_num)+'.csv', header=False, index=False)
    os.chdir('..')

