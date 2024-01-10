def load_ct_data(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                 data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',um_per_pixel = 0.108333,
                tag_cts = 'cts_all',tag_fl = r'\Decoded\*_cts.npz'):
    dic_coords = get_all_pos(analysis_folder = analysis_folder,data_folder =data_folder,set_=set_)

    fls = glob.glob(analysis_folder+os.sep+tag_fl)
    ctM = None
    cm_cells,ifovs = [],[]
    for fl in tqdm(np.sort(fls)):
        if set_ in fl:
            dic = np.load(fl)
            gns_names = dic['gns_names']
            ctM_ = dic[tag_cts]
            cm_cells_ = dic['cm_cells']
            cm_cells.extend(cm_cells_)
            ifovs += [get_ifov(fl)]*len(cm_cells_)
            if ctM is None: ctM=ctM_ 
            else: ctM = np.concatenate([ctM,ctM_],axis=1)

    ifovs = np.array(ifovs)
    cm_cells = np.array(cm_cells)
    cm_cells = cm_cells[:,1:]
    abs_pos = np.array([dic_coords[ifov] for ifov in ifovs])
    abs_pos = abs_pos[:,::-1]*np.array([1,-1])
    
    cm_cellsf = cm_cells*um_per_pixel+abs_pos
    return ctM,gns_names,cm_cellsf
def load_ct_data_ptb_aso(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                 data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',um_per_pixel = 0.108333,
                tags_cts = ['ptbp_cts','aso_mean'],tag_fl = r'\Decoded\*_cts_ptb-aso.npz'):
    dic_coords = get_all_pos(analysis_folder = analysis_folder,data_folder =data_folder,set_=set_)

    fls = glob.glob(analysis_folder+os.sep+tag_fl)
    ctM = None
    cm_cells,ifovs = [],[]
    for fl in tqdm(np.sort(fls)):
        if set_ in fl:
            dic = np.load(fl)
            gns_names = tags_cts
            ctM_ = np.array([dic[tag_cts] for tag_cts in tags_cts])
            cm_cells_ = dic['cm_cells']
            cm_cells.extend(cm_cells_)
            ifovs += [get_ifov(fl)]*len(cm_cells_)
            if ctM is None: ctM=ctM_ 
            else: ctM = np.concatenate([ctM,ctM_],axis=1)

    ifovs = np.array(ifovs)
    cm_cells = np.array(cm_cells)
    cm_cells = cm_cells[:,1:]
    abs_pos = np.array([dic_coords[ifov] for ifov in ifovs])
    abs_pos = abs_pos[:,::-1]*np.array([1,-1])
    
    cm_cellsf = cm_cells*um_per_pixel+abs_pos
    return ctM,gns_names,cm_cellsf
def example_rerun():
    #from ioMicro import *
    set_='set1'
    ifov=0
    dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
    for ifov in tqdm(np.arange(320)):
        
        dec.get_set_ifov(ifov=ifov,set_=set_,keepH = [1,2,3,4,5,6,7,8],ncols=3)
        save_fl = dec.save_file_cts.replace('_cts.npz','_ctsV2.npz')
        if not os.path.exists(save_fl):
            dec.load_segmentation()
            dec.load_library()
            dic = np.load(dec.save_file_dec)
            for key in list(dic.keys()):
                setattr(dec,key,dic[key])
            
            ###             perform some refinement
            
            dec.cts_all_pm = dec.get_counts_per_cell(nbad=0)
            dec.cts_all = dec.get_counts_per_cell(nbad=1)
            np.savez(save_fl,
                     cts_all_pm = dec.cts_all_pm,cts_all = dec.cts_all,
                     gns_names=dec.gns_names,cm_cells=dec.cm_cells,vols=dec.vols)


#import napari
import numpy as np,pickle,glob,os
import cv2
from scipy.signal import convolve,fftconvolve
from tqdm import tqdm
import matplotlib.pylab as plt
from scipy.spatial import cKDTree

def get_p99(fl_dapi,resc=4):
    im = read_im(fl_dapi)
    im_ = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
    img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(50,50)))[::resc,::resc]
    p99 = np.percentile(img,99.9)
    p1 = np.percentile(img,1)
    img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
    plt.figure()
    plt.imshow(img,cmap='gray')
    return p99
def resize(im,shape_ = [50,2048,2048]):
    """Given an 3d image <im> this provides a quick way to resize based on nneighbor sampling"""
    z_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    x_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[2]-1,shape_[2])).astype(int)
    return im[z_int][:,x_int][:,:,y_int]

import scipy.ndimage as ndimage
def get_final_cells_cyto(im_polyA,final_cells,icells_keep=None,ires = 4,iresf=10,dist_cutoff=10):
    """Given a 3D im_polyA signal and a segmentation fie final_cells """
    incell = final_cells>0
    med_polyA = np.median(im_polyA[incell])
    med_nonpolyA = np.median(im_polyA[~incell])
    im_ext_cells = im_polyA>(med_polyA+med_nonpolyA)/2


    X = np.array(np.where(im_ext_cells[:,::ires,::ires])).T
    Xcells = np.array(np.where(final_cells[:,::ires,::ires]>0)).T
    from sklearn.neighbors import KDTree

    kdt = KDTree(Xcells[::iresf], leaf_size=30, metric='euclidean')
    icells_neigh = final_cells[:,::ires,::ires][Xcells[::iresf,0],Xcells[::iresf,1],Xcells[::iresf,2]]
    dist,neighs = kdt.query(X, k=1, return_distance=True)
    dist,neighs = np.squeeze(dist),np.squeeze(neighs)

    final_cells_cyto = im_ext_cells[:,::ires,::ires]*0
    if icells_keep is not None:
        keep_cyto = (dist<dist_cutoff)&np.in1d(icells_neigh[neighs],icells_keep)
    else:
        keep_cyto = (dist<dist_cutoff)
    final_cells_cyto[X[keep_cyto,0],X[keep_cyto,1],X[keep_cyto,2]] = icells_neigh[neighs[keep_cyto]]
    final_cells_cyto = resize(final_cells_cyto,im_polyA.shape)
    return final_cells_cyto
def slice_pair_to_info(pair):
    sl1,sl2 = pair
    xm,ym,sx,sy = sl2.start,sl1.start,sl2.stop-sl2.start,sl1.stop-sl1.start
    A = sx*sy
    return [xm,ym,sx,sy,A]
def get_coords(imlab1,infos1,cell1):
    xm,ym,sx,sy,A,icl = infos1[cell1-1]
    return np.array(np.where(imlab1[ym:ym+sy,xm:xm+sx]==icl)).T+[ym,xm]
def cells_to_coords(imlab1,return_labs=False):
    """return the coordinates of cells with some additional info"""
    infos1 = [slice_pair_to_info(pair)+[icell+1] for icell,pair in enumerate(ndimage.find_objects(imlab1))
    if pair is not None]
    cms1 = np.array([np.mean(get_coords(imlab1,infos1,cl+1),0) for cl in range(len(infos1))])
    if len(cms1)==0: cms1 = []
    else: cms1 = cms1[:,::-1]
    ies = [info[-1] for info in infos1]
    if return_labs:
        return imlab1.copy(),infos1,cms1,ies
    return imlab1.copy(),infos1,cms1
def resplit(cells1,cells2,nmin=100):
    """intermediate function used by standard_segmentation.
    Decide when comparing two planes which cells to split"""
    imlab1,infos1,cms1 = cells_to_coords(cells1)
    imlab2,infos2,cms2 = cells_to_coords(cells2)
    if len(cms1)==0 or len(cms2)==0:
        return imlab1,imlab2,[],0
    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
    dic_cell2_1_split = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if len(dic_cell2_1[cell])>1 and cell>0}
    cells1_split = list(dic_cell2_1_split.keys())
    imlab1_cp = imlab1.copy()
    number_of_cells_to_split = len(cells1_split)
    for cell1_split in cells1_split:
        count = np.max(imlab1_cp)+1
        cells2_to1 = dic_cell2_1_split[cell1_split]
        X1 = get_coords(imlab1,infos1,cell1_split)
        X2s = [get_coords(imlab2,infos2,cell2) for cell2 in cells2_to1]
        from scipy.spatial.distance import cdist
        X1_K = np.argmin([np.min(cdist(X1,X2),axis=-1) for X2 in X2s],0)

        for k in range(len(X2s)):
            X_ = X1[X1_K==k]
            if len(X_)>nmin:
                imlab1_cp[X_[:,0],X_[:,1]]=count+k
            else:
                #number_of_cells_to_split-=1
                pass
    imlab1_,infos1_,cms1_ = cells_to_coords(imlab1_cp)
    return imlab1_,infos1_,cms1_,number_of_cells_to_split

def converge(cells1,cells2):
    imlab1,infos1,cms1,labs1 = cells_to_coords(cells1,return_labs=True)
    imlab2,infos2,cms2 = cells_to_coords(cells2)
    
    if len(cms1)==0 or len(cms2)==0:
        return imlab1,imlab2
    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
        
    dic_cell2_1_match = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if cell>0}
    cells2_kp = [e_ for e in dic_cell2_1_match for e_ in dic_cell2_1_match[e]]
    modify_cells2 = np.setdiff1d(np.arange(len(cms2)),cells2_kp)
    imlab2_ = imlab2*0
    for cell1 in dic_cell2_1_match:
        for cell2 in dic_cell2_1_match[cell1]:
            xm,ym,sx,sy,A,icl = infos2[cell2-1]
            im_sm = imlab2[ym:ym+sy,xm:xm+sx]
            imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=labs1[cell1-1]
    count_cell = max(np.max(imlab2_),np.max(labs1))
    for cell2 in modify_cells2:
        count_cell+=1
        xm,ym,sx,sy,A,icl = infos2[cell2-1]
        im_sm = imlab2[ym:ym+sy,xm:xm+sx]
        imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=count_cell
    return imlab1,imlab2_
def final_segmentation(fl_dapi,
                        analysis_folder=r'X:\DCBB_human__11_18_2022_Analysis',
                        plt_val=True,
                        rescz = 4,trimz=0, resc=4,p99=2000,force=False):
    segm_folder = analysis_folder+os.sep+'Segmentation'
    if not os.path.exists(segm_folder): os.makedirs(segm_folder)
    
    save_fl  = segm_folder+os.sep+os.path.basename(fl_dapi).split('.')[0]+'--'+os.path.basename(os.path.dirname(fl_dapi))+'--dapi_segm.npz'
    
    if not os.path.exists(save_fl) or force:
        im = read_im(fl_dapi)
        #im_mid_dapi = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
        im_dapi = im[-1,::rescz]
        if trimz!=0:
            im_dapi = im_dapi[trimz:-trimz]
        
        im_seg_2,im_seg_1 = standard_segmentation(im_dapi,resc=resc,sz_min_2d=400,sz_cell=22,use_gpu=True,model='cyto',p99=p99)
        shape = np.array(im[-1].shape)
        np.savez_compressed(save_fl,segm = im_seg_2,shape = shape,segm_2d=im_seg_1)

        

    if plt_val:
        fl_png = save_fl.replace('.npz','__segim.png')
        #if not os.path.exists(fl_png):
        im = read_im(fl_dapi)
        im_seg_2 = np.load(save_fl)['segm']
        shape =  np.load(save_fl)['shape']
        
        im_dapi_sm = resize(im[-1],im_seg_2.shape)
        img = np.array(im_dapi_sm[im_dapi_sm.shape[0]//2],dtype=np.float32)
        masks_ = im_seg_2[im_seg_2.shape[0]//2]
        from cellpose import utils
        outlines = utils.masks_to_outlines(masks_)
        p1,p99 = np.percentile(img,1),np.percentile(img,99.9)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        outX, outY = np.nonzero(outlines)
        imgout= np.dstack([img]*3)
        imgout[outX, outY] = np.array([1,0,0]) # pure red
        fig = plt.figure(figsize=(20,20))
        plt.imshow(imgout)
        
        fig.savefig(fl_png)
        plt.close('all')
        print("Saved file:"+fl_png)
        
def get_counts_per_cell(self,th_cor=0.5):
    dic_th = self.dic_th
    icol = self.icol
    im_segm = self.im_segm
    shapesm = self.im_segm.shape
    shape = self.shape

    Xh = self.Xh.copy()
    cor = Xh[:,-2]
    h = Xh[:,-1]
    keep = h>dic_th.get(icol,0)
    Xh = Xh[keep]
    txyz = np.array(self.dic_drift['txyz'])
    txyz[txyz>0]=0
    Xcms = Xh[:,:3]-txyz
    Xred = np.round((Xcms/shape)*shapesm).astype(int)
    good = ~np.any((Xred>=shapesm)|(Xred<0),axis=-1)
    Xh,Xred = Xh[good],Xred[good]
    self.Xred = Xred
    icells = im_segm[tuple(Xred.T)]
    cells,cts = np.unique(icells[(Xh[:,-2]>th_cor)],return_counts=True)
    self.good_counts = {c_+self.ifov*10**6:ct_ for c_,ct_ in zip(cells,cts) if c_>0}
    cells,cts = np.unique(icells[(Xh[:,-2]<th_cor)],return_counts=True)
    self.bad_counts = {c_+self.ifov*10**6:ct_ for c_,ct_ in zip(cells,cts) if c_>0}
from scipy import ndimage as nd
def get_int_im1_im2(im1,im2,th_int=0.5):
    inters = ((im1>0)&(im2>0)).astype(int)
    im1_in2 = im1*inters
    N1max = np.max(im1)+1
    im2_in1 = im2*inters*N1max
    iint,cts = np.unique(im1_in2+im2_in1,return_counts=True)
    c1,cts1 = np.unique(im1,return_counts=True)
    dic_c1 = {c_:ct_ for c_,ct_ in zip(c1,cts1) if c_>0}
    c2,cts2 = np.unique(im2,return_counts=True)
    dic_c2 = {c_:ct_ for c_,ct_ in zip(c2,cts2) if c_>0}
    dic_int= {(c1,c2):(ct/dic_c1[c1],ct/dic_c2[c2]) for c2,c1,ct in zip(iint//N1max,iint%N1max,cts) 
         if (c1>0) and (c2>0) and (c1!=c2)}
    objs1 = nd.find_objects(im1)
    objs2 = nd.find_objects(im2)
    for cch in dic_int:
        c1,c2 = cch
        ic1,ic2 = dic_int[cch]
        obj1,obj2 = objs1[c1-1],objs2[c2-1]
        if (ic1>th_int) or (ic2>th_int):
            c_ = np.min([c1,c2])
            im1[obj1][im1[obj1]==c1]=c_
            im2[obj2][im2[obj2]==c2]=c_
    return im1,im2
def stitch3D(im_segm,niter=5,th_int=0.5):
    for it_ in range(niter):
        for iim in range(len(im_segm)-1):
            im_segm[iim],im_segm[iim+1] = get_int_im1_im2(im_segm[iim],im_segm[iim+1],th_int=th_int)
    return im_segm
def standard_segmentation(im_dapi,resc=4,sz_min_2d=400,sz_cell=22,use_gpu=True,model='cyto',p99=2000):
    """Using cellpose with nuclei mode"""
    from cellpose import models, io,utils
    from scipy import ndimage
    model = models.Cellpose(gpu=use_gpu, model_type=model)
    #decided that resampling to the 4-2-2 will make it faster
    #im_dapi_3d = im_dapi[::rescz,::resc,::resc].astype(np.float32)
    chan = [0,0]
    masks_all = []
    flows_all = []
    from tqdm import tqdm
    for im in tqdm(im_dapi):
        im_ = np.array(im,dtype=np.float32)
        img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(150,150)))[::resc,::resc]
        p1 = np.percentile(img,1)
        if p99 is None:
            p99 = np.percentile(img,99.9)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        masks, flows, styles, diams = model.eval(img, diameter=sz_cell, channels=chan,
                                             flow_threshold=0.85,cellprob_threshold=-2,min_size=sz_min_2d,normalize=False)
        
        
        umsks = np.unique(masks)
        means = ndimage.mean(im_[::resc,::resc],masks,index=umsks)
        th = (np.percentile(means[1:],25)+means[0])/2
        bad = np.in1d(masks,umsks[means<th])
        #masks_ = masks.copy()
        masks[bad.reshape(masks.shape)]=0
        
        masks_all.append(masks)#,hole_size=3
        flows_all.append(flows[0])
    masks_all = np.array(masks_all)

    sec_half = list(np.arange(int(len(masks_all)/2),len(masks_all)-1))
    first_half = list(np.arange(0,int(len(masks_all)/2)))[::-1]
    indexes = first_half+sec_half
    masks_all_cp = masks_all.copy()
    max_split = 1
    niter = 0
    while max_split>0 and niter<2:
        max_split = 0
        for index in tqdm(indexes):
            cells1,cells2 = masks_all_cp[index],masks_all_cp[index+1]
            imlab1_,infos1_,cms1_,no1 = resplit(cells1,cells2)
            imlab2_,infos2_,cms2_,no2 = resplit(cells2,cells1)
            masks_all_cp[index],masks_all_cp[index+1] = imlab1_,imlab2_
            max_split += max(no1,no2)
            #print(no1,no2)
        niter+=1
    masks_all_cpf = masks_all_cp.copy()
    for index in tqdm(range(len(masks_all_cpf)-1)):
        cells1,cells2 = masks_all_cpf[index],masks_all_cpf[index+1]
        cells1_,cells2_ = converge(cells1,cells2)
        masks_all_cpf[index+1]=cells2_
    #masks_all_cpf_ = stitch3D(masks_all_cpf,niter=5,th_int=0.75)
    return masks_all_cpf,masks_all

def get_dif_or_ratio(im_sig__,im_bk__,sx=20,sy=20,pad=5,col_align=-2):
    size_ = im_sig__.shape
    imf = np.ones(size_,dtype=np.float32)
    #resc=5
    #ratios = [np.percentile(im_,99.95)for im_ in im_sig__[:,::resc,::resc,::resc]/im_bk__[:,::resc,::resc,::resc]]
    for startx in tqdm(np.arange(0,size_[2],sx)[:]):
        for starty in np.arange(0,size_[3],sy)[:]:
            startx_ = startx-pad
            startx__ = startx_ if startx_>0 else 0
            endx_ = startx+sx+pad
            endx__ = endx_ if endx_<size_[2] else size_[2]-1

            starty_ = starty-pad
            starty__ = starty_ if starty_>0 else 0
            endy_ = starty+sy+pad
            endy__ = endy_ if endy_<size_[3] else size_[3]-1

            padx_end = pad+endx_-endx__
            pady_end = pad+endy_-endy__
            padx_st = pad+startx_-startx__
            pady_st = pad+starty_-starty__

            ims___ = im_sig__[:,:,startx__:endx__,starty__:endy__]
            imb___ = im_bk__[:,:,startx__:endx__,starty__:endy__]

            txy = get_txy_small(np.max(imb___[col_align],axis=0),np.max(ims___[col_align],axis=0),sz_norm=5,delta=3,plt_val=False)
            tzy = get_txy_small(np.max(imb___[col_align],axis=1),np.max(ims___[col_align],axis=1),sz_norm=5,delta=3,plt_val=False)
            txyz = np.array([tzy[0]]+list(txy))
            #print(txyz)
            from scipy import ndimage
            for icol in range(len(imf)):
                imBT = ndimage.shift(imb___[icol],txyz,mode='nearest',order=0)
                im_rat = ims___[icol]/imBT
                #im_rat = ims___[icol]-imBT*ratios[icol]
                im_rat = im_rat[:,padx_st:-padx_end,pady_st:-pady_end]

                imf[icol,:,startx__+padx_st:endx__-padx_end,starty__+pady_st:endy__-pady_end]=im_rat
                if False:
                    plt.figure()
                    plt.imshow(np.max((im_rat),0))
                    plt.figure()
                    plt.imshow(np.max((imb___[icol,:,pad:-pad,pad:-pad]),0))
    return imf

def get_txy_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    if sz_norm>0:
        im0 -= cv2.blur(im0,(sz_norm,sz_norm))
        im1 -= cv2.blur(im1,(sz_norm,sz_norm))
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    im_cor = convolve(im0[::-1,::-1],im1[delta:-delta,delta:-delta], mode='valid')
    #print(im_cor.shape)
    if plt_val:
        plt.figure()
        plt.imshow(im_cor)
    txy = np.array(np.unravel_index(np.argmax(im_cor), im_cor.shape))-delta
    return txy
def resize_slice(slices,shape0,shapef,fullz=True):
    slices_ = []
    for sl,sm,sM in zip(slices,shape0,shapef):
        start = sl.start*sM//sm
        end = sl.stop*sM//sm
        slices_.append(slice(start,end))
    if fullz:
        slices_[0]=slice(0,shapef[0])
    return tuple(slices_)

    
def get_txyz_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    im0 = norm_slice(im0,sz_norm)
    im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        #print(txyz)
    min_ = np.array(im0.shape)-delta
    min_[min_<0]=0
    max_ = np.array(im0.shape)+delta+1
    im_cor-=np.min(im_cor)
    im_cor[tuple([slice(m,M,None)for m,M in zip(min_,max_)])]*=-1
    txyz = np.unravel_index(np.argmin(im_cor), im_cor.shape)-np.array(im0.shape)+1
    #txyz = np.unravel_index(np.argmax(im_cor_),im_cor_.shape)+delta_
    return txyz

def get_txyz_small(im0_,im1_,sz_norm=10,plt_val=False,return_cor=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    if sz_norm>0:
        im0 = norm_slice(im0,sz_norm)
        im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im0/=np.std(im0)
    im1-=np.mean(im1)
    im1/=np.std(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    
        #print(txyz)
    imax = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    cor = im_cor[tuple(imax)]/np.prod(im0.shape)
    txyz = imax-np.array(im0.shape)+1
    if plt_val:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        import napari
        from scipy.ndimage import shift
        viewer = napari.view_image(im0)
        viewer.add_image(shift(im1,-txyz,mode='nearest'))
    
    if return_cor:
        return txyz,cor
    return txyz


def get_local_max(im_dif,th_fit,im_raw=None,dic_psf=None,delta=1,delta_fit=3,dbscan=True,return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5):
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    (This is important if saturating the camera values.)
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[],[]]
        Xft = []
        
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                        if im_raw is not None:
                            im_centers[4].append(im_raw[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                        Xft.append([d1,d2,d3])
              
        Xft = np.array(Xft)
        im_centers_ = np.array(im_centers)
        bk = np.min(im_centers_[3],axis=0)
        im_centers_[3] -= bk
        a = np.sum(im_centers_[3],axis=0)
        habs = np.zeros_like(bk)
        if im_raw is not None:
            habs = im_raw[z%zmax,x%xmax,y%ymax]
          
        if dic_psf is not None:
            keys = list(dic_psf.keys())
            ### calculate spacing
            im0 = dic_psf[keys[0]]
            space = np.sort(np.diff(keys,axis=0).ravel())
            space = space[space!=0][0]
            ### convert to reduced space
            zi,xi,yi = (z/space).astype(int),(x/space).astype(int),(y/space).astype(int)

            keys_ =  np.array(keys)
            sz_ = list(np.max(keys_//space,axis=0)+1)

            ind_ = tuple(Xft.T+np.array(im0.shape)[:,np.newaxis]//2-1)

            im_psf = np.zeros(sz_+[len(ind_[0])])
            for key in keys_:
                coord = tuple((key/space).astype(int))
                im__ = dic_psf[tuple(key)][ind_]
                im_psf[coord]=(im__-np.mean(im__))/np.std(im__)
            im_psf_ = im_psf[zi,xi,yi]
            im_centers__ = im_centers_[3].T.copy()
            im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
            hn = np.mean(im_centers__*im_psf_,axis=-1)
        else:
            

            #im_sm = im_[tuple([slice(x_-sz,x_+sz+1) for x_ in Xc])]
            sz = delta_fit
            Xft = (np.indices([2*sz+1]*3)-sz).reshape([3,-1]).T
            Xft = Xft[np.linalg.norm(Xft,axis=1)<=sz]
            
            sigma = np.array([sigmaZ,sigmaXY,sigmaXY])[np.newaxis]
            Xft_ = Xft/sigma
            norm_G = np.exp(-np.sum(Xft_*Xft_,axis=-1)/2.)
            norm_G=(norm_G-np.mean(norm_G))/np.std(norm_G)
            im_centers__ = im_centers_[3].T.copy()
            im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
            hn = np.mean(im_centers__*norm_G,axis=-1)
        
        zc = np.sum(im_centers_[0]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        Xh = np.array([zc,xc,yc,bk,a,habs,hn,h]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh
    

def _wiener_3d(self, image):
    """Monkey pathc to compute the 3D wiener deconvolution

    Parameters
    ----------
    image: torch.Tensor
        3D image tensor

    Returns
    -------
    torch.Tensor of the 3D deblurred image

    """
    import torch
    from sdeconv.core import SSettings
    from sdeconv.deconv.wiener import pad_3d,laplacian_3d,unpad_3d
    
    image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

    fft_source = torch.fft.fftn(image_pad)
    
    container = SSettings.instance()
    if not hasattr(container,'dic_psf'): container.dic_psf = {}
    if container.dic_psf.get('den'+str(fft_source.shape),None) is None:        
        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

        fft_psf = torch.fft.fftn(psf_roll)
        fft_laplacian = torch.fft.fftn(laplacian_3d(image_pad.shape))
        den = fft_psf * torch.conj(fft_psf) + self.beta * fft_laplacian * torch.conj(fft_laplacian)
        
        
        
        container.dic_psf['den'+str(fft_source.shape)] = den
        container.dic_psf['fft_psf'+str(fft_source.shape)] = fft_psf
    else:
        den = container.dic_psf['den'+str(fft_source.shape)].to(SSettings.instance().device)
        fft_psf = container.dic_psf['fft_psf'+str(fft_source.shape)].to(SSettings.instance().device)
    
    out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf)) / den))
    if image_pad.shape != image.shape:
        return unpad_3d(out_image, padding)
    return out_image


def apply_deconv(imsm,psf=None,plt_val=False,parameters = {'method':'wiener','beta':0.001,'niter':50},gpu=False,force=False,pad=None):
    r"""Applies deconvolution to image <imsm> using sdeconv: https://github.com/sylvainprigent/sdeconv/
    Currently assumes 60x objective with ~1.4 NA using SPSFGibsonLanni. Should be modified to find 
    
    Recomendations: the default wiener method with a low beta is the best for very fast local fitting. Albeit larger scale artifacts.
    For images: recommend the lucy method with ~30 iterations.
    
    This wraps around pytoch.
    
    To install:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    pip install sdeconv
    Optional: decided to modify the __init__ file of the SSettingsContainer in 
    C:\Users\BintuLabUser\anaconda3\envs\cellpose\Lib\site-packages\sdeconv\core\_settings.py
    
    import os
    gpu = True
    if os.path.exists("use_gpu.txt"):
        gpu = eval(open("use_gpu.txt").read())
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    to toggle the GPU on or off. By default it just uses the GPU if GPU detected by pytorch"""
    
    #import sdeconv,os
    #fl = os.path.dirname(sdeconv.__file__)+os.sep+'core'+os.sep+'use_gpu.txt'
    #fid = open(fl,'w')
    #fid.write('True')
    #fid.close()
    import torch
    from sdeconv.core import SSettings
    obj = SSettings.instance()
    obj.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    if force:
        if hasattr(obj,'dic_psf'): del obj.dic_psf
    # map to tensor
    
    imsm_ = torch.from_numpy(np.array(imsm,dtype=np.float32))
    if psf is None:
        from sdeconv.psfs import SPSFGibsonLanni
        #psf_generator = SPSFGaussian((1,1.5, 1.5), imsm_.shape)
        psf_generator = SPSFGibsonLanni(M=60,shape=imsm_.shape)
        psf = psf_generator().to(obj.device)
    else:
        psff = np.zeros(imsm_.shape,dtype=np.float32)
                
        slices = [(slice((s_psff-s_psf_full_)//2,(s_psff+s_psf_full_)//2),slice(None)) if s_psff>s_psf_full_ else
         (slice(None),slice((s_psf_full_-s_psff)//2,(s_psf_full_+s_psff)//2))
          
          for s_psff,s_psf_full_ in zip(psff.shape,psf.shape)]
        sl_psff,sl_psf_full_ = list(zip(*slices))
        psff[sl_psff]=psf[sl_psf_full_]
        psf = torch.from_numpy(np.array(psff,dtype=np.float32)).to(obj.device)
        
    method = parameters.get('method','wiener')
    if pad is None:
        pad = int(np.min(list(np.array(imsm.shape)-1)+[50]))
    if method=='wiener':
        from sdeconv.deconv import SWiener
        beta = parameters.get('beta',0.001)
        filter_ = SWiener(psf, beta=beta, pad=pad)
        #monkey patch _wiener_3d to allow recycling the fft of the psf components
        filter_._wiener_3d = _wiener_3d.__get__(filter_, SWiener)
    elif method=='lucy':
        from sdeconv.deconv import SRichardsonLucy
        niter = parameters.get('niter',50)
        filter_ = SRichardsonLucy(psf, niter=niter, pad=pad)
    elif method=='spitfire':
        from sdeconv.deconv import Spitfire
        filter_ = Spitfire(psf, weight=0.6, reg=0.995, gradient_step=0.01, precision=1e-6, pad=pad)
    
    out_image = filter_(imsm_)
     
    #out_image = out_image.cpu().detach().numpy().astype(np.float32)
    out_image_ = out_image.cpu().detach().numpy().astype(np.float32)
    if plt_val:
        import napari
        viewer = napari.view_image(out_image_)
        viewer.add_image(imsm)
    return out_image_
def get_local_maxfast(im_dif,th_fit,im_raw=None,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5):
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    def get_ind(x_,xmax):
        # modify x_ to be within image
        bad = x_>=xmax
        x_[bad]=xmax-x_[bad]-1
        bad = x_<0
        x_[bad]=-x_[bad]
        return x_
        
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta*delta):
                    z_ = get_ind(z+d1,zmax)
                    x_ = get_ind(x+d2,xmax)
                    y_ = get_ind(y+d3,ymax)
                    keep = im_dif[z,x,y]>=im_dif[z_,x_,y_]
                    z,x,y = z[keep],x[keep],y[keep]
    h = im_dif[z,x,y]
    Xh = np.array([z,x,y,h]).T

    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[],[]]
        Xft = []
        
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        z_ = get_ind(z+d1,zmax)
                        x_ = get_ind(x+d2,xmax)
                        y_ = get_ind(y+d3,ymax)
                        im_centers[3].append(im_dif[z_,x_,y_])
                        if im_raw is not None:
                            if False:
                                z_ = get_ind(z+d1-1,zmax)## weiner adds a shift of 1
                                x_ = get_ind(x+d2-1,xmax)
                                y_ = get_ind(y+d3-1,ymax)
                            im_centers[4].append(im_raw[z_,x_,y_])
                        else:
                            im_centers[4].append(im_dif[z_,x_,y_])
                        Xft.append([d1,d2,d3])

        Xft = np.array(Xft)
        im_centers = np.array(im_centers)
        bk = np.min(im_centers[3],axis=0)
        im_centers[3] -= bk
        #a = np.sum(im_centers[3],axis=0)
        habs = np.zeros_like(bk)
        a= habs
        if im_raw is not None:
            habs = im_raw[z,x,y]
        if dic_psf is None:
            ### Compute correlation with gaussian PSF
            sigma = np.array([sigmaZ,sigmaXY,sigmaXY])[np.newaxis]
            Xft_ = Xft/sigma
            norm_G = np.exp(-np.sum(Xft_*Xft_,axis=-1)/2.)
            norm_G=(norm_G-np.mean(norm_G))/np.std(norm_G)
            
            im_centers__ = im_centers[3].T.copy()
            im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
            hn = np.mean(im_centers__*norm_G,axis=-1)## This is the crosscorealtion
            im_centers__ = im_centers[4].T.copy()
            im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
            a = np.mean(im_centers__*norm_G,axis=-1)
            
        zc = np.sum(im_centers[0]*im_centers[3],axis=0)/np.sum(im_centers[3],axis=0)
        xc = np.sum(im_centers[1]*im_centers[3],axis=0)/np.sum(im_centers[3],axis=0)
        yc = np.sum(im_centers[2]*im_centers[3],axis=0)/np.sum(im_centers[3],axis=0)
        Xh = np.array([zc,xc,yc,bk,a,habs,hn,h]).T
    return Xh
def get_local_maxfast_tensor(im_dif_npy,th_fit=500,im_raw=None,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False):
    import torch
    dev = "cuda:0" if (torch.cuda.is_available() and gpu) else "cpu"
    im_dif = torch.from_numpy(im_dif_npy).to(dev)
    z,x,y = torch.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    def get_ind(x,xmax):
        # modify x_ to be within image
        x_ = torch.clone(x)
        bad = x_>=xmax
        x_[bad]=xmax-x_[bad]-2
        bad = x_<0
        x_[bad]=-x_[bad]
        return x_
    #def get_ind(x,xmax):return x%xmax
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta*delta):
                    z_ = get_ind(z+d1,zmax)
                    x_ = get_ind(x+d2,xmax)
                    y_ = get_ind(y+d3,ymax)
                    keep = im_dif[z,x,y]>=im_dif[z_,x_,y_]
                    z,x,y = z[keep],x[keep],y[keep]
    h = im_dif[z,x,y]
    
    
    if len(x)==0:
        return []
    if delta_fit>0:
        d1,d2,d3 = np.indices([2*delta_fit+1]*3).reshape([3,-1])-delta_fit
        kp = (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit)
        d1,d2,d3 = d1[kp],d2[kp],d3[kp]
        d1 = torch.from_numpy(d1).to(dev)
        d2 = torch.from_numpy(d2).to(dev)
        d3 = torch.from_numpy(d3).to(dev)
        im_centers0 = (z.reshape(-1, 1)+d1).T
        im_centers1 = (x.reshape(-1, 1)+d2).T
        im_centers2 = (y.reshape(-1, 1)+d3).T
        z_ = get_ind(im_centers0,zmax)
        x_ = get_ind(im_centers1,xmax)
        y_ = get_ind(im_centers2,ymax)
        im_centers3 = im_dif[z_,x_,y_]
        if im_raw is not None:
            im_raw_ = torch.from_numpy(im_raw).to(dev)
            im_centers4 = im_raw_[z_,x_,y_]
            habs = im_raw_[z,x,y]
        else:
            im_centers4 = im_dif[z_,x_,y_]
            habs = x*0
            a = x*0
        Xft = torch.stack([d1,d2,d3]).T
    
        bk = torch.min(im_centers3,0).values
        im_centers3 = im_centers3-bk
        im_centers3 = im_centers3/torch.sum(im_centers3,0)
        if dic_psf is None:
            sigma = torch.tensor([sigmaZ,sigmaXY,sigmaXY],dtype=torch.float32,device=dev)#np.array([sigmaZ,sigmaXY,sigmaXY],dtype=np.flaot32)[np.newaxis]
            Xft_ = Xft/sigma
            norm_G = torch.exp(-torch.sum(Xft_*Xft_,-1)/2.)
            norm_G=(norm_G-torch.mean(norm_G))/torch.std(norm_G)
    
            hn = torch.mean(((im_centers3-im_centers3.mean(0))/im_centers3.std(0))*norm_G.reshape(-1,1),0)
            a = torch.mean(((im_centers4-im_centers4.mean(0))/im_centers4.std(0))*norm_G.reshape(-1,1),0)
            
        zc = torch.sum(im_centers0*im_centers3,0)
        xc = torch.sum(im_centers1*im_centers3,0)
        yc = torch.sum(im_centers2*im_centers3,0)
        Xh = torch.stack([zc,xc,yc,bk,a,habs,hn,h]).T.cpu().detach().numpy()
    else:
        Xh =  torch.stack([z,x,y,h]).T.cpu().detach().numpy()
    return Xh
def get_local_max_tile(im_,th=2500,s_ = 512,pad=50,psf=None,plt_val=None,snorm=30,gpu=False,deconv={'method':'wiener','beta':0.001},
                        delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5):
    sx,sy = im_.shape[1:]
    ixys = []
    for ix in np.arange(0,sx,s_):
        for iy in np.arange(0,sy,s_):
            ixys.append([ix,iy])
    Xhf = None
    for ix,iy in ixys:#tqdm(ixys):
        imsm = im_[:,ix:ix+pad+s_,iy:iy+pad+s_]
        out_im = imsm
        if deconv is not None:
            out_im = apply_deconv(imsm,psf=psf,plt_val=False,parameters = deconv,gpu=gpu,force=True,pad=None)
        out_im2 = norm_slice(out_im,s=snorm)
        #print(time.time()-t)
        Xh = get_local_maxfast_tensor(out_im2,th,im_raw=imsm,dic_psf=None,delta=delta,delta_fit=delta_fit,sigmaZ=sigmaZ,sigmaXY=sigmaXY,gpu=gpu)
        ### exclude outside the padded area
        if Xh is not None:
            if len(Xh)>0:
                keep = np.all(Xh[:,1:3]<(s_+pad/2),axis=-1)
                keep &= np.all(Xh[:,1:3]>=(pad/2*np.array([ix>0,iy>0])),axis=-1)
                Xh = Xh[keep]
                Xh[:,1]+=ix
                Xh[:,2]+=iy
                #Xh[:,:3]-=1
                if Xhf is None: Xhf=Xh
                else: Xhf = np.concatenate([Xhf,Xh])
        #print(time.time()-t)
    if plt_val is not None:
        import napari
        v = napari.Viewer()
        #im__ = norm_slice(im_,s=30)
        v.add_image(im_,name='Original image')
        v.add_image(out_im2,name='Deconv image')
        H= Xhf[:,-1]
        size=None
        if type(plt_val) is dict:
            size = plt_val.get('size')
        if size is None:
            size = np.clip(H/np.percentile(H,99.99),0,1)*5
        v.add_points(Xhf[:,:3],face_color=[0,0,0,0],edge_color='y',size=size)
    return Xhf
from scipy.spatial.distance import cdist
def get_set(fl):
     if '_set' in fl: 
        return int(fl.split('_set')[-1].split(os.sep)[0].split('_')[0])
     else:
        return 0
from dask.array import concatenate
def concat(ims):
    shape = np.min([im.shape for im in ims],axis=0)
    ims_ = []
    for im in ims:
        shape_ = im.shape
        tupl = tuple([slice((sh_-sh)//2, -(sh_-sh)//2 if sh_>sh else None) for sh,sh_ in zip(shape,shape_)])
        ims_.append(im[tupl][np.newaxis])
    
    return concatenate(ims_)
class analysis_smFISH():
    def __init__(self,data_folders = [r'X:\DCBB_human__11_18_2022'],
                 save_folder = r'X:\DCBB_human__11_18_2022_Analysis',
                 H0folder=  r'X:\DCBB_human__11_18_2022\H0*',exclude_H0=True):
        self.Qfolders = [fld for data_folder in data_folders 
                             for fld in glob.glob(data_folder+os.sep+'H*')]
        self.H0folders = glob.glob(H0folder)
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        if exclude_H0:
            self.Qfolders = [fld for fld in self.Qfolders if fld not in self.H0folders]
        self.fls_bk = np.sort([fl for H0fld in self.H0folders for fl in glob.glob(H0fld+os.sep+'*.zarr')])
        print("Found files:"+str(len(self.fls_bk)))
        print("Found hybe folders:"+str(len(self.Qfolders)))
    def set_set(self,set_=''):
        self.set_ = set_
        self.fls_bk_ = [fl for fl in self.fls_bk if set_ in fl]
    def set_fov(self,ifl,set_=None):
        if set_ is not None:
            self.set_set(set_)
        self.fl_bk = self.fls_bk_[ifl]
    def set_hybe(self,iQ):
        self.Qfolder = [qfld for qfld in self.Qfolders if self.set_ in qfld][iQ]
        self.fl = self.Qfolder+os.sep+os.path.basename(self.fl_bk)
    def get_background(self,force=False):
        ### define H0
        print('### define H0 and load background')
        if not (getattr(self,'previous_fl_bk','None')==self.fl_bk) or force:
            print("Background file: "+self.fl_bk)
            path0 =  self.fl_bk
            im0,x0,y0=read_im(path0,return_pos=True)
            self.im_bk_ = np.array(im0,dtype=np.float32)
            self.previous_fl_bk = self.fl_bk
    def get_signal(self):
        print('### load signal')
        print("Signal file: "+self.fl)
        path =  self.fl
        im,x,y=read_im(path,return_pos=True)
        self.ncols,self.szz,self.szx,self.szy = im.shape
        self.im_sig_ = np.array(im,dtype=np.float32)
    def compute_drift(self,sz=200):
        im0 = self.im_bk_[-1]
        im = self.im_sig_[-1]
        txyz,txyzs = get_txyz(im0,im,sz_norm=40,sz = sz,nelems=5,plt_val=False)
        self.txyz,self.txyzs=txyz,txyzs
        self.dic_drift = {'txyz':self.txyz,'Ds':self.txyzs,'drift_fl':self.fl_bk}
        print("Found drift:"+str(self.txyz))
    def get_aligned_ims(self):
        txyz = self.txyz
        Tref = np.round(txyz).astype(int)
        slices_bk = tuple([slice(None,None,None)]+[slice(-t_,None,None) if t_<=0 else slice(None,-t_,None) for t_ in Tref])
        slices_sig = tuple([slice(None,None,None)]+[slice(t_,None,None) if t_>=0 else slice(None,t_,None) for t_ in Tref])
        self.im_sig__ = np.array(self.im_sig_[slices_sig],dtype=np.float32)
        self.im_bk__ = np.array(self.im_bk_[slices_bk],dtype=np.float32)
    def subtract_background(self,ssub=40,s=10,plt_val=False):
        print("Reducing background...")
        self.im_ratio = get_dif_or_ratio(self.im_sig__,self.im_bk__,sx=ssub,sy=ssub,pad=5,col_align=-2)
        self.im_ration = np.array([norm_slice(im_,s=s) for im_ in self.im_ratio])
        if plt_val:
            import napari
            napari.view_image(self.im_ration,contrast_limits=[0,0.7])
    def get_Xh_simple(self,th = 4,s=30,dic_psf=None,normalized=False):
        resc=  5
        self.Xhs = []
        for im_raw in self.im_sig_[:-1]:
            im_ = norm_slice(im_raw,s=s)
            th_ = np.std(im_[::resc,::resc,::resc])*th
            self.Xhs.append(get_local_max(im_,th_,im_raw=im_raw,dic_psf=dic_psf))
    def get_Xh(self,th = 5,s=30,dic_psf=None,subtract_bk=False,trim0=True,fr=1.25):
        """
        This fits each color image and saves the results in self.Xhs.
        It employs "get_local_max"
        """
        resc=  5
        self.Xhs = []
        self.plot_ims = []
        for icol in range(self.ncols-1):
            print("Fitting color "+str(icol))
            if subtract_bk:
                imsg = self.im_sig__[icol].astype(np.float32)
                imbk = self.im_bk__[icol].astype(np.float32)
                #fr = np.min([1.25,np.median(imsg[::resc,::resc,::resc]/imbk[::resc,::resc,::resc])])
                
                im_raw = imsg-imbk*fr
                im_raw = im_raw-np.median(im_raw)
                if trim0:
                    im_raw[im_raw<0]=0
            else:
                im_raw = self.im_sig_[icol]
            im_ = norm_slice(im_raw,s=s)
            std_=np.std(im_[::resc,::resc,::resc])
            #std_ = np.median(np.abs(im_[::resc,::resc,::resc] - np.median(im_[::resc,::resc,::resc])))
            #th_ = std_*th
            th_=th
            self.plot_ims.append(np.max(im_,0))
            self.Xhs.append(get_local_max(im_,th_,im_raw=im_raw,dic_psf=dic_psf))
        self.plot_ims = np.array(self.plot_ims)               
    def check_finished_file(self):
        file_sig = self.fl
        save_folder = self.save_folder
        fov_ = os.path.basename(file_sig).split('.')[0]
        hfld_ = os.path.basename(os.path.dirname(file_sig))
        self.base_save = self.save_folder+os.sep+fov_+'--'+hfld_
        self.Xh_fl = self.base_save+'--'+'_Xh_RNAs.pkl'
        return os.path.exists(self.Xh_fl)
    def save_fits(self,icols=None,plt_val=True,save_max=False):
        if plt_val:
            if icols is None:
                icols =  range(self.ncols-1)
            for icol in icols:

                fig = plt.figure(figsize=(40,40))
                if not hasattr(self,'dic_th'): self.dic_th={}
                vmax = self.dic_th.get(icol,3000)
                #std = np.std(self.plot_ims[icol])
                plt.imshow(self.plot_ims[icol],vmin=0,vmax=vmax,cmap='gray')
                #plt.show()
                fig.savefig(self.base_save+'_signal-col'+str(icol)+'.png')
                plt.close('all')
        if save_max:
            np.savez_compressed(self.base_save+'_plot_ims.npz',plot_ims = self.plot_ims)
        pickle.dump([self.Xhs,self.dic_drift],open(self.Xh_fl,'wb'))
        
def get_best_trans(Xh1,Xh2,th_h=1,th_dist = 2,return_pairs=False):
    mdelta = np.array([np.nan,np.nan,np.nan])
    if len(Xh1)==0 or len(Xh2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    X1,X2 = Xh1[:,:3],Xh2[:,:3]
    h1,h2 = Xh1[:,-1],Xh2[:,-1]
    i1 = np.where(h1>th_h)[0]
    i2 = np.where(h2>th_h)[0]
    if len(i1)==0 or len(i2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    i2_ = np.argmin(cdist(X1[i1],X2[i2]),axis=-1)
    i2 = i2[i2_]
    deltas = X1[i1]-X2[i2]
    dif_ = deltas
    bins = [np.arange(m,M+th_dist*2+1,th_dist*2) for m,M in zip(np.min(dif_,0),np.max(dif_,0))]
    hhist,bins_ = np.histogramdd(dif_,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    #plt.figure()
    #plt.imshow(np.max(hhist,0))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
    center_ = np.mean(dif_[keep],0)
    for i in range(5):
        keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
        center_ = np.mean(dif_[keep],0)
    mdelta = center_
    keep = np.all(np.abs(deltas-mdelta)<=th_dist,1)
    if return_pairs:
        return mdelta,Xh1[i1[keep]],Xh2[i2[keep]]
    return mdelta
    
def norm_im_med(im,im_med):
    if len(im_med)==2:
        return (im.astype(np.float32)-im_med[0])/im_med[1]
    else:
        return im.astype(np.float32)/im_med
def read_im(path,return_pos=False):
    import zarr,os
    from dask import array as da
    dirname = os.path.dirname(path)
    fov = os.path.basename(path).split('_')[-1].split('.')[0]
    #print("Bogdan path:",path)
    file_ = dirname+os.sep+fov+os.sep+'data'
    #image = zarr.load(file_)[1:]
    image = da.from_zarr(file_)[1:]

    shape = image.shape
    #nchannels = 4
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<z_offsets type="string">'
        zstack = txt.split(tag)[-1].split('</')[0]
        
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
        
        nchannels = int(zstack.split(':')[-1])
        nzs = (shape[0]//nchannels)*nchannels
        image = image[:nzs].reshape([shape[0]//nchannels,nchannels,shape[-2],shape[-1]])
        image = image.swapaxes(0,1)
    shape = image.shape
    if return_pos:
        return image,x,y
    return image



def linear_flat_correction(ims,fl=None,reshape=True,resample=4,vec=[0.1,0.15,0.25,0.5,0.75,0.9]):
    #correct image as (im-bM[1])/bM[0]
    #ims=np.array(ims)
    if reshape:
        ims_pix = np.reshape(ims,[ims.shape[0]*ims.shape[1],ims.shape[2],ims.shape[3]])
    else:
        ims_pix = np.array(ims[::resample])
    ims_pix_sort = np.sort(ims_pix[::resample],axis=0)
    ims_perc = np.array([ims_pix_sort[int(frac*len(ims_pix_sort))] for frac in vec])
    i1,i2=np.array(np.array(ims_perc.shape)[1:]/2,dtype=int)
    x = ims_perc[:,i1,i2]
    X = np.array([x,np.ones(len(x))]).T
    y=ims_perc
    a = np.linalg.inv(np.dot(X.T,X))
    cM = np.swapaxes(np.dot(X.T,np.swapaxes(y,0,-2)),-2,1)
    bM = np.swapaxes(np.dot(a,np.swapaxes(cM,0,-2)),-2,1)
    if fl is not None:
        folder = os.path.dirname(fl)
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(bM,open(fl,'wb'))
    return bM 
def compose_mosaic(ims,xs_um,ys_um,ims_c=None,um_per_pix=0.108333,rot = 0,return_coords=False):
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    sx,sy = szs[-2],szs[-1]
    ### Apply rotation:
    theta=-np.deg2rad(rot)
    xs_um_ = np.array(xs_um)*np.cos(theta)-np.array(ys_um)*np.sin(theta)
    ys_um_ = np.array(ys_um)*np.cos(theta)+np.array(xs_um)*np.sin(theta)
    ### Calculate per pixel
    xs_pix = np.array(xs_um_)/um_per_pix
    xs_pix = np.array(xs_pix-np.min(xs_pix),dtype=int)
    ys_pix = np.array(ys_um_)/um_per_pix
    ys_pix = np.array(ys_pix-np.min(ys_pix),dtype=int)
    sx_big = np.max(xs_pix)+sx+1
    sy_big = np.max(ys_pix)+sy+1
    dim = [sx_big,sy_big]
    if len(szs)==3:
        dim = [szs[0],sx_big,sy_big]

    if ims_c is None:
        if len(ims)>25:
            try:
                ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5,0.65,0.75,0.9])
            except:
                imc_c = np.median(ims,axis=0)
        else:
            ims_c = np.median(ims,axis=0)

    im_big = np.zeros(dim,dtype = dtype)
    sh_ = np.nan
    for i,(im_,x_,y_) in enumerate(zip(ims,xs_pix,ys_pix)):
        if ims_c is not None:
            if len(ims_c)==2:
                im_coef,im_inters = np.array(ims_c,dtype = 'float32')
                im__=(np.array(im_,dtype = 'float32')-im_inters)/im_coef
            else:
                ims_c_ = np.array(ims_c,dtype = 'float32')
                im__=np.array(im_,dtype = 'float32')/ims_c_*np.median(ims_c_)
        else:
            im__=np.array(im_,dtype = 'float32')
        im__ = np.array(im__,dtype = dtype)
        im_big[...,x_:x_+sx,y_:y_+sy]=im__
        sh_ = im__.shape
    if return_coords:
        return im_big,xs_pix+sh_[-2]/2,ys_pix+sh_[-1]/2
    return im_big
import cv2

def get_tiles(im_3d,size=256,delete_edges=False):
    sz,sx,sy = im_3d.shape
    if not delete_edges:
        Mz = int(np.ceil(sz/float(size)))
        Mx = int(np.ceil(sx/float(size)))
        My = int(np.ceil(sy/float(size)))
    else:
        Mz = np.max([1,int(sz/float(size))])
        Mx = np.max([1,int(sx/float(size))])
        My = np.max([1,int(sy/float(size))])
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic
def norm_slice(im,s=50):
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)

def fftconvolve_torch(in1_,in2_,gpu=True):
    
    import torch
    ifft,fft=torch.fft.irfftn,torch.fft.rfftn
    
    dev = "cuda:0" if (torch.cuda.is_available() and gpu) else "cpu"
    in1 = torch.from_numpy(in1_).to(dev)
    in2 = torch.from_numpy(in2_).to(dev)
    #print(in1.is_cpu)
    s1 = in1.shape
    s2 = in2.shape
    shape = [s1[i] + s2[i] - 1 for i in range(in1.ndim)]
    
    import scipy.fft as sp_fft
    complex_result=False
    fshape = [sp_fft.next_fast_len(shape[a], not complex_result) for a in [0,1,2]]
    
    sp1 = fft(in1, fshape)
    sp2 = fft(in2, fshape)
    ret = ifft(sp1 * sp2, fshape)
    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]
    return ret.cpu().detach().numpy()
def get_txyz(im_dapi0,im_dapi1,sz_norm=40,sz = 200,nelems=5,plt_val=False,gpu=False):
    """
    Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background (gaussian size sz_norm)
    and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
    It will return median value and a list of single values.
    """
    im_dapi0 = np.array(im_dapi0,dtype=np.float32)
    im_dapi1 = np.array(im_dapi1,dtype=np.float32)
    im_dapi0_ = norm_slice(im_dapi0,sz_norm)
    im_dapi1_ = norm_slice(im_dapi1,sz_norm)
    dic_ims0 = get_tiles(im_dapi0_,size=sz,delete_edges=True)
    dic_ims1 = get_tiles(im_dapi1_,size=sz,delete_edges=True)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    txyzs = []
    im_cors = []
    for ib in range(min(nelems,len(best))):
        im0 = dic_ims0[keys[best[ib]]][0].copy()
        im1 = dic_ims1[keys[best[ib]]][0].copy()
        im0-=np.mean(im0)
        im1-=np.mean(im1)
        from scipy.signal import fftconvolve
        #im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
        im0 = im0[::-1,::-1,::-1].copy()
        im_cor = fftconvolve_torch(im0,im1,gpu=gpu)
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im_cor,0))
            #print(txyz)
        txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
        
        im_cors.append(im_cor)
        txyzs.append(txyz)
    txyz = np.median(txyzs,0).astype(int)
    return txyz,txyzs
def expand_segmentation(im_segm_,nexpand=4):
    im_segm__ = im_segm_.copy()
    im_bw = im_segm_>0
    im_bwe = nd.binary_dilation(im_bw,iterations=nexpand)
    Xe = np.array(np.where(im_bwe>0)).T
    X = np.array(np.where(im_bw>0)).T
    Cell = im_segm_[tuple(X.T)]
    from scipy.spatial import KDTree
    inds = KDTree(X).query(Xe)[-1]
    im_segm__[tuple(Xe.T)]=Cell[inds]
    return im_segm__
class drift_refiner():
    def __init__(self,data_folder=r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022',
                 analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis'):
        """
        Example use:
        
        drift_  = drift_refiner()
        drift_.get_fov(0,'set1')
        Hs = len(drift_.raw_fls)
        ifovs = len(drift_.dapi_fls_)
        for ifov in tqdm(np.arange(ifovs)):
            drift_  = drift_refiner()
            drift_.get_fov(ifov,'set1')
            for iR in np.arange(Hs):
                analysis_folder_ = drift_.analysis_folder+os.sep+'distortion'
                if not os.path.exists(analysis_folder_):os.makedirs(analysis_folder_)
                fl = analysis_folder_+os.sep+os.path.basename(drift_.raw_fls[0]).split('.')[0]+'--'+drift_.set_+'--iR'+str(iR)+'.npy'
                if not os.path.exists(fl):
                    drift_.load_images(iR)
                    drift_.normalize_ims(zm=30,zM=50)
                    drift_.get_Tmed(sz_=300,th_cor=0.6,nkeep=9)
                    try:
                        P1_,P2_ = drift_.get_P1_P2_plus();
                        P1__,P2__ = drift_.get_P1_P2_minus();
                        P1f,P2f = np.concatenate([P1_,P1__]),np.concatenate([P2_,P2__])
                    except:
                        P1f,P2f = [],[]

                    if False:
                        import napari
                        viewer = napari.view_image(drift_.im2n,name='im2',colormap='green')
                        viewer.add_image(drift_.im1n,name='im1',colormap='red')
                        viewer.add_points(P2_,face_color='g',size=10)
                        viewer.add_points(P1_,face_color='r',size=10) 
                        drift_.check_transf(P1f,P2f)
                    try:
                        print("Error:",np.percentile(np.abs((P1f-P2f)-np.median(P1f-P2f,axis=0)),75,axis=0))
                        P1fT = drift_.get_Xwarp(P1f,P1f,P2f-P1f,nneigh=50,sgaus=20)
                        print("Error:",np.percentile(np.abs(P1fT-P2f),75,axis=0))
                    except:
                        pass

                    print(fl)
                    np.save(fl,np.array([P1f,P2f]))
        
        """         
                 
        
        self.data_folder = data_folder
        self.analysis_folder = analysis_folder
        self.dapi_fls = np.sort(glob.glob(analysis_folder+os.sep+'Segmentation'+os.sep+'*--dapi_segm.npz'))
    #adding additional method missing: get_XB
    def get_XB(self,im_,th=3):
        #im_ = self.im1n
        std_ = np.std(im_[::5,::5,::5])
        #im_base = im_[1:-1,1:-1,1:-1]
        #keep=(im_base>im_[:-2,1:-1,1:-1])&(im_base>im_[1:-1,:-2,1:-1])&(im_base>im_[1:-1,1:-1,:-2])&\
        #    (im_base>im_[2:,1:-1,1:-1])&(im_base>im_[1:-1,2:,1:-1])&(im_base>im_[1:-1,1:-1,2:])&(im_base>std_*3);#&(im_[:1]>=im_[1:])
        #XB = np.array(np.where(keep)).T+1

        keep = im_>std_*th
        XB = np.array(np.where(keep)).T
        from tqdm import tqdm
        for delta_fit in tqdm([1,2,3,5,7,10,15]):
            XI = np.indices([2*delta_fit+1]*3)-delta_fit
            keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
            XI = XI[:,keep].T
            XS = (XB[:,np.newaxis]+XI[np.newaxis])
            shape = self.sh
            XS = XS%shape

            keep = im_[tuple(XB.T)]>=np.max(im_[tuple(XS.T)],axis=0)
            XB = XB[keep]
        return XB
    def get_fov(self,ifov=10,set_='set1',keepH = ['H'+str(i)+'_' for i in range(1,9)]):
        self.set_ = set_
        self.ifov = ifov
        self.dapi_fls_ = [fl for fl in self.dapi_fls if set_+'-' in os.path.basename(fl)]
        self.dapi_fl = self.dapi_fls_[ifov]
        
        fov = os.path.basename(self.dapi_fl).split('--')[0]
        
        self.allHfolders = glob.glob(self.data_folder+os.sep+'H*')
        self.raw_fls = [[fld+os.sep+fov+'.zarr' for fld in self.allHfolders 
                         if (tag in os.path.basename(fld)) and (self.set_ in os.path.basename(fld))][0] for tag in keepH]
    def load_segmentation(self):
        dapi_fl  = self.dapi_fl
        im_segm = np.load(dapi_fl)['segm']
        shape = np.load(dapi_fl)['shape']
        cellcaps = [resize_slice(pair,im_segm.shape,shape) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None]
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in cellcaps])
        self.cellcaps=cellcaps
        self.cm_cells=cm_cells
    def load_images(self,ifl2,ifl1=None,icol=-1):
        if ifl1 is None:
            ifl1 = len(self.raw_fls)//2
        self.reloadfl1=True
        if hasattr(self,'im1'):
            if self.fl1==self.raw_fls[ifl1]:
                self.reloadfl1=False
        
        self.fl1,self.fl2 = self.raw_fls[ifl1],self.raw_fls[ifl2]
        print("Loading images:",self.fl1,self.fl2)
        
        if self.reloadfl1:
            self.im1 = np.array(read_im(self.fl1)[icol],np.float32)
        self.im2 = np.array(read_im(self.fl2)[icol],np.float32)
        
        self.sh = np.array(self.im1.shape)
    def normalize_ims(self,zm=5,zM=50):
        im01 = self.im1
        if self.reloadfl1 or not hasattr(self,'im1n'):
            self.im1n = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im01])
        im02 = self.im2
        self.im2n = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im02])
    def apply_drift(self,cell,Tmed,sh=None):
        if sh is None:
            sh = np.array(self.im1.shape)
        xm,xM = np.array([(sl.start,sl.stop)for sl in cell]).T
        xm1,xM1 = xm-Tmed,xM-Tmed
        xm2,xM2 = xm,xM
        bad = xm1<0
        xm2[bad]=xm2[bad]-xm1[bad]
        xm1[bad]=0
        bad = xM2>sh
        xM1[bad]=xM1[bad]-(xM2-sh)[bad]
        xM2[bad]=sh[bad]
        return tuple([slice(x_,x__) for x_,x__ in zip(xm1,xM1)]),tuple([slice(x_,x__) for x_,x__ in zip(xm2,xM2)])
    
    def get_cell_caps(self,sz_ = 40):
        sh = self.sh
        szz,szx,szy = np.ceil(sh/sz_).astype(int)
        cellcaps = [(slice(iz*sz_,(iz+1)*sz_),slice(ix*sz_,(ix+1)*sz_),slice(iy*sz_,(iy+1)*sz_))
                      for ix in range(szx) for iy in range(szy) for iz in range(szz)]
        return cellcaps
    def filter_cor(self,P1,h1,P2,h2,cor_th=0.75):
        h1_ = h1.copy().T
        h1_ = h1_-np.nanmean(h1_,axis=0)
        h1_ = h1_/np.nanstd(h1_,axis=0)

        h2_ = h2.copy().T
        h2_ = h2_-np.nanmean(h2_,axis=0)
        h2_ = h2_/np.nanstd(h2_,axis=0)
        cors = np.nanmean(h1_*h2_,axis=0)
        keep=cors>cor_th
        P1_ = P1[keep]
        P2_ = P2[keep]
        return P1_,P2_


    def get_Xwarp(self,x_ch,X,T,nneigh=10,sgaus=20,szero=10):
        #X,T = cm_cells[keep],txyzs[keep]
        #T = T+Tmed
        from scipy.spatial import cKDTree
        tree = cKDTree(X)


        dists,inds = tree.query(x_ch,nneigh);
        ss=sgaus
        Tch = T[inds].copy()
        #Tch[:,-1]=0
        #dists[:,-1] = ss*szero
        M = np.exp(-dists*dists/(2*ss*ss))
        TF = np.sum(Tch*M[...,np.newaxis],axis=1)/np.sum(M,axis=1)[...,np.newaxis]
        #TF = Tch[:,0]#np.median(Tch,axis=1)
        TF = np.round(TF).astype(int)


        XF = x_ch+TF
        XF[XF<0]=0
        sh = self.sh
        for idim in range(len(sh)):
            XF[XF[:,idim]>=sh[idim],idim]=sh[idim]-1

        return XF
    def get_Tmed(self,sz_=300,th_cor=0.75,nkeep=5):
        """Assuming that self.imn1 and self.imn2 are loaded and normalized, this takes """
        cellcaps = self.get_cell_caps(sz_ = sz_)#self.cellcaps
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in cellcaps])
        txyzs,cors = [],[]
        icells = np.argsort([np.std(self.im1n[cell]) for cell in cellcaps])[::-1][:nkeep]
        for icell in tqdm(icells):
            cell = cellcaps[icell]
            imsm1,imsm2 = self.im1n[cell],self.im2n[cell]
            txyz,cor = get_txyz_small(imsm1,imsm2,sz_norm=0,plt_val=False,return_cor=True)
            txyzs.append(txyz)
            cors.append(cor)
        cors = np.array(cors)
        txyzs = np.array(txyzs)
        keep = cors>th_cor
        print("Keeping fraction of cells: ",np.mean(keep))
        self.Ts = txyzs[keep]
        self.Tmed = np.median(txyzs[keep],axis=0).astype(int)
        
    def get_max_min(self,P,imn,delta_fit=5,ismax=True,return_ims=False):
        XI = np.indices([2*delta_fit+1]*3)-delta_fit
        keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
        XI = XI[:,keep].T
        XS = (P[:,np.newaxis]+XI[np.newaxis])
        shape = self.sh
        XSS = XS.copy()
        XS = XS%shape
        #XSS = XS.copy()
        is_bad = np.any((XSS!=XS),axis=-1)


        sh_ = XS.shape
        XS = XS.reshape([-1,3])
        im1n_local = imn[tuple(XS.T)].reshape(sh_[:-1])
        #print(is_bad.shape,im1n_local.shape)
        im1n_local[is_bad]=np.nan
        im1n_local_ = im1n_local.copy()
        im1n_local_[is_bad]=-np.inf
        XF = XSS[np.arange(len(XSS)),np.nanargmax(im1n_local_,axis=1)]
        #im1n_med = np.min(im1n_local,axis=1)[:,np.newaxis]
        #im1n_local_ = im1n_local.copy()
        #im1n_local_ = np.clip(im1n_local_-im1n_med,0,np.inf)
        if return_ims:
            return XF,im1n_local
        return XF
    def get_XB(self,im_,th=3):
        #im_ = self.im1n
        std_ = np.std(im_[::5,::5,::5])
        #im_base = im_[1:-1,1:-1,1:-1]
        #keep=(im_base>im_[:-2,1:-1,1:-1])&(im_base>im_[1:-1,:-2,1:-1])&(im_base>im_[1:-1,1:-1,:-2])&\
        #    (im_base>im_[2:,1:-1,1:-1])&(im_base>im_[1:-1,2:,1:-1])&(im_base>im_[1:-1,1:-1,2:])&(im_base>std_*3);#&(im_[:1]>=im_[1:])
        #XB = np.array(np.where(keep)).T+1

        keep = im_>std_*th
        XB = np.array(np.where(keep)).T
        from tqdm import tqdm
        for delta_fit in tqdm([1,2,3,5,7,10,15]):
            XI = np.indices([2*delta_fit+1]*3)-delta_fit
            keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
            XI = XI[:,keep].T
            XS = (XB[:,np.newaxis]+XI[np.newaxis])
            shape = self.sh
            XS = XS%shape

            keep = im_[tuple(XB.T)]>=np.max(im_[tuple(XS.T)],axis=0)
            XB = XB[keep]
        return XB
    def get_P1_P2_plus(self):
        if self.reloadfl1 or not hasattr(self,'P1_plus'):
            P10 = self.get_XB(self.im1n,th=3)
            P1,h1 = self.get_max_min(P10,self.im1n,delta_fit=15,ismax=True,return_ims=True)
            P1,h1 = self.get_max_min(P1,self.im1n,delta_fit=7,ismax=True,return_ims=True)
            self.P1_plus,self.h1_plus = P1,h1
        P1,h1 = self.P1_plus,self.h1_plus
        Tmed = self.Tmed.astype(int)
        P20 = P1+Tmed
        P2,h2 = self.get_max_min(P20,self.im2n,delta_fit=15,ismax=True,return_ims=True)
        P2,h2 = self.get_max_min(P2,self.im2n,delta_fit=7,ismax=True,return_ims=True)
        P1_,P2_ = self.filter_cor(P1,h1,P2,h2,cor_th=0.75)
        print(len(P1_)/len(P1))
        return P1_,P2_
    def get_P1_P2_minus(self):
        if self.reloadfl1 or not hasattr(self,'P1_minus'):
            P10 = self.get_XB(-self.im1n,th=2)
            P1,h1 = self.get_max_min(P10,-self.im1n,delta_fit=15,ismax=True,return_ims=True)
            P1,h1 = self.get_max_min(P1,-self.im1n,delta_fit=7,ismax=True,return_ims=True)
            self.P1_minus,self.h1_minus = P1,h1
        P1,h1 = self.P1_minus,self.h1_minus
        Tmed = self.Tmed.astype(int)
        P20 = P1+Tmed
        P2,h2 = self.get_max_min(P20,-self.im2n,delta_fit=15,ismax=True,return_ims=True)
        P2,h2 = self.get_max_min(P2,-self.im2n,delta_fit=7,ismax=True,return_ims=True)
        #P2 = get_max_min(self,P2,self.im2n,delta_fit=10,ismax=True)
        #P2 = get_max_min(self,P2,self.im2n,delta_fit=5,ismax=True)
        P1_,P2_ = self.filter_cor(P1,h1,P2,h2,cor_th=0.75)
        print(len(P1_)/len(P1))
        return P1_,P2_
    def check_transf(self,P1_,P2_,nneigh=30,sgaus=20):
        shape = self.sh
        Tmed = self.Tmed.astype(int)
        X_,Y_,Z_=np.indices([1,shape[1],shape[2]])
        X_+=shape[0]//2
        x_ch = np.array([X_,Y_,Z_]).reshape([3,-1]).T
        X=P1_
        T=P2_-P1_
        XF = self.get_Xwarp(x_ch,X,T,nneigh=nneigh,sgaus=sgaus)
        im2_ = self.im2n[tuple(XF.T)].reshape([shape[1],shape[2]])
        im1_ = self.im1n[tuple(x_ch.T)].reshape([shape[1],shape[2]])
        import napari
        from scipy.ndimage import shift
        viewer=napari.view_image(np.array([shift(self.im2n[shape[0]//2+Tmed[0]],-Tmed[1:]),im1_,im2_]))
        viewer.add_points(P1_[:,1:],face_color='g',size=10)
        P2__ = P2_+np.median(P1_-P2_,axis=0)
        viewer.add_points(P2__[:,1:],face_color='r',size=10)
        
        
        
        
import glob,os,numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial import cKDTree
def get_all_pos(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',ext='.zarr',force=False):
    
    fl_pos = analysis_folder+os.sep+'pos_'+set_+'.pkl'
    if os.path.exists(fl_pos) and not force:
        dic_coords = pickle.load(open(fl_pos,'rb'))
    else:
        dic_coords = {}
        allflds = glob.glob(data_folder)
        for fld in allflds:
            if set_ in fld:
                for fl in tqdm(glob.glob(fld+os.sep+'*'+ext)):
                    dic_coords[get_ifov(fl)]=get_pos(fl)
        pickle.dump(dic_coords,open(fl_pos,'wb'))
    return dic_coords
def get_pos(path):
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    x,y=0,0
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
    return x,y
def get_ifov(fl):return int(os.path.basename(fl).split('--')[0].split('_')[-1].split('.')[0])
def get_H(fl):return int(os.path.basename(fl).split('--')[1].split('_')[0][1:])
def get_iH_npy(fl): return int(os.path.basename(fl).split('--iR')[-1].split('.')[0])
def get_Xwarp(x_ch,X,T,nneigh=50,sgaus=100):
    from scipy.spatial import cKDTree
    tree = cKDTree(X)


    dists,inds = tree.query(x_ch,nneigh);
    dists = dists[:,:len(X)]
    inds = inds[:,:len(X)]
    ss=sgaus
    Tch = T[inds].copy()
    
    M = -dists*dists/(2*ss*ss)
    M = M-np.max(M,axis=-1)[...,np.newaxis]
    M = np.exp(M)
    #M = dists<sgaus
    TF = np.sum(Tch*M[...,np.newaxis],axis=1)/np.sum(M,axis=1)[...,np.newaxis]
    #bad = np.any(np.isnan(TF),axis=1)
    #TF[bad] = np.median(TF[~bad],axis=0)
    XF = x_ch+TF
    return XF
def compute_hybe_drift(dic_comp,npoint=50,ncols=3,color=1):
    """
    Given list of difference of points in dic_comp
    this will compute the best drift
    
    """
    
    iHs = list(np.unique([iH for iHjH in list(dic_comp.keys()) for iH in iHjH]))
    iHs = [iH for iH in iHs if (iH%ncols)==color]
    nH = len(iHs)
    #for i in range(len())
    a = [np.zeros(nH)]
    a[0][nH//2]=1
    b = [[0,0,0]]
    count=1
    
    for (iH,jH) in dic_comp:
        if (iH in iHs) and (jH in iHs):
            X = dic_comp[(iH,jH)]
            if len(X)>npoint:
                b_ = np.mean(X,axis=0)
                b.append(b_)
                arow = np.zeros(nH)
                iH_,jH_=iHs.index(iH),iHs.index(jH)
                arow[iH_],arow[jH_]=1,-1
                a.append(arow)
                count+=1
    a=np.array(a)
    b=np.array(b)
    res = np.linalg.lstsq(a,b)[0]
    drift_hybe = {iH:res[iH_]for iH_,iH in enumerate(iHs)}
    return drift_hybe
class decoder():
    def __init__(self,analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',force=False):
        """
        dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
        dec.get_set_ifov(ifov=2,set_='set1',keepH = [1,2,3,4,5,6,7,8],ncols=3)
        print("Loading the fitted molecules")
        dec.get_XH()
        print("Correcting for distortion acrossbits")
        dec.apply_distortion_correction()
        dec.load_library()
        dec.XH = dec.XH[dec.XH[:,-4]>0.25]
        dec.get_inters(dinstance_th=3)
        dec.pick_best_brightness(nUR_cutoff = 3,resample = 10000)
        dec.pick_best_score(nUR_cutoff = 3)
        """
        self.analysis_folder=analysis_folder
        
        self.files_map = self.analysis_folder+os.sep+'files_map.npz'
        if not os.path.exists(self.files_map) or force:
            self.remap_files()
        else:   
            result = np.load(self.files_map)
            self.files,self.dapi_fls = result['files'],result['dapi_fls']
            if len(self.files)==0: self.remap_files()
        self.save_folder = self.analysis_folder+os.sep+'Decoded'
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
    def remap_files(self):
        self.files=glob.glob(self.analysis_folder+os.sep+'*--_Xh_RNAs.pkl')
        self.dapi_fls = np.sort(glob.glob(self.analysis_folder+os.sep+'Segmentation'+os.sep+'*--dapi_segm.npz'))
        np.savez(self.files_map,files=self.files,dapi_fls=self.dapi_fls)
    def get_set_ifov(self,ifov=0,set_='',keepH = [1,2,3,4,5,6,7,8],ncols=3):
        """map all the complete files in self.dic_fls_final for the hybes H<*> in keepH.
        Put the files for fov <ifov> in self.fls_fov"""
        
        self.set_ = set_
        self.ifov = ifov
        
        self.keepH = keepH
        self.files_set = [fl for fl in self.files if (set_ in os.path.basename(fl))]
        self.ncols=ncols
        def refine_fls(fls,keepH):
            fls_ = [fl for fl in fls if get_H(fl) in keepH]
            Hs = [fl for fl in fls_ if get_H(fl)]
            fls_ = np.array(fls_)[np.argsort(Hs)]
            return fls_
        dic_fls = {}
        
        for fl in self.files_set:
            ifov_ = get_ifov(fl)
            if ifov_ not in dic_fls: dic_fls[ifov_]=[]
            dic_fls[ifov_].append(fl)
        
        
        dic_fls_final = {ifv:refine_fls(dic_fls[ifv],keepH) for ifv in dic_fls}
        self.dic_fls_final = {ifv:dic_fls_final[ifv] for ifv in np.sort(list(dic_fls_final.keys())) 
                              if len(dic_fls_final[ifv])==len(keepH)}
        self.fls_fov = self.dic_fls_final.get(ifov,[])
        self.is_complete=False
        self.out_of_range = False
        if len(self.fls_fov)>0:
            self.fov = os.path.basename(self.fls_fov[0]).split('--')[0]
            self.save_file_dec = self.save_folder+os.sep+self.fov+'--'+self.set_+'_decoded.npz'
            self.save_file_cts = self.save_folder+os.sep+self.fov+'--'+self.set_+'_cts.npz'
            if os.path.exists(self.save_file_cts):
                self.is_complete=True
        else:
            self.out_of_range = True
            self.is_complete=True
    def get_XH(self):
        """given self.fls_fov this loads each fitted file and keeps: zc,xc,yc,hn,h,icol into self.XH
        Also saves """
        self.drift_set=np.array([[0,0,0]for i_ in range(len(self.fls_fov)*self.ncols)])
        XH = np.zeros([0,7])
        for iH in tqdm(np.arange(len(self.fls_fov))):
            fl = self.fls_fov[iH]
            Xhs,dic_drift = pickle.load(open(fl,'rb'))
            #zc,xc,yc,bk,a,habs,hn,h
            for icol,Xh in enumerate(Xhs):
                X = Xh[:,:3]#-dic_drift['txyz']
                h = Xh[:,-2:]
                R = iH*len(Xhs)+icol
                self.drift_set[R]=dic_drift['txyz']
                icolRs = np.ones([len(X),2])
                icolRs[:,0]=icol
                icolRs[:,1]=R
                Xf = np.concatenate([X,h,icolRs],axis=-1)
                XH = np.concatenate([XH,Xf],axis=0)
        self.XH=XH
    def get_counts_per_cell(self,nbad=0):
        keep_good = np.sum(self.res_pruned==-1,axis=-1)<=nbad
        Xcms = self.Xcms[keep_good]
        icodes = self.icodes[keep_good]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        icodes = icodes[good]
        cts_all = []
        for ikeep in np.arange(len(self.gns_names)):
            Xred_ = Xred[icodes==ikeep]
            icells,cts = np.unique(self.im_segm[tuple(Xred_.T)],return_counts=True)
            dic_cts = {icell:ct for icell,ct in zip(icells,cts)}
            ctsf = [dic_cts.get(icell,0) for icell in self.icells]
            cts_all.append(ctsf)
        cts_all = np.array(cts_all)
        return cts_all
    def load_library(self,lib_fl = r'Z:\DCBBL1_3_2_2023\MERFISH_Analysis\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv',nblanks=-1):
        code_txt = np.array([ln.replace('\n','').split(',') for ln in open(lib_fl,'r') if ',' in ln])
        gns = code_txt[1:,0]
        code_01 = code_txt[1:,2:].astype(int)
        codes = np.array([np.where(cd)[0] for cd in code_01])
        codes_ = [list(np.sort(cd)) for cd in codes]
        nbits = np.max(codes)+1

        codes__ = codes_
        gns__ = list(gns)
        if nblanks>=0:
            ### get extrablanks
            from itertools import combinations
            X_codes = np.array((list(combinations(range(nbits),4))))
            X_code_01 = []
            for cd in X_codes:
                l_ = np.zeros(nbits)
                l_[cd] = 1
                X_code_01.append(l_)
            X_code_01 = np.array(X_code_01,dtype=int)
            from scipy.spatial.distance import cdist
            eblanks = np.where(np.min(cdist(code_01,X_code_01,metric='hamming'),0)>=4/float(nbits))[0]
            codes__ = codes_ + [list(e)for e in X_codes[eblanks]]
            gns__ = list(gns)+ ['blanke'+str(ign+1).zfill(4) for ign in range(len(eblanks))]
        
        bad_gns = np.array(['blank' in e for e in gns__])
        good_gns = np.where(~bad_gns)[0]
        bad_gns = np.where(bad_gns)[0]

        
        
        self.lib_fl = lib_fl ### name of coding library
        self.nbits = nbits ### number of bits
        self.gns_names = gns__  ### names of genes and blank codes
        self.bad_gns = bad_gns ### indices of the blank codes
        self.good_gns = good_gns ### indices of the good gene codes
        self.codes__ = codes__ ### final extended codes of form [bit1,bit2,bit3,bit4]
        self.codes_01 = code_01
        if nblanks>=0:
            self.codes_01 = np.concatenate([code_01,X_code_01[eblanks]],axis=0) ### final extended codes of form [0,1,0,0,1...]
        
        dic_bit_to_code = {}
        for icd,cd in enumerate(self.codes__): 
            for bit in cd:
                if bit not in dic_bit_to_code: dic_bit_to_code[bit]=[]
                dic_bit_to_code[bit].append(icd)
        self.dic_bit_to_code = dic_bit_to_code  ### a dictinary in which each bit is mapped to the inde of a code
    def get_inters(self,dinstance_th=2,enforce_color=False):
        """Get an initial intersection of points and save in self.res"""
        res =[]
        if enforce_color:
            icols = self.XH[:,-2]
            XH = self.XH
            for icol in tqdm(np.unique(icols)):
                inds = np.where(icols==icol)[0]
                Xs = XH[inds,:3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts,dinstance_th)
                res += [inds[r] for r in res_]
        else:
            XH = self.XH
            Xs = XH[:,:3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
        self.res = res
    def apply_distortion_correction(self):
        """
        This modifies self.XH to add the correction for distortion (and drift) for each hybe
        """
        fls_dist = glob.glob(self.analysis_folder+os.sep+'distortion\*.npy')
        fls_dist_ = np.sort([fl for fl in fls_dist if get_ifov(fl)==self.ifov and self.set_ in os.path.basename(fl)])
        self.dic_fl_distortion = {get_iH_npy(fl):fl for fl in fls_dist_}
        self.dic_pair = {}
        for iH in range(len(self.keepH)):
            Xf1,Xf2 = [],[]
            fl = self.dic_fl_distortion.get(iH,None)
            if fl is not None:
                Xf1,Xf2 = np.load(fl)
            for icol in range(self.ncols):
                self.dic_pair[iH*self.ncols+icol]=[Xf1,Xf2]

        if not hasattr(self,'XH_'):
            self.XH_ = self.XH.copy()
        self.XH = self.XH_.copy()
        for iR in tqdm(np.unique(self.XH[:,-1])):
            IR = int(iR)
            Xf1,Xf2= self.dic_pair[IR]
            Rs = self.XH[:,-1]
            keep = Rs==IR
            X = self.XH[keep,:3]
            if len(Xf1):
                XT = get_Xwarp(X,Xf2,Xf1-Xf2,nneigh=50,sgaus=100)
            else:
                XT = X-self.drift_set[IR]
            self.XH[keep,:3] = XT
            
            

    def pick_best_brightness(self,nUR_cutoff = 3,resample = 10000):
        """Pick the best code best on normalized brightness for a subsample of <resample> molecule-clusters.
        This is used to find the statistics for the distance and brightness distribution"""
        XH = self.XH
        res =self.res
        codes = self.codes__


        RS = XH[:,-1].astype(int)
        HS = XH[:,-3]
        colS = XH[:,-2].astype(int)
        colSu = np.unique(colS)
        #Ru_ = Ru[3]
        meds_col = np.array([np.median(HS[colS==cu]) for cu in colSu])
        self.meds_col = meds_col
        HN = HS/meds_col[colS]

        ncodes = len(codes)

        bucket = np.zeros(ncodes)
        nbits_per_code = np.array([len(cd) for cd in codes])

        icodes = []
        res_pruned = []

        Nres = len(res)
        resamp = max(Nres//resample,1)

        for r in tqdm(res[::resamp]):
            scores = HN[r]

            dic_sc = {r:sc for r,sc in zip(r,scores)}
            isort = np.argsort(scores)
            r = np.array(r)[isort]
            scores = scores[isort]
            R = RS[r]
            dic_u = {R_:r_ for r_,R_ in zip(r,R)}
            if len(dic_u)>=nUR_cutoff:
                bucket_ = np.zeros(ncodes)
                for R_ in dic_u:
                    if R_ in self.dic_bit_to_code:
                        icds = self.dic_bit_to_code[R_]
                        bucket_[icds]+=dic_sc[dic_u[R_]]
                bucket_/=nbits_per_code
                best_code = np.argmax(bucket_)
                icodes.append(best_code)
                res_pruned.append([dic_u.get(R_,-1) for R_ in codes[best_code]])
        self.res_pruned = res_pruned
        self.icodes = icodes
    def get_brightness_distance_distribution(self):
        XH = self.XH
        all_dists = []
        all_brightness = []
        for rs,icd in zip(tqdm(self.res_pruned),self.icodes):
            if icd in self.good_gns:
                rs = np.array(rs)
                rs = rs[rs>-1]
                X = XH[rs]
                h = X[:,-3]
                col = X[:,-2].astype(int)
                dists_ = np.linalg.norm(np.mean(X[:,:3],axis=0)-X[:,:3],axis=-1)
                all_dists.extend(dists_)
                all_brightness.extend(h/self.meds_col[col])
        all_brightness = np.sort(all_brightness)[:,np.newaxis]
        all_dists = np.sort(all_dists)[::-1,np.newaxis]
        self.tree_br = cKDTree(all_brightness)
        self.tree_dist = cKDTree(all_dists)
    def get_score_brightness(self,x):
        return (self.tree_br.query(x[:,np.newaxis])[-1]+1)/(len(self.tree_br.data)+1)
    def get_score_distance(self,x):
        return (self.tree_dist.query(x[:,np.newaxis])[-1]+1)/(len(self.tree_dist.data)+1)
    def pick_best_score(self,nUR_cutoff = 3,resample=1):
        """Pick the best code for each molecular cluster based on the fisher statistics 
        for the distance and brightness distribution"""
        self.get_brightness_distance_distribution()
        res =self.res
        XH = self.XH
        codes = self.codes__

        RS = XH[:,-1].astype(int)
        HS = XH[:,-3]
        colS = XH[:,-2].astype(int)
        colSu = np.unique(colS)
        meds_col = np.array([np.median(HS[colS==cu]) for cu in colSu])

        self.HN = HS/meds_col[colS]

        ncodes = len(codes)

        bucket = np.zeros(ncodes)
        nbits_per_code = np.array([len(cd) for cd in codes])

        icodes = []
        res_pruned = []
        scores_pruned = []
        for r in tqdm(res[::resample]):
            hn = self.HN[r]
            X = XH[r,:3]
            dn = np.linalg.norm(X-np.mean(X,axis=0),axis=-1)
            sc_dist = self.get_score_distance(dn)
            sc_br = self.get_score_brightness(hn)
            scores = sc_dist*sc_br
            dic_sc = {r:sc for r,sc in zip(r,scores)}
            isort = np.argsort(scores)
            r = np.array(r)[isort]
            #scores = scores[isort]
            R = RS[r]
            dic_u = {R_:r_ for r_,R_ in zip(r,R)}
            if len(dic_u)>=nUR_cutoff:
                bucket_ = np.zeros(ncodes)
                for R_ in dic_u:
                    if R_ in self.dic_bit_to_code:
                        icds = self.dic_bit_to_code[R_]
                        bucket_[icds]+=dic_sc[dic_u[R_]]
                bucket_/=nbits_per_code
                best_code = np.argmax(bucket_)
                icodes.append(best_code)
                rf = [dic_u.get(R_,-1) for R_ in codes[best_code]]
                res_pruned.append(rf)
                scores_pruned.append([dic_sc.get(r_,-1000) for r_ in rf])
        self.res_pruned = np.array(res_pruned)
        self.icodes = np.array(icodes)
        self.scores_pruned = np.array(scores_pruned)
        X1f,X2f = self.dic_pair.get(0,[[[0,0,0]],[[0,0,0]]])
        driftf = np.mean(np.array(X1f)-X2f,axis=0)
        X = self.XH[:,:3]-driftf
        res_ = np.array(self.res_pruned)
        keep = (res_>=0)[...,np.newaxis]
        self.Xcms = np.sum(X[res_]*keep,axis=1)/np.sum(keep,axis=1)
    def load_segmentation(self):
        dapi_fl  = [fl for fl in self.dapi_fls if self.set_ in os.path.basename(fl) and self.fov in os.path.basename(fl)][0]
        im_segm = np.load(dapi_fl)['segm']
        shape = np.load(dapi_fl)['shape']
        objs = ndimage.find_objects(im_segm)
        self.icells,self.cellcaps_,self.cellcaps = [],[],[]
        if len(objs): 
            self.icells,self.cellcaps_,self.cellcaps = zip(*[(icell+1,pair,resize_slice(pair,im_segm.shape,shape)) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None])
        #self.icells,self.cellcaps_,self.cellcaps = zip(*[(icell+1,pair,resize_slice(pair,im_segm.shape,shape)) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None])
        #self.icells = [icell+1 for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None]
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in self.cellcaps])
        #self.cellcaps=cellcaps
        self.cm_cells=cm_cells
        self.im_segm = im_segm
        self.shapesm = self.im_segm.shape
        self.shape = shape
        self.vols = [int(np.sum(self.im_segm[cell_cap]==icell)*np.prod(self.shape/self.shapesm)) for icell,cell_cap in zip(self.icells,self.cellcaps_)]
    def get_ptb_aso(self,icol_aso=0,icol_ptb=1,th_cor_ptb=0.5,th_ptb=2500):
        """
        This gets the ptb counts and the average aso level per cell assuming the data is already fitted
        use as:
        
        from ioMicro import *

        dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
        for set_ in ['set1','set2','set3','set4']:
            for ifov in tqdm(np.arange(400)):
                ### computation
                dec.get_set_ifov(ifov=ifov,set_=set_,keepH = [1,2,3,4,5,6,7,8],ncols=3)
                dec.save_file_cts_ptb = dec.save_file_cts.replace('.npz','_ptb-aso.npz')
                if not os.path.exists(dec.save_file_cts_ptb) and not dec.out_of_range:
                    dec.load_segmentation()
                    dec.get_ptb_aso(icol_aso=0,icol_ptb=1,th_cor_ptb=0.5,th_ptb=2500)
                    np.savez(dec.save_file_cts_ptb,aso_mean=dec.aso_mean,cm_cells=dec.cm_cells)
        
        """
    
    
        self.dic_ptb = {get_ifov(fl):fl for fl in self.files_set if 'ptb' in os.path.basename(fl).lower()}
        self.ptb_fl = self.dic_ptb.get(self.ifov,None)
        self.dic_aso = {get_ifov(fl):fl for fl in self.files_set if 'aso' in os.path.basename(fl).lower()}
        self.aso_fl = self.dic_aso.get(self.ifov,None)
        
        #load ptb file and correct drift
        Xhs,dic_drift = pickle.load(open(self.ptb_fl,'rb'))
        Xh = Xhs[icol_ptb]
        self.txyz=dic_drift['txyz']
        Xh[:,:3] = Xh[:,:3]-dic_drift['txyz']

        #filter based on correlation with PSF and brightness
        keep = Xh[:,-2]>th_cor_ptb
        keep &= Xh[:,-1]>th_ptb
        Xh = Xh[keep]
        
        #plotting
        if False:
            from matplotlib import cm as cmap
            import napari
            cts_ = np.clip(Xh[:,-1],0,15000)
            cts_ = cts_/np.max(cts_)
            sizes = 1+cts_*5
            colors = cmap.coolwarm(cts_)
            napari.view_points(Xh[:,1:3],face_color=colors,size=sizes)

        ### count per cell
        Xcms = Xh[:,:3]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        icells,cts = np.unique(self.im_segm[tuple(Xred.T)],return_counts=True)
        dic_cts = {icell:ct for icell,ct in zip(icells,cts)}
        ctsf = [dic_cts.get(icell,0) for icell in self.icells]
        self.ptbp_cts = ctsf

        if False:
            import napari
            viewer = napari.view_points(Xred,size=2)
            viewer.add_labels(self.im_segm)

        #do same for aso
        Xhs,dic_drift = pickle.load(open(self.aso_fl,'rb'))
        Xh = Xhs[0]
        self.txyz=dic_drift['txyz']
        Xh[:,:3] = Xh[:,:3]-dic_drift['txyz']
        Xcms = Xh[:,:3]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        labels = self.im_segm[tuple(Xred.T)]
        Xh = Xh[good]
        from scipy import ndimage
        self.aso_mean = ndimage.mean(Xh[:,5],labels=labels,index=self.icells)
        
def get_iH(fld): return int(os.path.basename(fld).split('_')[0][1:])
class decoder_simple():
    def __init__(self,save_folder,fov='Conv_zscan__001',set_='_set1'):
        self.save_folder = save_folder
        self.fov,self.set_ = fov,set_
        save_folder = self.save_folder
        self.decoded_fl = save_folder+os.sep+'decoded_'+fov.split('.')[0]+'--'+set_+'.npz'
        self.drift_fl = save_folder+os.sep+'drift_'+fov.split('.')[0]+'--'+set_+'.pkl'
    def check_is_complete(self):
        if os.path.exists(self.decoded_fl):
            print("Completed")
            return 1
        if not os.path.exists(self.drift_fl):
            print("Did not detect fit files")
            return -1
        print("Not completed")
        return 0
        
    def get_fovs_sets(self):
        self.drift_fls = glob.glob(self.save_folder+os.sep+'drift_*.pkl')
        self.fov_sets = [os.path.basename(fl).replace('drift_','').replace('.pkl','').split('--')
                         for fl in self.drift_fls]
    def get_XH(self,fov,set_,ncols=3):
        self.set_ = set_
        save_folder = self.save_folder
        drift_fl = save_folder+os.sep+'drift_'+fov.split('.')[0]+'--'+set_+'.pkl'
        drifts,all_flds,fov = pickle.load(open(drift_fl,'rb'))
        self.drifts,self.all_flds,self.fov = drifts,all_flds,fov

        XH = []
        for iH in tqdm(np.arange(len(all_flds))):
            fld = all_flds[iH]
            if 'MER' in os.path.basename(fld):
                for icol in range(ncols):
                    tag = os.path.basename(fld)
                    save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npy.npz'
                    if not os.path.exists(save_fl):save_fl = save_fl.replace('.npy','')
                    Xh = np.load(save_fl)['Xh']
                    tzxy = drifts[iH][0]
                    Xh[:,:3]+=tzxy# drift correction
                    ih = get_iH(fld) # get bit
                    bit = (ih-1)*3+icol
                    icolR = np.array([[icol,bit]]*len(Xh))
                    XH_ = np.concatenate([Xh,icolR],axis=-1)
                    XH.extend(XH_)
        self.XH = np.array(XH)
    def get_inters(self,dinstance_th=2,enforce_color=False):
        """Get an initial intersection of points and save in self.res"""
        res =[]
        if enforce_color:
            icols = self.XH[:,-2]
            XH = self.XH
            for icol in tqdm(np.unique(icols)):
                inds = np.where(icols==icol)[0]
                Xs = XH[inds,:3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts,dinstance_th)
                res += [inds[r] for r in res_]
        else:
            XH = self.XH
            Xs = XH[:,:3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
        self.res = res
        
    def load_library(self,lib_fl = r'Z:\DCBBL1_3_2_2023\MERFISH_Analysis\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv',nblanks=-1):
        code_txt = np.array([ln.replace('\n','').split(',') for ln in open(lib_fl,'r') if ',' in ln])
        gns = code_txt[1:,0]
        code_01 = code_txt[1:,2:].astype(int)
        codes = np.array([np.where(cd)[0] for cd in code_01])
        codes_ = [list(np.sort(cd)) for cd in codes]
        nbits = np.max(codes)+1

        codes__ = codes_
        gns__ = list(gns)
        if nblanks>=0:
            ### get extrablanks
            from itertools import combinations
            X_codes = np.array((list(combinations(range(nbits),4))))
            X_code_01 = []
            for cd in X_codes:
                l_ = np.zeros(nbits)
                l_[cd] = 1
                X_code_01.append(l_)
            X_code_01 = np.array(X_code_01,dtype=int)
            from scipy.spatial.distance import cdist
            eblanks = np.where(np.min(cdist(code_01,X_code_01,metric='hamming'),0)>=4/float(nbits))[0]
            codes__ = codes_ + [list(e)for e in X_codes[eblanks]]
            gns__ = list(gns)+ ['blanke'+str(ign+1).zfill(4) for ign in range(len(eblanks))]
        
        bad_gns = np.array(['blank' in e for e in gns__])
        good_gns = np.where(~bad_gns)[0]
        bad_gns = np.where(bad_gns)[0]

        
        
        self.lib_fl = lib_fl ### name of coding library
        self.nbits = nbits ### number of bits
        self.gns_names = gns__  ### names of genes and blank codes
        self.bad_gns = bad_gns ### indices of the blank codes
        self.good_gns = good_gns ### indices of the good gene codes
        self.codes__ = codes__ ### final extended codes of form [bit1,bit2,bit3,bit4]
        self.codes_01 = code_01
        if nblanks>=0:
            self.codes_01 = np.concatenate([code_01,X_code_01[eblanks]],axis=0) ### final extended codes of form [0,1,0,0,1...]
        
        dic_bit_to_code = {}
        for icd,cd in enumerate(self.codes__): 
            for bit in cd:
                if bit not in dic_bit_to_code: dic_bit_to_code[bit]=[]
                dic_bit_to_code[bit].append(icd)
        self.dic_bit_to_code = dic_bit_to_code  ### a dictinary in which each bit is mapped to the inde of a code
    def get_icodes(self,nmin_bits=4,method = 'top4',redo=False,norm_brightness=None):    
        #### unfold res which is a list of list with clusters of loc.
        
        
        res = self.res

        import time
        start = time.time()
        res = [r for r in res if len(r)>=nmin_bits]
        #rlens = [len(r) for r in res]
        #edges = np.cumsum([0]+rlens)
        res_unfolder = np.array([r_ for r in res for r_ in r])
        #res0 = np.array([r[0] for r in res for r_ in r])
        ires = np.array([ir for ir,r in enumerate(res) for r_ in r])
        print("Unfolded molecules:",time.time()-start)

        ### get scores across bits
        import time
        start = time.time()
        RS = self.XH[:,-1].astype(int)
        brighness = self.XH[:,-3]
        brighness_n = brighness.copy()
        if norm_brightness is not None:
            colors = self.XH[:,norm_brightness]#self.XH[:,-1] for bits
            med_cols = {col: np.median(brighness[col==colors])for col in np.unique(colors)}
            for col in np.unique(colors):
                brighness_n[col==colors]=brighness[col==colors]/med_cols[col]
        scores = brighness_n[res_unfolder]
       
        bits_unfold = RS[res_unfolder]
        nbits = len(np.unique(RS))
        scores_bits = np.zeros([len(res),nbits])
        arg_scores = np.argsort(scores)
        scores_bits[ires[arg_scores],bits_unfold[arg_scores]]=scores[arg_scores]

        import time
        start = time.time()
        ### There are multiple avenues here: 
        #### nearest neighbors - slowest
        #### best dot product - reasonable and can return missing elements - medium speed
        #### find top 4 bits and call that a code - simplest and fastest


        if method == 'top4':
            codes = self.codes__
            vals = np.argsort(scores_bits,axis=-1)
            bcodes = np.sort(vals[:,-4:],axis=-1)
            base = [nbits**3,nbits**2,nbits**1,nbits**0]
            bcodes_b = np.sum(bcodes*base,axis=1)
            codes_b = np.sum(np.sort(codes,axis=-1)*base,axis=1)
            icodesN = np.zeros(len(bcodes_b),dtype=int)-1
            for icd,cd in enumerate(codes_b):
                icodesN[bcodes_b==cd]=icd
            bad = np.sum(scores_bits>0,axis=-1)<4
            icodesN[bad]=-1
            igood = np.where(icodesN>-1)[0]
            inds_spotsN =  np.zeros([len(res),nbits],dtype=int)-1
            inds_spotsN[ires[arg_scores],bits_unfold[arg_scores]]=res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            icodesN = icodesN[igood]
        elif method == 'dot':
            icodesN = np.argmax(np.dot(scores_bits[:],self.codes_01.T),axis=-1)
            inds_spotsN =  np.zeros([len(res),nbits],dtype=int)-1
            inds_spotsN[ires[arg_scores],bits_unfold[arg_scores]]=res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])

        print("Computed the decoding:",time.time()-start)

        import time
        start = time.time()

        mean_scores = np.mean(scores_prunedN,axis=-1)
        ordered_mols = np.argsort(mean_scores)[::-1]
        keep_mols = []
        visited = np.zeros(len(self.XH))
        for imol in tqdm(ordered_mols):
            r = np.array(res_prunedN[imol])
            r_ = r[r>=0]
            if np.all(visited[r_]==0):
                keep_mols.append(imol)
                visited[r_]=1
        keep_mols = np.array(keep_mols)
        self.scores_prunedN = scores_prunedN[keep_mols]
        self.res_prunedN = res_prunedN[keep_mols]
        self.icodesN = icodesN[keep_mols]
        print("Computed best unique assigment:",time.time()-start)
        
        XH_pruned = self.XH[self.res_prunedN]
        self.XH_pruned = XH_pruned#self.XH[self.res_prunedN]
        np.savez_compressed(self.decoded_fl,XH_pruned=XH_pruned,icodesN=self.icodesN,gns_names = np.array(self.gns_names))
        #XH_pruned -> 10000000 X 4 X 10 [z,x,y,bk...,corpsf,h,col,bit] 
        #icodesN -> 10000000 index of the decoded molecules in gns_names
        #gns_names
    def load_decoded(self):
        import time
        start= time.time()
        self.decoded_fl = self.save_folder+os.sep+'decoded_'+self.fov.split('.')[0]+'--'+self.set_+'.npz'
        self.XH_pruned = np.load(self.decoded_fl)['XH_pruned']
        self.icodesN = np.load(self.decoded_fl)['icodesN']
        self.gns_names = np.load(self.decoded_fl)['gns_names']
        print("Loaded decoded:",start-time.time())
    def get_is_bright(self,th_dic = {0:1500,1:1500,2:750},get_stats=True):
        self.th_dic = th_dic
        gns_names,icodesN,XH_pruned = self.gns_names,self.icodesN,self.XH_pruned
        th_arr = np.array([th_dic[e]for e in np.sort(list(th_dic.keys()))])
        good_codes = np.where(~np.array(['blank' in gn_nm for gn_nm in gns_names]))[0]
        is_blank = ~np.in1d(icodesN,good_codes)
        Rs = XH_pruned[:,:,-2].astype(int)
        scores_prunedN = XH_pruned[:,:,-3]
        is_bright= np.all(scores_prunedN>th_arr[Rs],axis=-1)
        self.is_bright = is_bright
        fr_blank = np.sum(is_bright[is_blank])/np.sum(is_bright)
        if get_stats:
            print("Fraction error:",fr_blank)
            icodesN_ = icodesN[is_bright]
            icds,ncts = np.unique(icodesN_,return_counts=True)
            keep_good = np.in1d(icds,good_codes)

            plt.figure()
            plt.plot(ncts,'-')
            plt.plot(ncts[~keep_good],'-')
    def get_XH_tag(self,tag='GFP',ncols=3):
        """This looks through all the fitted files (stored in the drift_fl)
        and will load self.Xh the drift corrected fits from file containing <tag>"""
        set_,fov = self.set_,self.fov
        save_folder = self.save_folder
        drift_fl = save_folder+os.sep+'drift_'+fov.split('.')[0]+'--'+set_+'.pkl'
        drifts,all_flds,fov = pickle.load(open(drift_fl,'rb'))
        self.drifts,self.all_flds,self.fov = drifts,all_flds,fov

        XH = []
        
        for iH in np.arange(len(all_flds)):
            fld = all_flds[iH]
            if tag in os.path.basename(fld):
                for icol in range(ncols):
                    tag = os.path.basename(fld)
                    save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npy.npz'
                    if not os.path.exists(save_fl):save_fl = save_fl.replace('.npy','')
                    Xh = np.load(save_fl)['Xh']
                    tzxy = drifts[iH][0]
                    Xh[:,:3]+=tzxy# drift correction
                    #ih = get_iH(fld) # get bit
                    bit = -1#(ih-1)*3+icol
                    if len(Xh):
                        icolR = np.array([[icol,bit]]*len(Xh))
                        #print(icolR.shape,Xh.shape)
                        XH_ = np.concatenate([Xh,icolR],axis=-1)
                        XH.extend(XH_)
        self.Xh = np.array(XH)
            
    def plot_points(self,genes=['Olig2','Gfap'],cols=['r','g'],viewer = None):
        icodesN,XH_pruned = self.icodesN,self.XH_pruned
        is_bright = self.is_bright
        gns_names = list(self.gns_names)
        icodesf = icodesN[is_bright]
        Xcms = np.mean(XH_pruned[is_bright],axis=1)
        H = Xcms[:,-3]
        X = Xcms[:,:3]
        size = 1+np.clip(H/np.percentile(H,95),0,1)*20

        if viewer is None:
            import napari
            viewer = napari.Viewer()
        for ign in range(len(genes)):
            if cols is not None:
                color = cols[ign%len(cols)]
            else:
                color='white'
            gene = genes[ign]
            icode = gns_names.index(gene)
            is_code = icode==icodesf
            viewer.add_points(X[is_code],size=size[is_code],face_color=color,name=gene)
        return viewer
def apply_fine_drift(dec,plt_val=True,npts=50000):
    bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
    good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
    is_good_gn = np.in1d(dec.icodesN,good_igns)
    XHG = dec.XH_pruned[is_good_gn]
    
    RG = XHG[:,:,-1].astype(int)
    iRs=np.unique(RG)
    dic_fine_drift = {}
    for iR in tqdm(iRs):
        XHGiR = XHG[np.any(RG==iR,axis=1)]
        RGiR  = XHGiR[...,-1].astype(int)
        mH = np.median(XHGiR[:,:,-3],axis=1)
        XHF = XHGiR[np.argsort(mH)[::-1][:npts]]
        RF  = XHF[...,-1].astype(int)
        XHFinR = XHF.copy()
        XHFiR = XHF.copy()
        XHFiR[~(RF==iR)]=np.nan
        XHFinR[(RF==iR)]=np.nan
        drift = np.mean(np.nanmean(XHFiR[:,:,:3],axis=1)-np.nanmean(XHFinR[:,:,:3],axis=1),axis=0)
        dic_fine_drift[iR]=drift
    drift_arr = np.array([dic_fine_drift[iR] for iR in iRs])
    if plt_val:
        ncols = len(np.unique(XHG[:,:,-2]))
        X1 = np.array([dic_fine_drift[iR] for iR in iRs[0::ncols]])
        X3 = np.array([dic_fine_drift[iR] for iR in iRs[(ncols-1)::ncols]])

        plt.figure()
        plt.plot(X1[:,0],X3[:,0],'o',label='z-color0-2')
        plt.plot(X1[:,1],X3[:,1],'o',label='x-color0-2')
        plt.plot(X1[:,2],X3[:,2],'o',label='y-color0-2')

        plt.xlabel("Drift estimation color 1 (pixels)")
        plt.ylabel("Drift estimation color 2 (pixels)")
        plt.legend()
    dec.drift_arr = drift_arr
    R = dec.XH_pruned[:,:,-1].astype(int)#
    dec.XH_pruned[:,:,:3] -= drift_arr[R]
def apply_brightness_correction(dec,plt_val=True,npts=50000):
    bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
    good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
    is_good_gn = np.in1d(dec.icodesN,good_igns)
    dec.XH_pruned[:,:,4]=dec.XH_pruned[:,:,-3]
    RT = dec.XH_pruned[:,:,-1].astype(int)
    XHG = dec.XH_pruned[is_good_gn]
    RG = XHG[:,:,-1].astype(int)
    iRs=np.unique(RG)
    #XHG = dec.XH_pruned[is_good_gn]
    ratios = []
    meds = []
    for iR in tqdm(iRs):
        hasiR = np.any(RG==iR,axis=1)
        XHGiR = XHG[hasiR]
        RGiR  = XHGiR[...,-1].astype(int)
        mH = np.median(XHGiR[:,:,-3],axis=1)
        keep_top = np.argsort(mH)[::-1][:npts]
        XHF = XHGiR[keep_top]
        RF  = XHF[...,-1].astype(int)
        XHFinR = XHF.copy()
        XHFiR = XHF.copy()
        isR = (RF==iR)
        XHFiR[~isR]=np.nan
        XHFinR[isR]=np.nan
        ratio = np.median(np.nanmean(XHFiR[:,:,4],axis=1)/np.nanmean(XHFinR[:,:,4],axis=1),axis=0)
        med = np.median(np.nanmean(XHFiR[:,:,4],axis=1))
        ratios.append(ratio)
        meds.append(med)
    ratios = np.array(ratios)
    meds = np.array(meds)
    dec.ratiosH = ratios/meds
    dec.medsH = meds
    dec.XH_pruned[:,:,4]=dec.XH_pruned[:,:,4]/ratios[RT]/meds[RT]
def combine_scoresRef(scoresRef,scoresRefT):
    return [np.sort(np.concatenate([scoresRef[icol],scoresRefT[icol]]),axis=0)
     for icol in np.arange(len(scoresRef))]
def get_score_per_color(dec):
    H = np.median(dec.XH_pruned[...,-3],axis=1)
    D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
    D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
    score = np.array([H,-D]).T
    score = np.sort(score,axis=0)
    return [score[dec.XH_pruned[:,0,-2]==icol] for icol in np.arange(dec.ncols)]
    
def get_score_withRef(dec,scoresRef,plt_val=False,gene=None,iSs=None):
    H = np.median(dec.XH_pruned[...,-3],axis=1)
    D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
    D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
    score = np.array([H,-D]).T
    keep_color = [dec.XH_pruned[:,0,-2]==icol for icol in np.arange(dec.ncols)]
    scoreA = np.zeros(len(H))
    for icol in range(dec.ncols):
        scoresRef_ = scoresRef[icol]
        score_ = score[keep_color[icol]]
        from scipy.spatial import KDTree
        scoreA_ = np.zeros(len(score_))
        if iSs is None: iSs = np.arange(scoresRef_.shape[-1])
        for iS in iSs:
            dist_,inds_ = KDTree(scoresRef_[:,[iS]]).query(score_[:,[iS]])
            scoreA_+=np.log((inds_+1))-np.log(len(scoresRef_))
        scoreA[keep_color[icol]]=scoreA_
    dec.scoreA =scoreA
    if plt_val:
        bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
        good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
        is_good_gn = np.in1d(dec.icodesN,good_igns)
        
        plt.figure()
        plt.hist(scoreA[is_good_gn],density=True,bins=100,alpha=0.5,label='all genes')
        if gene is not None:
            is_gn = dec.icodesN==(list(dec.gns_names).index(gene))
            plt.hist(scoreA[is_gn],density=True,bins=100,alpha=0.5,label=gene)
        plt.hist(scoreA[~is_good_gn],density=True,bins=100,alpha=0.5,label='blanks');
        plt.legend()
def get_scores(dec,plt_val=True,gene='Ptbp1'):
    H = np.median(dec.XH_pruned[...,4],axis=1)
    Hd = np.std(dec.XH_pruned[...,4],axis=1)/H
    D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
    D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
    score = np.array([H,-D])
    scoreA = np.argsort(np.argsort(score,axis=-1),axis=-1)+1
    scoreA = np.sum(np.log(scoreA)-np.log(len(D)),axis=0)
    dec.scoreA = scoreA
    if plt_val:
        bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
        good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
        is_good_gn = np.in1d(dec.icodesN,good_igns)
        
        plt.figure()
        plt.hist(scoreA[is_good_gn],density=True,bins=100,alpha=0.5,label='all genes')
        if gene is not None:
            is_gn = dec.icodesN==(list(dec.gns_names).index(gene))
            plt.hist(scoreA[is_gn],density=True,bins=100,alpha=0.5,label=gene)
        plt.hist(scoreA[~is_good_gn],density=True,bins=100,alpha=0.5,label='blanks');
        plt.legend()
        
def load_segmentation(dec,nexpand=5):
    dec.fl_dapi = glob.glob(dec.save_folder+os.sep+'Segmentation'+os.sep+dec.fov+'*'+dec.set_+'*.npz')[0]
    dic = np.load(dec.fl_dapi)
    im_segm = dic['segm']
    dec.shape = dic['shape']
    dec.im_segm_=stitch3D(im_segm,niter=5,th_int=0.75)
    dec.im_segm_ = expand_segmentation(dec.im_segm_,nexpand=nexpand)
    drift_fl = dec.save_folder+os.sep+'drift_'+dec.fov+'--'+dec.set_+'.pkl'
    drifts,fls,fov = pickle.load(open(drift_fl,'rb'))
    dec.drifts = np.array([drft[0]for drft in drifts])
    dec.drifts = drifts
    tag_dapi = os.path.basename(dec.fl_dapi).split('--')[1]
    tags_drifts = [os.path.basename(fld)for fld in fls]
    itag_dapi = tags_drifts.index(tag_dapi)
    #dec.drifts -= dec.drifts[itag_dapi]
    dec.drift = dec.drifts[itag_dapi]
def plot_1gene(self,gene='Gad1',viewer = None):
    icodesN,XH_pruned = self.icodesN,self.XH_pruned
    scoreA=self.scoreA
    th=self.th
    gns_names = list(self.gns_names)
    icodesf = icodesN
    Xcms = np.mean(XH_pruned,axis=1)
    H = Xcms[:,-3]
    X = Xcms[:,:3]
    size = 1+np.clip(H/np.percentile(H,95),0,1)*20
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()

    icode = gns_names.index(gene)
    is_code = icode==icodesf
    viewer.add_points(X[is_code],size=size[is_code],face_color='r',name=gene)

    is_gn = self.icodesN==(list(self.gns_names).index(gene))
    keep_gn = scoreA[is_gn]>th
    Xcms = np.mean(self.XH_pruned,axis=1)
    viewer.add_points(Xcms[is_gn][keep_gn][:,:3],size=10,face_color='g',name=gene)
    return viewer
def plot_multigenes(self,genes=['Gad1','Sox9'],colors=['r','g','b','m','c','y','w'],smin=3,smax=10,viewer = None,
                    drift=[0,0,0],resc=[1,1,1]):
    icodesN,XH_pruned = self.icodesN,self.XH_pruned
    scoreA=self.scoreA
    th=self.th
    gns_names = list(self.gns_names)
    
    Xcms = np.mean(XH_pruned,axis=1)
    keep = scoreA>th
    X = (Xcms[:,:3][keep]-drift)/resc  
    H = scoreA[keep]
    H -= np.min(H)
    icodesf = icodesN[keep]
    size = smin+np.clip(H/np.max(H),0,1)*(smax-smin)
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()
    for igene,gene in enumerate(genes):
        color= colors[igene%len(colors)]
        icode = gns_names.index(gene)
        is_code = icode==icodesf
        viewer.add_points(X[is_code],size=size[is_code],face_color=color,name=gene)

    return viewer
def plot_points_direct(Xh,gene='gene',color='g',minsz=0,maxsz=20,percentage_max = 95,viewer=None):
    H = Xh[:,-3]
    X = Xh[:,:3]
    Hmax = np.percentile(H,percentage_max)
    keep = H>Hmax/10
    H = Xh[keep,-3]
    X = Xh[keep,:3]
    size = minsz+np.clip(H/Hmax,0,1)*maxsz
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()

    viewer.add_points(X,size=size,face_color=color,name=gene)
    return viewer
    
def load_GFP(dec,th_cor=0.25,th_h=2000,th_d=2,plt_val=True):
    dec.get_XH_tag(tag='GFP')
    Xh1 = dec.Xh[dec.Xh[:,-2]==1]
    Xh2 = dec.Xh[dec.Xh[:,-2]==2]
    Xh1 = Xh1[Xh1[:,-4]>th_cor]
    Xh2 = Xh2[Xh2[:,-4]>th_cor]
    Xh1 = Xh1[Xh1[:,-3]>th_h]
    Xh2 = Xh2[Xh2[:,-3]>th_h]
    #viewer = plot_points_direct(Xh1,gene='GFP',color=[0,1,0],minsz=0,maxsz=20,percentage_max = 95,viewer = None)
    #plot_points_direct(Xh2,gene='GFP',color=[0,0.5,0],minsz=0,maxsz=20,percentage_max = 95,viewer=viewer);
    from scipy.spatial import KDTree
    tree = KDTree(Xh1[:,:3])
    dist,iXh1 = tree.query(Xh2[:,:3])
    iXh2 = dist<th_d
    iXh1 = iXh1[iXh2]
    dec.Xh1GFP,dec.Xh2GFP =Xh1[iXh1],Xh2[iXh2]
    if plt_val:
        viewer = None
        viewer=plot_points_direct(dec.Xh1GFP,gene='GFP',color=[0,1,0],minsz=0,maxsz=20,percentage_max = 95,viewer = viewer)
        viewer=plot_points_direct(dec.Xh2GFP,gene='GFP',color=[0,0.5,0],minsz=0,maxsz=20,percentage_max = 95,viewer=viewer);
        return viewer
        
def get_cell_id(dec,Xh):
    tzxy = dec.drift[0]#dec.drift_dapi
    im_segm = dec.im_segm_
    dec.shapesm = dec.im_segm_.shape
    
    Xcms = Xh[:,:3]-tzxy#?
    Xred = np.round((Xcms/dec.shape)*dec.shapesm).astype(int)
    good = ~np.any((Xred>=dec.shapesm)|(Xred<0),axis=-1)
    Xred = Xred[good]
    return im_segm[tuple(Xred.T)],good
    
def get_counts_per_cell(dec,Xh):
    tzxy = dec.drift[0]#dec.drift_dapi
    im_segm = dec.im_segm_
    dec.shapesm = dec.im_segm_.shape
    
    Xcms = Xh[:,:3]-tzxy#?
    Xred = np.round((Xcms/dec.shape)*dec.shapesm).astype(int)
    good = ~np.any((Xred>=dec.shapesm)|(Xred<0),axis=-1)
    Xred = Xred[good]
    
    icells,cts = np.unique(im_segm[tuple(Xred.T)],return_counts=True)
    dic_cts = {icell:ct for icell,ct in zip(icells,cts)}
    ctsf = np.array([dic_cts.get(icell,0) for icell in dec.icells])
    return ctsf
    
def get_signal_ab(dec,fld_dapi = r'Y:\DCBBL1_3_15_2023__GFP\H9_MER',
              fld_ab= r'Y:\DCBBL1_3_15_2023__GFP\A5_GFPAb_B_B_',th_sig = 5000,sz_drift=20,icol=0):
    dec.fl_ab_raw=fld_ab+dec.set_+os.sep+dec.fov+'.zarr'
    dec.fl_dapi_raw=fld_dapi+dec.set_+os.sep+dec.fov+'.zarr'
    imab,dec.xfov,dec.yfov = read_im(dec.fl_ab_raw,return_pos=True)
    ncols,sz,sx,sy = imab.shape
    imab_ = np.array(imab[-1,(sz-sz_drift)//2:(sz+sz_drift)//2],dtype=np.float32)
    imdapi = read_im(dec.fl_dapi_raw)
    imdapi_ = np.array(imdapi[-1,(sz-sz_drift)//2:(sz+sz_drift)//2],dtype=np.float32)
    dec.im_ab = np.array(imab[icol],dtype=np.float32)
    txyz,txyzs = get_txyz(imdapi_, imab_, sz_norm=30, sz=300, nelems=5)
    dec.im_abn = nd.shift(norm_slice(dec.im_ab,s=250),-txyz,order = 0)

    dec.im_abn_sm = resize(dec.im_abn,dec.im_segm_.shape)

    ab_sigs = nd.mean(dec.im_abn_sm,dec.im_segm_,dec.icells)
    ab_sigs2 = nd.sum(dec.im_abn_sm>th_sig,dec.im_segm_,dec.icells)
    vols = nd.sum(dec.im_segm_>0,dec.im_segm_,dec.icells)
    dec.ab_sigs,dec.ab_sigs2,dec.vols=ab_sigs,ab_sigs2,vols
    
from scipy import ndimage
def Xh_to_im(Xh,resc= 10,sx=3000,sy=3000):
    X = Xh[:,1:3].astype(int)//resc
    Xf = X[:,0]+sx//resc*X[:,1]
    Xim = np.indices([sx//resc,sy//resc]).reshape([2,-1]).T
    Ximf = Xim[:,0]+sx//resc*Xim[:,1]
    im_sum = ndimage.mean(Xh[:,-1], Xf,Ximf).reshape([sx//resc,sy//resc])
    return im_sum.astype(np.float32)
def compute_flat_fields(save_folder=r'\\192.168.0.10\bbfishdc13\DCBBL1_3_2_2023\MERFISH_Analysis',ncols=3,resc=10,nfls = 1000):
    for icol in range(ncols):
        fls = glob.glob(save_folder+os.sep+'*H2_*--col'+str(icol)+'__Xhfits.npz')[:nfls]
        imf = []
        for fl in tqdm(fls[:]):
            try:
                imf.append(Xh_to_im(np.load(fl)['Xh'],resc))
            except:
                pass
        imf = np.array(imf)
        imff = np.nanmedian(imf,axis=0)
        np.savez(save_folder+os.sep+'med_col'+str(icol)+'.npz',im=imff,resc=resc)
def norm_brightness(dec,Xh):
    ### renormalize the brightness according to flatfield
    Icol = Xh[:,-2].astype(int)
    H = Xh[:,-3].copy()
    cols=np.unique(Icol)
    for icol in cols:
        keep = Icol==icol
        immed = dec.immeds[icol].copy()
        immed = immed/np.median(immed)
        x_,y_ = ((Xh[keep][:,1:3]/dec.resc).astype(int)%immed.shape).T
        norm_ = immed[x_,y_] 
        H[keep]=H[keep]/norm_
    Xh[:,-3] = H
    return Xh
def apply_flat_field(dec,tag='med_col_raw'):
    ### load the immeds
    Icol = dec.XH_pruned[:,:,-2].astype(int)
    uIcols = np.unique(Icol)
    dec.ncols = len(uIcols)
    save_folder=dec.save_folder#r'\\192.168.0.10\bbfishdc13\DCBBL1_3_2_2023\MERFISH_Analysis'
    immeds = []
    for icol in range(dec.ncols):
        dic = np.load(save_folder+os.sep+tag+str(icol)+'.npz')
        immed,resc=dic['im'],dic['resc']
        immeds.append(immed)
    dec.immeds = np.array(immeds)
    dec.resc = resc
    
    ### renormalize the brightness according to flatfield
    XH = dec.XH_pruned
    Icol = XH[:,:,-2].astype(int)
    H = dec.XH_pruned[:,:,-3].copy()
    for icol in range(dec.ncols):
        keep = Icol==icol
        immed = dec.immeds[icol].copy()
        immed = immed/np.median(immed)
        x_,y_ = ((XH[keep][:,1:3]/dec.resc).astype(int)%immed.shape).T
        norm_ = immed[x_,y_] 
        H[keep]=H[keep]/norm_
    dec.XH_pruned[:,:,-3] = H
def example_run():
    dec.fov,dec.set_ = 'Conv_zscan__111','_set1'
    for dec.fov,dec.set_ in tqdm(dec.fov_sets):
        save_fl_final = dec.save_folder+os.sep+'ctspercell_'+dec.fov.split('.')[0]+'--'+dec.set_+'.npz'
        if not os.path.exists(save_fl_final):
            try:
                dec.decoded_fl = dec.save_folder+os.sep+'decoded_'+dec.fov.split('.')[0]+'--'+dec.set_+'.npz'
                load_segmentation(dec)
                dec.load_decoded()
                apply_fine_drift(dec,plt_val=False)
                for i in range(3):
                    apply_brightness_correction(dec)
                get_scores(dec,plt_val=False)
                dec.th=-0.75
                #plot_1gene(dec,gene='Gad1',viewer = None)


                keepf=  dec.scoreA>-0.75 ### keep good score
                XHf = np.mean(dec.XH_pruned[keepf],axis=1)
                icodesf = dec.icodesN[keepf]
                dec.icells = np.unique(dec.im_segm_)
                dec.icells = dec.icells[dec.icells>0]
                cts_all = []
                gns_all = []
                for ign,gn in enumerate(tqdm(dec.gns_names)):
                    Xh = XHf[icodesf==ign]
                    ctsf = get_counts_per_cell(dec,Xh)
                    gns_all.append(gn)
                    cts_all.append(ctsf)


                ### get ALdh1l1
                dec.get_XH_tag(tag='Aldh1')
                Xh = dec.Xh[dec.Xh[:,-2]==1]
                Xh = Xh[Xh[:,-3]>4500]
                ctsf = get_counts_per_cell(dec,Xh)
                gns_all.append('Aldh1l1')
                cts_all.append(ctsf)
                #viewer = plot_points_direct(Xh,gene='Aldh1l1',percentage_max=100)
                ### get GFP - RNA
                load_GFP(dec,th_cor=0.25,th_h=2000,th_d=2,plt_val=False)

                ctsf = get_counts_per_cell(dec,dec.Xh1GFP)
                gns_all.append('GFP_rna')
                cts_all.append(ctsf)

                ### Get antibody

                get_signal_ab(dec,fld_dapi = r'Y:\DCBBL1_3_15_2023__GFP\H9_MER',
                              fld_ab= r'Y:\DCBBL1_3_15_2023__GFP\A5_GFPAb_B_B_',th_sig = 5000,sz_drift=20,icol=0)

                gns_all.append('GFP_Ab1_mean')
                cts_all.append(dec.ab_sigs)

                gns_all.append('GFP_Ab1_th')
                cts_all.append(dec.ab_sigs2)

                Xcells = nd.center_of_mass(dec.im_segm_>0,dec.im_segm_,dec.icells)


                np.savez(save_fl_final,gns_all=gns_all,cts_all=cts_all,vols=dec.vols,Xcells=Xcells,Xfov=[dec.xfov,dec.yfov],icells = dec.icells)
            except:
                print("Failed",save_fl_final)
def plot_statistics(dec):
    if hasattr(dec,'im_segm_'):
        ncells = len(np.unique(dec.im_segm_))-1
    else:
        ncells = 1
    icds,ncds = np.unique(dec.icodesN[dec.scoreA>dec.th],return_counts=True)
    good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
    kp = np.in1d(icds,good_igns)
    ncds = ncds/ncells
    plt.figure()
    plt.xlabel('Genes')
    plt.plot(icds[kp],ncds[kp],label='genes')
    plt.plot(icds[~kp],ncds[~kp],label='blank')
    plt.ylabel('Number of molecules in the fov')
    plt.title(str(np.round(np.mean(ncds[~kp])/np.mean(ncds[kp]),3)))
    plt.legend()
def get_xyfov(dec):
    drifts,fls,fov = pickle.load(open(dec.drift_fl,'rb'))
    fl = fls[0]+os.sep+dec.fov.split('.')[0]+'.xml'
    txt = open(fl,'r').read()
    dec.xfov,dec.yfov = eval(txt.split('<stage_position type="custom">')[-1].split('<')[0])
def save_final_decoding(save_folder,fov,set_,scoresRef,th=-1.5,plt_val=False,
                        tags_smFISH=['Aldh','Sox11'],
                        genes_smFISH=[['Igfbpl1','Aldh1l1','Ptbp1'],['Sox11','Sox2','Dcx']],Hths=None,force=False):
    """
    This loads the decoded points renormalizes them and picks the most confident points
    """
    if type(scoresRef) is str: scoresRef = np.load(scoresRef,allow_pickle=True)
    dec = decoder_simple(save_folder,fov,set_)
    save_fl = dec.save_folder+os.sep+os.sep+'finaldecs_'+dec.fov.split('.')[0]+'--'+dec.set_+'.npz'
    if not os.path.exists(save_fl) or force:
        #print(dec.fov,dec.set_)
        try:
            load_segmentation(dec)
            dec.load_decoded()
            apply_flat_field(dec)
            apply_fine_drift(dec,plt_val=plt_val)
            
            #for i in range(3):
            #    apply_brightness_correction(dec)
            #get_scores(dec,plt_val=plt_val)
            get_score_withRef(dec,scoresRef,plt_val=plt_val,gene=None,iSs = None)
            dec.th=th
            #plot_1gene(dec,gene='Gad1',viewer = None)
            if plt_val:
                viewer = plot_multigenes(dec,genes=['Adcy1','Slc1a2','Psap'],colors=['r','g','b','m','c','y','w'],viewer = None,
                                         smin=1,smax=2.5,drift=dec.drift[0],
                                        resc = dec.shape/dec.im_segm_.shape)
                viewer.add_labels(dec.im_segm_);
            if plt_val:
                plot_statistics(dec)
            #print(dec.gns_ordered)

            keepf =  dec.scoreA>dec.th ### keep good score
            icodesf = dec.icodesN[keepf]
            XHfpr = dec.XH_pruned[keepf]
            XHf = np.mean(XHfpr,axis=1)
            if Hths is None:
                ICol = XHfpr[:,:,-2].astype(int)
                Hths = [np.percentile(XHfpr[ICol==icol][:,-3],15) for icol in np.unique(ICol)]
            
            ### deal with smFISH
            for tag_smFISH,gns_smFISH in zip(tags_smFISH,genes_smFISH):
                dec.get_XH_tag(tag=tag_smFISH)#dec.get_XH_tag(tag='Aldh1')
                Xh = norm_brightness(dec,dec.Xh)
                tags = [gn+'_smFISH' for gn in gns_smFISH]#['Igfbp_smFISH','Aldh1l1_smFISH','Ptbp1_smFISH']
                XF = XHf[:,[0,1,2,-5,-4,-3,-2,-1,-1,-1,-1]]
                #zc,xc,yc,bk-7,a-6,habs-5,hn-4,h-3
                XF[:,-1] = dec.scoreA[keepf]
                XF[:,-2] = np.where(keepf)[0]
                mnD = np.mean(np.linalg.norm((XHf[:,np.newaxis]-XHfpr)[:,:,:3],axis=-1),axis=-1)
                XF[:,-3]=mnD
                mnH = np.mean(np.abs((XHf[:,np.newaxis]-XHfpr)[:,:,-3]),axis=-1)
                XF[:,-4]=mnH
                genesf = dec.gns_names[icodesf]

                for icol,tag in enumerate(tags):
                    Xh_ = Xh[Xh[:,-2]==icol]
                    Xh_=Xh_[Xh_[:,-3]>Hths[icol]]
                    Xh_=Xh_[:,[0,1,2,-5,-4,-3,-2,-1,-1,-1,-1]]
                    Xh_[:,-1]=0
                    Xh_[:,-2]=-1
                    Xh_[:,-3]=0
                    Xh_[:,-4]=0
                    XF = np.concatenate([XF,Xh_])
                    genesf = np.concatenate([genesf,[tag]*len(Xh_)])

            cell_id,good = get_cell_id(dec,XF)
            XF_ = np.concatenate([XF[good],cell_id[:,np.newaxis]],axis=-1)
            genesf_ = genesf[good]
            iset = int(dec.set_.split('_set')[-1])
            ifov = int(dec.fov.split('_')[-1].split('.')[0])
            isets = np.array([iset]*len(cell_id))[:,np.newaxis]
            ifovs = np.array([ifov]*len(cell_id))[:,np.newaxis]
            cell_id = cell_id[:,np.newaxis]
            XF_ = np.concatenate([XF[good],cell_id,ifovs,isets],axis=-1)

            get_xyfov(dec)
            XF_ = XF_[:,list(np.arange(XF_.shape[-1]))+[-1,-1]]
            XF_[:,-2:]=dec.xfov,dec.yfov
            header = ['z','x','y','abs_brightness','cor','brightness','color','mean_bightness_variation','mean_distance_variation',
                      'index_from_XH_pruned','score','cell_id','ifov','iset','xfov','yfov']
            icells,vols = np.unique(dec.im_segm_,return_counts=True)
            cms = np.array(ndimage.center_of_mass(np.ones_like(dec.im_segm_),dec.im_segm_,icells))
            icells,vols = np.unique(dec.im_segm_,return_counts=True)
            cms = np.array(ndimage.center_of_mass(np.ones_like(dec.im_segm_),dec.im_segm_,icells))
            cellinfo = cms[:,[0,0,0,1,2,0,0]]
            cellinfo[:,0]=icells
            cellinfo[:,1]=vols
            cellinfo[:,-2:]=dec.xfov,dec.yfov
            header_cells = ['cell_id','volm','zc','xc','yc','xfov','yfov']

            np.savez_compressed(save_fl,XF=XF_.astype(np.float32),
                                genes = genesf_,cellinfo=cellinfo.astype(np.float32),header_cells=header_cells,header=header)
        except:
            print("Failed",dec.fov,dec.set_)
            
            
def plot_gene_mosaic_cells(df,cell_df,gene,plt_fov=False,pixel_size = 0.10833*4):
    xcells = cell_df['xc']*pixel_size+cell_df['yfov']
    ycells = cell_df['yc']*pixel_size-cell_df['xfov']
    Xcells = np.array([xcells,ycells]).T
    
    cts = np.array(df[gene])#Ptbp1_smFISH
    cts[np.isnan(cts)]=0
    ncts = np.clip(cts/20,0,1)
    size = 5+ncts*10
    from matplotlib import cm as cmap
    cols = cmap.coolwarm(ncts)
    import napari
    good_cells = slice(None)
    XC = -Xcells[good_cells,::-1]
    viewer = napari.view_points(XC,size=size,face_color=cols[good_cells],name=gene)
    if plt_fov:
        ifovs = np.array(list(df.index),dtype=int)//10**5
        ifov_unk = np.unique(ifovs)
        Xfov = np.array([np.mean(XC[ifovs==ifov],axis=0)for ifov in ifov_unk])
        features =  {'fov':ifov_unk}
        text = {
            'string': '{fov:.1f}',
            'size': 20,
            'color': 'gray',
            'translation': np.array([0, 0]),
        }
        viewer.add_points(Xfov,text=text,features=features)
def compute_flat_field_raw(data_fld,
            save_folder =r'\\192.168.0.6\bbfish1e3\DCBBL1_03_14_2023_big\MERFISH_Analysis',
            tag='microscope',
            ncols=4):
    zarrs = glob.glob(data_fld+os.sep+'*.zarr')
    for icol in range(ncols):
        ims_ = [np.array(read_im(fl)[icol][10],dtype=np.float32) for fl in tqdm(zarrs)]
        immed = np.median(ims_,axis=0)
        np.savez(save_folder+os.sep+tag+'__med_col_raw'+str(icol)+'.npz',im=immed,resc=1)
def get_psf(im_,th=1000,th_cor = 0.75,delta=3,delta_fit = 7,sxyzP = [15,30,30]):

    """
    Use as :
    
    psfs = []
    for ifov in tqdm(range(55,80)):
        im = read_im(r'X:\CGBB_embryo_4_28_2023\P1_Sox11_Sox2_Dcx_D16\Conv_zscan__'+str(ifov).zfill(3)+'.zarr')
        im_ = np.array(im[0][1:,500:2500,500:2500],dtype=np.float32)
        psf = get_psf(im_,th=1000,th_cor = 0.75,delta=3,delta_fit = 7,sxyzP = [15,60,60])
        psfs.append(psf)
    #napari.view_image(im)
    psff = np.mean([psf for psf in psfs if psf is not None],axis=0)

    psff = np.mean([psf for psf in psfs if psf is not None],axis=0)
    psff_ = np.array([p-np.median(p) for p in psff])
    from scipy.ndimage import median_filter
    psff_med = median_filter(psff_, size=15)
    psfff = (psff_-psff_med)[5:-5,5:-5,5:-5][:-1,:-1,:-1]
    psfff[psfff<0]=0
    psfff = psfff/np.max(psfff)
    np.save('psf_750_Scope1_embryo_big_final.npy',psfff)
    
    """

    im_n = norm_slice(im_,s=30)
    import torch
    Xh = get_local_maxfast_tensor(im_n,im_raw=im_,th_fit=th,gpu=True,delta=delta,delta_fit=delta_fit)
    if Xh is not None:
        if len(Xh)>0:
            Xh_ = Xh[(Xh[:,-2]>th_cor)&(Xh[:,0].astype(int)>0)]

            X = Xh_[:,:3]
            XT = np.round(X).astype(np.int16)
            szP,sxP,syP = sxyzP
            Xi = np.indices([2*szP+1,2*sxP+1,2*syP+1],dtype=np.int16).reshape([3,-1]).T-np.array([szP,sxP,syP])
            dev = 'cpu'
            XT = torch.from_numpy(XT).to(dev)
            Xi = torch.from_numpy(Xi[:,np.newaxis]).to(dev)
            shape_ = torch.from_numpy(np.array(im_.shape,dtype=np.int16)).to(dev)
            Xf = ((XT+Xi)%(shape_)).type(torch.int64)
            imdev = torch.from_numpy(im_).to(dev)
            ims = imdev[Xf[...,0],Xf[...,1],Xf[...,2]].reshape([2*szP+1,2*sxP+1,2*syP+1,-1])


            height,width,depth = [2*szP+1,2*sxP+1,2*syP+1]
            kx, ky, kz = np.mgrid[:height, :width, :depth]  # ,:self.sliceShape[2]]

            kx = np.fft.fftshift(kx - height / 2.) / height
            ky = np.fft.fftshift(ky - width / 2.) / width
            kz = np.fft.fftshift(kz - depth / 2.) / depth

            dx = torch.from_numpy((X-np.round(X)).astype(np.float32)).to(dev)
            k = torch.from_numpy(np.array([kx,ky,kz]).astype(np.float32)).to(dev)
            expK = torch.exp(-2j*np.pi*torch.tensordot(dx,k,dims=1)).moveaxis(0,-1)
            F = torch.fft.fftn(ims,dim=[0,1,2])
            psf = torch.mean(torch.fft.ifftn(F*expK,dim=[0,1,2]).real,-1)
            return psf.cpu().detach().numpy()
def full_deconv(im_,s_=500,pad=100,psf=None,parameters={'method': 'wiener', 'beta': 0.001, 'niter': 50},gpu=True,force=False):
    im0=np.zeros_like(im_)
    sx,sy = im_.shape[1:]
    ixys = []
    for ix in np.arange(0,sx,s_):
        for iy in np.arange(0,sy,s_):
            ixys.append([ix,iy])
    
    for ix,iy in tqdm(ixys):#ixys:#tqdm(ixys):
        imsm = im_[:,ix:ix+pad+s_,iy:iy+pad+s_]
        imt = apply_deconv(imsm,psf=psf,parameters=parameters,gpu=gpu,plt_val=False,force=force)
        start_x = ix+pad//2 if ix>0 else 0
        end_x = ix+pad//2+s_
        start_y = iy+pad//2 if iy>0 else 0
        end_y = iy+pad//2+s_
        #print(start_x,end_x,start_y,end_y)
        im0[:,start_x:end_x,start_y:end_y] = imt[:,(start_x-ix):(end_x-ix),(start_y-iy):(end_y-iy)]
    return im0
###added fine_drift
class fine_drift:
    def __init__(self,fl_ref,fl,verbose=True,sz_block=600):
        self.sz_block=sz_block
        self.verbose=verbose
        self.fl,self.fl_ref='',''
        self.get_drift(fl_ref,fl)
        
    def get_drift(self,fl_ref,fl):
        if fl_ref!=self.fl_ref:
            if self.verbose: print("Loading:",fl_ref)
            self.im_ref = self.load_im(fl_ref)
            if self.verbose: print("Finding markers...")
            self.XB1_minus,self.XB1_plus = self.get_X_plus_minus(self.im_ref)
            self.fl_ref = fl_ref
        if fl!=self.fl:
            if self.verbose: print("Loading:",fl)
            self.im = self.load_im(fl)
            if self.verbose: print("Finding markers...")
            self.XB2_minus,self.XB2_plus = self.get_X_plus_minus(self.im)
            if self.verbose: print("Finding rough drift...")
            self.drift = get_txyz(self.im_ref,self.im,sz_norm=30,sz=self.sz_block)
            self.exp_drift = self.drift[0]
            if self.verbose: print("Finding fine drift...")
            self.drft_minus,self.pair_minus = get_best_drift(self.XB1_minus,self.XB2_minus,self.exp_drift,5)
            self.drft_plus,self.pair_plus = get_best_drift(self.XB1_plus,self.XB2_plus,self.exp_drift,5)
            self.fl = fl
        return self.drft_plus,self.pair_plus,self.drft_minus,self.pair_minus
            
            
    def load_im(self,fl):
        return np.array(read_im(fl)[-1],dtype=np.float32)
        
    def get_X_plus_minus(self,im1):
        #load dapi
        im1n = normalize_ims(im1,zm=30,zM=60)
        XB1 = get_XB(-im1n,th=2.5)
        XB1_minus = get_max_min(XB1,-im1n,delta_fit=7,ismax=True,return_ims=False)
        XB1 = get_XB(im1n,th=2.5)
        XB1_plus = get_max_min(XB1,im1n,delta_fit=7,ismax=True,return_ims=False)
        return XB1_minus,XB1_plus
#added additional method missing normalize_ims
def normalize_ims(im0,zm=5,zM=50):
    imn = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im0])
    return imn
#added additional method, was missing get_XB
def get_XB(im_,th=3):
    #im_ = self.im1n
    std_ = np.std(im_[::5,::5,::5])
    keep = im_>std_*th
    XB = np.array(np.where(keep)).T
    from tqdm import tqdm
    for delta_fit in tqdm([1,2,3,5,7,10,15]):
        XI = np.indices([2*delta_fit+1]*3)-delta_fit
        keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
        XI = XI[:,keep].T
        XS = (XB[:,np.newaxis]+XI[np.newaxis])
        shape = im_.shape
        XS = XS%shape

        keep = im_[tuple(XB.T)]>=np.max(im_[tuple(XS.T)],axis=0)
        XB = XB[keep]
    return XB
#additional method added, missing get_max_min
def get_max_min(P,imn,delta_fit=5,ismax=True,return_ims=False):
    XI = np.indices([2*delta_fit+1]*3)-delta_fit
    keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
    XI = XI[:,keep].T
    XS = (P[:,np.newaxis]+XI[np.newaxis])
    shape = imn.shape
    XSS = XS.copy()
    XS = XS%shape
    #XSS = XS.copy()
    is_bad = np.any((XSS!=XS),axis=-1)


    sh_ = XS.shape
    XS = XS.reshape([-1,3])
    im1n_local = imn[tuple(XS.T)].reshape(sh_[:-1])
    #print(is_bad.shape,im1n_local.shape)
    im1n_local[is_bad]=np.nan

    im1n_local = im1n_local-np.nanmin(im1n_local,axis=1)[:,np.newaxis]
    im1n_local = im1n_local/np.nansum(im1n_local,axis=1)[:,np.newaxis]
    XS = XS.reshape(list(im1n_local.shape)+[3])
    return np.nansum(im1n_local[...,np.newaxis]*XS,axis=1)
#get best drift
def get_best_drift(XB1_,XB2_,exp_drift,th_d = 5):
    XB2T = XB2_-exp_drift     
    #XB1_ = XB1_minus
    ds,inds2 = cKDTree(XB2T).query(XB1_)
    inds1 = np.where(ds<th_d)[0]
    inds2 = inds2[inds1]
    XB1__ = XB1_[inds1]
    XB2__ = XB2_[inds2]
    pair_minus = [XB1__,XB2__]
    drft_minus = np.mean((XB2__-XB1__),axis=0)
    return drft_minus,pair_minus
class get_dapi_features:
    def __init__(self,fl,save_folder,set_='',gpu=True,im_med_fl = r'D:\Carlos\Scripts\flat_field\lemon__med_col_raw3.npz',
                psf_fl = r'D:\Carlos\Scripts\psfs\psf_647_Kiwi.npy',redo=False):
                
        """
        Given a file <fl> and a save folder <save_folder> this class will load the image fl, flat field correct it, it deconvolves it using <psf_fl> and then finds the local minimum and maximum.
        
        This saves data in: save_folder+os.sep+fov+'--'+htag+'--dapiFeatures.npz' which contains the local maxima: Xh_plus and local minima: Xh_min 
        """
        htag = os.path.basename(os.path.dirname(fl))
        fov = os.path.basename(os.path.splitext(fl)[0])

        self.gpu=gpu
        
        self.fl,self.fl_ref='',''
        self.im_med=None
        self.im_med_fl=im_med_fl
        
        
        self.save_fl = save_folder+os.sep+fov+'--'+htag+'--'+set_+'dapiFeatures.npz'
        self.fl = fl
        if os.path.exists(self.save_fl) and (not redo):
            try:
                dic = np.load(self.save_fl,allow_pickle=True)
                self.Xh_minus,self.Xh_plus = dic['Xh_minus'],dic['Xh_plus']
            except:
                redo=True
        if not os.path.exists(self.save_fl) or redo:
            self.psf = np.load(psf_fl)
            if im_med_fl is not None:
                im_med = np.load(im_med_fl)['im']
                im_med = cv2.blur(im_med,(20,20))
                self.im_med=im_med
            self.load_im()
            self.get_X_plus_minus()
            np.savez(self.save_fl,Xh_plus = self.Xh_plus,Xh_minus = self.Xh_minus)

    def load_im(self):
        """
        Load the image from file fl and apply: flat field, deconvolve, subtract local background and normalize by std
        """
        im = np.array(read_im(self.fl)[-1],dtype=np.float32)
        if self.im_med_fl is not None:
            im = im/self.im_med*np.median(self.im_med)
        imD = full_deconv(im,psf=self.psf,gpu=self.gpu,force=True)
        imDn = norm_slice(imD,s=30)
        imDn_ = imDn/np.std(imDn)
        self.im = imDn_
    def get_X_plus_minus(self):
        #load dapi
        im1 = self.im
        self.Xh_plus = get_local_maxfast_tensor(im1,th_fit=3,delta=5,delta_fit=5)
        self.Xh_minus = get_local_maxfast_tensor(-im1,th_fit=3,delta=5,delta_fit=5)



def load_segmentation(dec,segm_folder =  r'\\merfish8\merfish8v2\20230805_D103_Myh67_d80KO\DNA_singleCy5\AnalysisDeconvolve_CG\SegmentationDAPI_CG',tag='H1_R1',th_vol=5000):
    dec.fl_dapi = segm_folder+os.sep+dec.fov+'--'+tag+'--CYTO_segm.npz'
    dic = np.load(dec.fl_dapi)
    im_segm = dic['segm']
    dec.shape = dic['shape']
    dec.im_segm_=im_segm
    icells,vols = np.unique(dec.im_segm_,return_counts=True)
    dec.im_segm_ = replace_mat(dec.im_segm_,np.where(vols<th_vol)[0],0)
    icells,vols = np.unique(dec.im_segm_,return_counts=True)
    dec.icells = icells[icells>0]
    dec.cms = np.array(nd.center_of_mass(dec.im_segm_>0,dec.im_segm_,dec.icells))
    dec.resc_segm = dec.im_segm_.shape/dec.shape
def replace_mat(mat,vals_fr,vals_to):
    if len(vals_fr)>0:
        vmax = np.max(mat)+1
        vals = np.arange(vmax)
        vals[vals_fr]=vals_to
        return vals[mat]
    return mat
def get_xy_fl(fl):
    fl_xml = fl.replace('.zarr','.xml')
    txt = open(fl_xml,'r').read()
    xyfov = eval(txt.split('<stage_position type="custom">')[-1].split('<')[0])
    return xyfov
def load_segmentation(dec,segm_folder =  r'\\merfish8\merfish8v2\20230805_D103_Myh67_d80KO\DNA_singleCy5\AnalysisDeconvolve_CG\SegmentationDAPI_CG',tag='H1_R1',th_vol=5000):
    dec.fl_dapi = segm_folder+os.sep+dec.fov+'--'+tag+'--CYTO_segm.npz'
    dic = np.load(dec.fl_dapi)
    im_segm = dic['segm']
    dec.shape = dic['shape']
    dec.im_segm_=im_segm
    icells,vols = np.unique(dec.im_segm_,return_counts=True)
    dec.im_segm_ = replace_mat(dec.im_segm_,np.where(vols<th_vol)[0],0)
    icells,vols = np.unique(dec.im_segm_,return_counts=True)
    dec.icells = icells[icells>0]
    dec.cms = np.array(nd.center_of_mass(dec.im_segm_>0,dec.im_segm_,dec.icells))
    dec.resc_segm = dec.im_segm_.shape/dec.shape
    
def get_best_X(cp,ifov,ifl=0,htag_ref = 'H1_R1',resc=10):
    cp.ifov=ifov
    fls = cp.dic_fls[ifov]
    fl = fls[ifl]
    cp.fl = fl
    cp.htag_ref = htag_ref
    cp.fov,cp.htag = os.path.basename(fl).split('--')[:2]
    best_guess_raw_data = os.path.dirname(cp.save_folder)
    fl_T_ref = best_guess_raw_data+os.sep+htag_ref+os.sep+cp.fov
    fl_T = best_guess_raw_data+os.sep+cp.htag+os.sep+cp.fov
    cp.obj = get_dapi_features(fl_T,cp.save_folder)
    cp.obj_ref = get_dapi_features(fl_T_ref,cp.save_folder)

    cp.X = cp.obj.Xh_plus[:,:3]
    cp.X_ref = cp.obj_ref.Xh_plus[:,:3]
    tzxy_plus = get_best_translation_points(cp.X,cp.X_ref,resc=resc)

    cp.X = cp.obj.Xh_minus[:,:3]
    cp.X_ref = cp.obj_ref.Xh_minus[:,:3]
    tzxy_minus = get_best_translation_points(cp.X,cp.X_ref,resc=resc)
    cp.tzxy_plus = tzxy_plus
    cp.tzxy_minus = tzxy_minus
    cp.tzxyf = (cp.tzxy_plus+cp.tzxy_minus)/2
def load_Xfovs(cp,transpose=1,flipx=-1,flipy=1):
    cp.dic_pos = np.load(cp.save_folder+os.sep+'dic_pos.pkl',allow_pickle=True)
    cp.dic_pos = {fov:np.array(cp.dic_pos[fov])[::transpose]*[flipx,flipy] for fov in cp.dic_pos}
    #Xfov = cp.dic_pos[cp.fov]
    fovs = list(cp.dic_pos.keys())
    Xfovs = np.array([cp.dic_pos[fov] for fov in fovs])
    cp.Xfovs=Xfovs
    cp.fovs = fovs
def get_abs_pos_cells(cp,pix_size = 0.10833):
    cp.center = cp.shape[1:]/2
    cp.Xcells = (cp.cms[:,1:]/cp.resc_segm[1:]+cp.tzxyf[1:]-cp.center)*pix_size
    cp.Xcells = cp.Xcells+cp.dic_pos[cp.fov]
    cp.pix_size=pix_size
    cp.dic_Xcells = dict(zip(cp.icells,cp.Xcells))
def get_dic_fov_cells(cp):
    """
    This looks through all the cells in cp.im_segm_ and finds the best fov corresponding to each cell
    Data is stored in cp.dic_fov_cells
    """
    Xfov = cp.dic_pos[cp.fov]
    out_of_fov = np.any(np.abs((cp.Xcells-Xfov)/cp.pix_size)>cp.shape[1:]/2,axis=-1)
    cp.icells_in_fov = cp.icells[~out_of_fov]
    cp.icells_out_fov = cp.icells[out_of_fov]
    dists,ifovs_cells = cKDTree(cp.Xfovs).query(cp.Xcells[out_of_fov])
    ifovsE,ctsIfovs = np.unique(ifovs_cells,return_counts=True)
    ifovsE = ifovsE[ctsIfovs>2]
    keep_cells = np.in1d(ifovs_cells,ifovsE)
    cp.icells_out_fov = cp.icells_out_fov[keep_cells]
    ifovs_cells = ifovs_cells[keep_cells]
    cp.dic_fov_cells = {cp.fovs[ifov]:cp.icells_out_fov[ifovs_cells==ifov] for ifov in ifovsE}
    cp.dic_fov_cells[cp.fov]=cp.icells_in_fov


def get_info_cp_fl(cp):
    cp.fovT,htag,coltag = os.path.basename(cp.fl).split('--')
    cp.icol = int(coltag.split('_')[0].replace('col',''))
    cp.iR = int(htag.replace('_','').split('R')[cp.icol+1])-1
    cp.ifovT = int(cp.fovT.split('_')[-1])
def load_Xh(cp):
    Xh = np.load(cp.fl)['Xh']
    get_info_cp_fl(cp)
    Xhf = None
    for icell in cp.dic_cell_driftf:
        drft,npts,fov_ = cp.dic_cell_driftf[icell]
        if cp.fovT==fov_:
            X_ = Xh[:,:3]-drft
            Xh_red = np.round(X_*cp.resc_segm).astype(int)
            keep_red = np.all((Xh_red<cp.im_segm_.shape)&(Xh_red>=0),axis=-1)
            
            icell_id_X = cp.im_segm_[tuple(Xh_red[keep_red].T)]
            Xh_ = Xh[keep_red][icell_id_X==icell]
            if len(Xh_):
                Xh_E = np.concatenate([Xh_[:,:3]-drft,Xh_[:,:3],[[cp.ifovT]]*len(Xh_),Xh_[:,3:],[[cp.icol,cp.iR,icell]]*len(Xh_)],axis=-1)
                Xhf = (Xh_E if Xhf is None else np.concatenate([Xhf,Xh_E]))
    cp.Xhf = Xhf



def get_cp_drfit(cp,fov,tzxyf):
    """The purpose of this is to populate cp.dic_cell_driftf such that each cell has the best guess of drift"""
    cp.fovT = fov
    icells_fov = cp.dic_fov_cells[cp.fovT]
    if not hasattr(cp,'dic_cell_drift_minus'):
        cp.dic_cell_drift_minus={}
    X = cp.obj.Xh_minus[:,:3]
    X_ref = cp.obj_ref.Xh_minus[:,:3]
    
    #X = cp.obj.Xh_plus[:,:3]
    #X_ref = cp.obj_ref.Xh_plus[:,:3]
    
    XT = X-tzxyf
    X_ref_red = np.round(X_ref*cp.resc_segm).astype(int)
    XT_red = np.round(XT*cp.resc_segm).astype(int)
    keep_ref = np.all((X_ref_red<cp.im_segm_.shape)&(X_ref_red>=0),axis=-1)
    keep_ld = np.all((XT_red<cp.im_segm_.shape)&(XT_red>=0),axis=-1)
    icell_id_X = cp.im_segm_[tuple(XT_red[keep_ld].T)]
    icell_id_Xref = cp.im_segm_[tuple(X_ref_red[keep_ref].T)]
    
    for icell in icells_fov:
        tzxy = [0,0,0]
        Npts = 0
        XTC = XT[keep_ld][icell_id_X==icell]
        XRC = X_ref[keep_ref][icell_id_Xref==icell]
        if len(XTC)>0 and len(XRC)>0:
            tzxy,Npts = get_best_translation_points(XTC,XRC,resc=5,learn=0.9,return_counts=True)
            #print(icell,tzxy,Npts)
        cp.tzxyf_cell = tzxyf+tzxy
        cp.Npts = Npts
        cp.dic_cell_drift_minus[icell]=[cp.tzxyf_cell,cp.Npts,cp.fovT]
    
    ####################################Plus - local maxima
    if not hasattr(cp,'dic_cell_drift_plus'):
        cp.dic_cell_drift_plus={}
    X = cp.obj.Xh_plus[:,:3]
    X_ref = cp.obj_ref.Xh_plus[:,:3]
    
   
    XT = X-tzxyf
    X_ref_red = np.round(X_ref*cp.resc_segm).astype(int)
    XT_red = np.round(XT*cp.resc_segm).astype(int)
    keep_ref = np.all((X_ref_red<cp.im_segm_.shape)&(X_ref_red>=0),axis=-1)
    keep_ld = np.all((XT_red<cp.im_segm_.shape)&(XT_red>=0),axis=-1)
    icell_id_X = cp.im_segm_[tuple(XT_red[keep_ld].T)]
    icell_id_Xref = cp.im_segm_[tuple(X_ref_red[keep_ref].T)]
    
    for icell in icells_fov:
        tzxy = [0,0,0]
        Npts = 0
        XTC = XT[keep_ld][icell_id_X==icell]
        XRC = X_ref[keep_ref][icell_id_Xref==icell]
        if len(XTC)>0 and len(XRC)>0:
            tzxy,Npts = get_best_translation_points(XTC,XRC,resc=5,learn=0.9,return_counts=True)
            #print(icell,tzxy,Npts)
        cp.tzxyf_cell = tzxyf+tzxy
        cp.Npts = Npts
        cp.dic_cell_drift_plus[icell]=[cp.tzxyf_cell,cp.Npts,cp.fovT]
    def combine_drft(X1n1fov,X2n2fov,ncutoff=10):
        X1,n1,fov=X1n1fov
        X2,n2,fov=X2n2fov
        if n1>ncutoff and n2>ncutoff:
            return [(X1*n1+X2*n2)/(n1+n2),n1+n2,fov]
        else:
            if n1>ncutoff:
                return X1n1fov
            if n2>ncutoff:
                return X1n1fov
            return [[np.nan,np.nan,np.nan],0,fov]

    if not hasattr(cp,'dic_cell_driftf'):
        cp.dic_cell_driftf={}
    cp.dic_cell_driftf.update({icell:combine_drft(cp.dic_cell_drift_plus.get(icell,[[np.nan,np.nan,np.nan],0,cp.fovT]),
                        cp.dic_cell_drift_minus.get(icell,[[np.nan,np.nan,np.nan],0,cp.fovT]),ncutoff=10) for icell in icells_fov})

    bad_info = [(icell,cp.dic_Xcells[icell])for icell in cp.dic_cell_driftf if np.isnan(cp.dic_cell_driftf[icell][0][0])]
    good_info  = [(icell,cp.dic_Xcells[icell])for icell in cp.dic_cell_driftf if ~np.isnan(cp.dic_cell_driftf[icell][0][0])]
    if (len(bad_info)>0) and (len(good_info)>0):
        bad_icells,bad_X = zip(*bad_info)
        good_icells,good_X = zip(*good_info)
        dsts,inds = cKDTree(good_X).query(bad_X)
        for bad_i,good_i in zip(bad_icells,np.array(good_icells)[inds]):
            cp.dic_cell_driftf[bad_i]=cp.dic_cell_driftf[good_i]

    if False:
        ### plot the dependence of drift on position
        x_,y_ = zip(*[(cp.dic_cell_driftf[icell][0][2],cp.dic_Xcells[icell][1]) for icell in cp.dic_cell_driftf])
        plt.plot(x_,y_,'o')
def get_best_XV2(cp,resc=10):
    best_guess_raw_data = os.path.dirname(cp.save_folder)
    fl_T = best_guess_raw_data+os.sep+cp.htag+os.sep+cp.fovT
    cp.obj = get_dapi_features(fl_T,cp.save_folder)
    
    cp.X = cp.obj.Xh_plus[:,:3]
    cp.X_ref = cp.obj_ref.Xh_plus[:,:3]
    tzxy_plus = get_best_translation_points(cp.X,cp.X_ref,resc=resc)

    cp.X = cp.obj.Xh_minus[:,:3]
    cp.X_ref = cp.obj_ref.Xh_minus[:,:3]
    tzxy_minus = get_best_translation_points(cp.X,cp.X_ref,resc=resc)
    cp.tzxy_plusT = tzxy_plus
    cp.tzxy_minusT = tzxy_minus
    cp.tzxyfT = (cp.tzxy_plusT+cp.tzxy_minusT)/2
    
    

def get_im_from_Xh(Xh,resc=5,pad=10):
    X = np.round(Xh[:,:3]/resc).astype(int)
    Xm = np.min(X,axis=0)
    XM = np.max(X,axis=0)
    keep = X<=(XM-[0,pad,pad])
    keep &= X>=(Xm+[0,pad,pad])
    keep = np.all(keep,-1)
    X = X[keep]
    if False:
        Xm = np.min(X,axis=0)
        X-=Xm
    else:
        Xm=np.array([0,0,0])
    
    sz = np.max(X,axis=0)
    imf = np.zeros(sz+1,dtype=np.float32)
    imf[tuple(X.T)]=1
    return imf,Xm
def get_Xtzxy(X,X_ref,tzxy0,resc,target=3):
    tzxy = tzxy0
    Npts =0
    for dist_th in np.linspace(resc,target,5):
        XT = X-tzxy
        ds,inds = cKDTree(X_ref).query(XT)
        keep = ds<dist_th
        X_ref_ = X_ref[inds[keep]]
        X_ = X[keep]
        tzxy = np.mean(X_-X_ref_,axis=0)
        #print(tzxy)
        Npts = np.sum(keep)
    return tzxy,Npts
def get_best_translation_points(X,X_ref,resc=5,pad=10,target=3,return_counts=False):
    
    im,Xm = get_im_from_Xh(X,resc=resc,pad=pad)
    im_ref,Xm_ref = get_im_from_Xh(X_ref,resc=resc,pad=pad)
    
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im,im_ref[::-1,::-1,::-1])
    #plt.imshow(np.max(im_cor,0))
    tzxy = np.array(np.unravel_index(np.argmax(im_cor),im_cor.shape))-im_ref.shape+1+Xm-Xm_ref
    tzxy = tzxy*resc
    Npts=0
    tzxy,Npts = get_Xtzxy(X,X_ref,tzxy,resc=resc,target=target)
    if return_counts:
        return tzxy,Npts
    return tzxy
