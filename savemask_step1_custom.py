#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:53:29 2023

@author: obaiga
"""

# In[packs]
import numpy as np
from scipy.io import loadmat
import glob
from os.path import join,exists 
import skimage.io 
import pandas as pd
from os import makedirs
import shutil

import pyhesaff

from skimage.measure import find_contours
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib import patches,  lines



# In[paths]
db_dir = '/Users/obaiga/Research/Snow Leopard/CAT IDs_zip'
# mask_dir = '/Users/obaiga/github/Automatic-individual-animal-identification-maskrcnn/Maskrcnn/results/leopard/test/'

subset = 'Cat 3'
print(subset)

img_dir = join(db_dir,subset)
mask_dir = join(img_dir,'annotation')

# img_lis = glob.glob(join(img_dir,'*.JPG'))
img_lis=[join(img_dir,'02__Station09__Camera2__2012-9-16__18-51-12(2).JPG'),]

# roi_hs_lis = []
query_save_name = join(db_dir,'query.JPG')

mask_save_dir = join(img_dir,'mask')
checkmask_save_dir = join(img_dir,'mask_polygon')

if not exists(mask_save_dir):
    makedirs(mask_save_dir)
if not exists(checkmask_save_dir):
    makedirs(checkmask_save_dir)

delete_dir = join(img_dir,'delete')
if not exists(delete_dir):
    makedirs(delete_dir)

#%%
# for ii,imaskdir in enumerate(mask_lis[:1]):
for ii,iimgpath in enumerate(img_lis):
    imgname = iimgpath[len(img_dir)+1:]
    # print(imgname)
    imaskdir = join(mask_dir,(imgname[:-4]+'.mat'))
    # try:
    result = loadmat(imaskdir)

    classids = result['class_ids']
    rois = result['rois']    #### roi represent: y_min, x_min, y_max, x_max
    masks = result['masks']
    nmask = len(classids[0])
    image = skimage.io.imread(iimgpath)
    
    if nmask > 1:
        # print(ii,len(classids[0])) 
        nkpts_lis = np.ones(nmask) * -1
        for jj in np.arange(nmask):
            mask_3d = np.zeros((masks[:,:,jj].shape[0], masks[:,:,jj].shape[1],3))
            mask_3d[:,:,0] = masks[:,:,jj]
            mask_3d[:,:,1] = masks[:,:,jj]
            mask_3d[:,:,2] = masks[:,:,jj]
            obj_image = image * mask_3d
            
            skimage.io.imsave(query_save_name, obj_image.astype(np.uint8))
            
            kpts,_ = pyhesaff.detect_feats(query_save_name)
            nkpts_lis[jj] = len(kpts)
            
        bst = np.argmax(nkpts_lis)
        # bst = np.argmin(nkpts_lis)
    else:
        bst = 0
        
    req_mask = masks[:,:,bst]
    
    # ###---------------------PLOT-------------
    if 1:
        _,ax = plt.subplots(1,1,figsize=(16,12),dpi=100)
        for jj in range(nmask): 
            padded_mask = np.zeros(
                (masks.shape[0], masks.shape[1]), dtype=np.uint8)
            padded_mask = masks[:,:,jj]
            ans = np.where(padded_mask==1)
            ymin = np.min(ans[0])
            ymax = np.max(ans[0])
            xmin = np.min(ans[1])
            xmax = np.max(ans[1])
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                if jj == bst:
                    color = 'r'
                else:
                    color = 'blue'
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
                p = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)
        plt.xticks([]),plt.yticks([])
        ax.imshow(image.astype(np.uint8))
        fig = plt.gcf()
        save_path = join(checkmask_save_dir,imgname)
        fig.savefig(save_path,dpi=100,transparent=True,bbox_inches = 'tight')
        plt.close()
    # ####-------------------     
    
    save_name = join(mask_save_dir,imgname)
    skimage.io.imsave(save_name, (req_mask*255).astype(np.uint8))
    
    # except:
    #     print('no mask (%d):%s'%(ii,imgname))
    #     # Move the file
    #     shutil.move(iimgpath, delete_dir)
        
    
