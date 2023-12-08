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

import pyhesaff

# In[paths]
db_dir = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/snowleop/'
mask_dir = '/Users/obaiga/github/Automatic-individual-animal-identification-maskrcnn/Maskrcnn/results/leopard/test/'

img_dir = join(db_dir,'images_db')

#%%
name = 'raw_data.csv'
table_dir = join(db_dir,name)
table = pd.read_csv(table_dir,skipinitialspace=True)
# print(table.columns.tolist())
img_lis = table['Image']

#%%
# mask_lis = glob.glob(join(mask_dir,'*.mat'))
roi_hs_lis = []
query_save_name = join(db_dir,'query.JPG')

mask_save_dir = join(db_dir,'mask')
if not exists(mask_save_dir):
    makedirs(mask_save_dir)

# for ii,imaskdir in enumerate(mask_lis[:1]):
for ii,imgname in enumerate(img_lis):
    imaskdir = join(mask_dir,imgname[:-4]+'.mat')
    result = loadmat(imaskdir)
    classids = result['class_ids']
    rois = result['rois']    #### roi represent: y_min, x_min, y_max, x_max
    masks = result['masks']
    try:
        nmask = len(classids[0])
        if nmask > 1:
            # print(ii,len(classids[0]))
            image = skimage.io.imread(join(img_dir,imgname))
            
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
            req_mask = masks[:,:,bst]
        else:
            req_mask = masks[:,:,0]
    except:
        print('no mask (%d):%s'%(ii,imgname))
        
        image = skimage.io.imread(join(img_dir,imgname))
        # xmin,ymin = 0, 0 
        # xmax,ymax = image.shape[1], image.shape[0]   ### width, height
        req_mask = np.ones((image.shape[0],image.shape[1]),dtype=np.uint8)
        
    save_name = join(mask_save_dir,imgname)
    skimage.io.imsave(save_name, (req_mask*255).astype(np.uint8))
    
    # Find white pixel indices
    y_indices, x_indices = np.where(req_mask == 1)
    
    # Find min and max indices
    xmin, xmax = x_indices.min(), x_indices.max()
    ymin, ymax = y_indices.min(), y_indices.max()
    
    ## for hotspotter ROI
    xywh = list(map(int, map(round, (xmin, ymin, xmax - xmin, ymax - ymin)))) 
    roi_hs = np.array(xywh, dtype=np.int32)
    roi_hs_lis.append(str(roi_hs))
    
#%%
table['roi[tl_x  tl_y  w  h]'] = roi_hs_lis

#%%

table.to_csv(join(db_dir,'raw_data.csv'), index=False)
