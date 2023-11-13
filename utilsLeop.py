#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import glob
import matplotlib.pyplot as plt
from mrcnn import visualize


from matplotlib.patches import Polygon
from skimage.measure import find_contours
from matplotlib import patches,  lines
from scipy.io import loadmat

import xml.dom.minidom as minidom

from lxml.etree import Element,SubElement,tostring
from xml.dom.minidom import parseString
############################################################
#  Configurations
############################################################


class LeopardDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_leopard(self,dataset_dir,subset,img_type):
        """Load a subset of the Leopard dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("leopard", 1, "leopard")

#         # Train or validation set
#         assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir,subset)
        

        # Add images
        img_source = str(dataset_dir) +'/*.'+ str(img_type)

        for img_path in glob.iglob(img_source):    ### i means file name
            img_id = img_path[len(dataset_dir)+1:]
#             img = skimage.io.imread(img_path)
#             height,width = img.shape[:2]

            self.add_image(
                source = "leopard",
                image_id = img_id,          ### use file name as a unique image id
                path = img_path)
    
#                 self.add_image(
#                 source = "leopard",
#                 image_id = img_id,          ### use file name as a unique image id
#                 path = img_path,
#                 width = width, height = height)
                ### requires: soource, id, path; (utils.py)
    
    
    def load_polygon(self,img_id):
  
        """Generate approximate object polygons for an image.
        Returns:
        polygon: 
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[img_id]
        path = image_info["path"]
        image_name = image_info["id"]
        polygon_path = str(path[:-len(image_name)]) + 'polygon/' + str(image_name[:-4]) + '.xml'
        domTree = minidom.parse(polygon_path)

        root = domTree.documentElement
        
        name_obj = []
        xy_obj = []
        name_lis = root.getElementsByTagName('name')
        for i in range(len(name_lis)):
            leopard_class = name_lis[i].childNodes[0].nodeValue
            if leopard_class == 'R-leopard' or leopard_class == 'L-leopard':
                name_obj.append('leopard')
                xy = []
                for ele in root.getElementsByTagName('polygon')[i].childNodes:
                    if ele.nodeName != '#text':
                        xy.append(ele.childNodes[0].nodeValue)   ## xy coordinate 
                x_idx = xy[::2]
                y_idx = xy[1::2]
                xy_obj.append([x_idx,y_idx])
        height = int(root.getElementsByTagName('height')[0].childNodes[0].nodeValue)
        width = int(root.getElementsByTagName('width')[0].childNodes[0].nodeValue)
        return xy_obj,name_obj,height,width

    
                
    def load_mask(self,img_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        xy_obj,name_obj,height,width = self.load_polygon(img_id)

        num = len(name_obj)
        mask = np.zeros([height,width,num],dtype=np.uint8)
        
        for i in range(len(xy_obj)):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(np.array(xy_obj[i][1], dtype = int), np.array(xy_obj[i][0], dtype = int))
            mask[rr, cc, i] = 1

        mask = mask.astype(np.bool)
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in name_obj])
        return mask,class_ids

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     _,ax = plt.subplots(rows,cols)    # More details shows in plt.figure
    # figsize: width,height in inches
    return ax



def save_instances(image, boxes, masks, class_ids, class_names,save_path,
                      scores=None, title="", figsize=(16,16),
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,Evaluate_results=""):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    _,ax = plt.subplots(1,1,figsize=figsize,dpi=100)

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=14, backgroundcolor="none")
        ax.text(0,30,Evaluate_results,color='r',size=14,backgroundcolor="none") 

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    fig = plt.gcf()
    fig.savefig(save_path,transparent=True)
    plt.close()
    
def evaluate_MaskRCNN(gt_mask, pred_mask):
    """Computes IoU overlaps between two sets of masks.
    masks1=gt_mask, masks2=pred_mask: [Height, Width, instances]
    Notes: Intersection over union, also known as Jaccard similarity coefficient
    IoU is the ratio of correctly classified pixels to 
        the total number of groundtruth and predicted pixels in that class
    """
    # Calculate leopard class
    # If either set of masks is empty return empty result
    if gt_mask.shape[-1] == 0 or pred_mask.shape[-1] == 0:
        return np.zeros((gt_mask.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(gt_mask > .5, (-1, gt_mask.shape[-1])).astype(np.float32)
    masks2 = np.reshape(pred_mask > .5, (-1, pred_mask.shape[-1])).astype(np.float32)
    masks2_total = np.zeros(masks2.shape[0])
    for i in range(masks2.shape[-1]):
        masks2_total = masks2_total + masks2[:,i]
    area2_total = np.sum(masks2_total, axis=0)
    area1 = np.sum(masks1, axis=0)

    # intersections and union
    # intersections: correctly classified pixels
    # union: the total number of groundtruth and predicted pixels
    intersections_fg = np.dot(masks1.T, masks2_total)
    union = area1[:, None] + area2_total - intersections_fg
    fg_IoU = intersections_fg / union
    
    ## Calculate background class
    masks1_bg = abs(masks1-1)
    masks2_bg = abs(masks2_total-1)
    area2_bg = np.sum(masks2_bg, axis=0)
    area1_bg = np.sum(masks1_bg, axis=0)

    # intersections and union
    intersections_bg = np.dot(masks1_bg.T, masks2_bg)
    union_bg = area1_bg + area2_bg - intersections_bg
    bg_IoU = intersections_bg / union_bg
    
    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    area2 = np.sum(masks2, axis=0)
    union = area1[:, None] + area2[None, :] - intersections
    IoU_instance = intersections / union

    """Computes accuracy between two sets of masks.
    masks1=gt_mask, masks2=pred_mask: [Height, Width, instances]
    Note: Accuracy is the ratio of correctly classified pixels 
        to the total number of pixels in that class
    """
    # intersections
    fg_acc = intersections_fg / area1[:,None]
    bg_acc = intersections_bg / area1_bg
    global_acc = (intersections_fg[0] + intersections_bg) / (area1[:,None][0] + area1_bg)
    mean_acc = (fg_acc+bg_acc)/2
    acc_instance = intersections / area1[:,None]
    fg_count = area1[:, None][0][0]
    bg_count = area1_bg[0]
    count = [fg_count,bg_count]
    meanIoU = (fg_IoU+bg_IoU)/2
    weightedIoU = (fg_IoU*fg_count+bg_IoU*bg_count)/(fg_count+bg_count)
    
    return IoU_instance,fg_IoU,bg_IoU,weightedIoU,meanIoU,acc_instance,fg_acc,bg_acc,global_acc,mean_acc,count