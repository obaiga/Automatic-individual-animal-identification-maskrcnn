#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Mask R-CNN - Train on Leopard Dataset
"""


# In[packs]:
# IMPORTS
################################################################################
# basic python
from os.path import abspath,join,exists 
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import skimage.draw
from scipy.io import loadmat

from os import chdir
main_dir = 'C:\\Users\SHF\Documents\GitHub\Automatic-individual-animal-identification-maskrcnn\Maskrcnn'
chdir(main_dir)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from customPacks import utilsLeop
########################### END IMPORTS ########################################

#%%
## Check whether GPU is being or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# In[Config]:
# CLASS LeopardConfig
################################################################################

class LeopardConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "leopard"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + leopard

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000   # default value: 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50   # default value: 50
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5   # default value = 0.7
    
########################### END CLASS LeopardConfig ###############################


# In[ ]:
# MAIN
################################################################################
    
# Directory to save logs and trained model
MODEL_DIR = join(main_dir, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = join(MODEL_DIR, "leopard_mask.h5")
# Download COCO trained weights from Releases if needed
if not exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

LEOPARD_DIR = join(main_dir,"datasets/leopard")

config = LeopardConfig()
    
    #%%
#### Dataset
IMG_TYPE = 'JPG'

dataset_train = utilsLeop.LeopardDataset()
dataset_train.load_leopard(LEOPARD_DIR,"train",IMG_TYPE)
dataset_train.prepare()

dataset_val = utilsLeop.LeopardDataset()
dataset_val.load_leopard(LEOPARD_DIR,"val",IMG_TYPE)
dataset_val.prepare()

#%%
print("Training Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Validation Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
# In[]
# Load and display random samples
image_ids = np.random.choice(dataset_val.image_ids, 1)
##self._image_ids = np.arange(self.num_images)

for image_id in image_ids:
    image = dataset_val.load_image(image_id)    #source from utils.py
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names,limit=1)

#%%
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
init_with = "last"

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    #### Load weights trained on MS COCO, but skip layers that
    ##### are different due to the different number of classes
    ##### See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    # model.load_weights(COCO_MODEL_PATH, by_name=True)
    
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
#%%

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

#%%
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")

############################### END  MAIN ######################################

