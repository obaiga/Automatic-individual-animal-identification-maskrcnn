#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Mask R-CNN - Inference on Leopard Dataset
"""


# In[packs]:
# IMPORTS
################################################################################
# basic python
from os.path import abspath,join,exists 
from os import makedirs
from scipy.io import loadmat,savemat

import copy
from os import chdir
main_dir = 'C:\\Users\SHF\Documents\GitHub\Automatic-individual-animal-identification-maskrcnn\Maskrcnn'
chdir(main_dir)

from mrcnn.config import Config
# from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# from mrcnn.model import log

from customPacks import utilsLeop
########################### END IMPORTS ########################################


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

class InferenceConfig(LeopardConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    


#%%
# Directory to save logs and trained model
MODEL_DIR = join(main_dir, "logs")

Model_Name ="mask_rcnn_snowleopard_0001.h5"
# Local path to trained weights file
LEOPARD_MODEL_PATH = join(MODEL_DIR,Model_Name)

print(LEOPARD_MODEL_PATH)

config = LeopardConfig()
config.display()

#%%

# Create model in inference mode
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model.load_weights(LEOPARD_MODEL_PATH, by_name=True)
print("Loading weights from ", LEOPARD_MODEL_PATH)

#%%
#### Dataset
IMG_TYPE = 'JPG'
LEOPARD_DIR = join(main_dir,"datasets/leopard")

dataset_train = utilsLeop.LeopardDataset()
dataset_train.load_leopard(LEOPARD_DIR,"train",IMG_TYPE)
dataset_train.prepare()

dataset_val = utilsLeop.LeopardDataset()
dataset_val.load_leopard(LEOPARD_DIR,"val",IMG_TYPE)
dataset_val.prepare()

dataset_test = utilsLeop.LeopardDataset()
dataset_test.load_leopard(LEOPARD_DIR,"test",IMG_TYPE)
dataset_test.prepare()

print("Training Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Validation Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
print("Test Image Count: {}".format(len(dataset_test.image_ids)))
print("Class Count: {}".format(dataset_test.num_classes))
for i, info in enumerate(dataset_test.class_info):
    print("{:3}. {:50}".format(i, info['name']))
#%%
# Run object detection
save_dir = join(main_dir,'results','leopard','test')
if not exists(save_dir):
    makedirs(save_dir)
    
req = copy.copy(dataset_test)

for image_id in req.image_ids:
    image = req.load_image(image_id)    #source from utils.py
    result = model.detect([image])  
    image_name = req.image_info[image_id]['id'][:-4]
    
    if len(result) > 1:
        print(image_id,len(result))
    else:
        savemat(join(save_dir,image_name+'.mat'), result[0])
    
    #%%
# image = dataset_val.load_image(image_id)    #source from utils.py
# mask, class_ids = dataset_val.load_mask(image_id)
visualize.display_top_masks(image, result[0]['masks'], result[0]['class_ids'],
                            dataset_val.class_names,limit=1)


