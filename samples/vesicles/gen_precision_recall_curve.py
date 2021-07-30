import tensorflow as tf
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import vesicle

%matplotlib inline 

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import pandas as pd

#########################################################################################################

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

#########################################################################################################

def gen_pred_results(dataset, image_id, model):
    val_error=[]
     try:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
        gt_match, pred_match, overlaps = utils.compute_matches(
                                                                gt_bbox, gt_class_id, gt_mask,
                                                                r["rois"], r["class_ids"], 
                                                                r["scores"], r['masks'],
                                                                iou_threshold=0.5
                                                                                    )
        except ValueError:
        print('Value Error was raised again')
        val_error.append(image_id) # ???????? How to deal with ValueError 
        
        return pred_match, r
    

def populate_model_score_table(dataset, image_id, model_score_tb, model):
    
    
    pred_match, r = gen_pred_results(dataset, image_id, model)
    

    for i in range(len(pred_match)):
        model_score_tb.loc[len(model_score_tb), ['image_id', 'scores', 'class_id', 'overlaps', 'TP']] = dataset.image_info[image_id]['id'], r['scores'][i], r['class_ids'][i], np.max(overlaps), pred_match[i]

    # Compute AP over range 0.5 to 0.95 and print it
    utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                           r['rois'], r['class_ids'], r['scores'], r['masks'],
                           verbose=1)
    
    

        

def create_model_score_table(dataset, model):
    
    for image_id in dataset.image_ids:
        model_score_tb = pd.DataFrame(columns = ['image_id', 'scores','class_id','overlaps', 'TP', 'FP'])
        print(image_id)
        populate_model_score_table(dataset, image_id, model_score_tb, model)
    return model_score_table


def plot_prc()

#########################################################################################################

if __main__=='__name__':
    
    
    # Configuration
    # Dataset directory
    
    DATASET_DIR = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo'
  
    # Inference Configuration
    config = vesicle.VesicleInferenceConfig()
    config.display()

    # Load test dataset

    dataset = vesicle.VesicleDataset()
    dataset.load_vesicle(DATASET_DIR, 'threeD_test')
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Load model

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=LOGS_DIR,
                                  config=config)


    weights_path ='/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN/samples/vesicles/logs/vesicle20210215T1012/mask_rcnn_vesicle_0175.h5'
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    print('************  Model is loaded  **************')
    
    model_score_table = create_model_score_table(dataset, model)
    
    
    
    