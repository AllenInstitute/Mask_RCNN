import os
os.chdir('/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN')
import datetime
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
from skimage.measure import regionprops

# Root directory of the project
ROOT_DIR = os.path.abspath('/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.model import log
VES_DIRECT = os.path.abspath('/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN/samples/vesicles')
sys.path.append(VES_DIRECT)
import vesicle

def setup_logs_dir():
    LOGS_DIR = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN/logs'
    return LOGS_DIR

import pandas as pd
from PIL import Image
#########################################################################################################

# Configuration
def setup_config():
    DATASET_DIR = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo'


    # Inference Configuration

    config = vesicle.NucleusInferenceConfig()
    return config


#########################################################################################################
# Load test dataset

def load_dataset():

    dataset = vesicle.NucleusDataset()
    DATASET_DIR = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo'
    dataset.load_nucleus(DATASET_DIR, 'TestPilot2')
    dataset.prepare()
    return dataset
    
#########################################################################################################
# Load model

def load_model(LOGS_DIR, config):
    DEVICE = "/gpu:0"
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=LOGS_DIR,
                                  config=config)
    return model

def load_weights(model):
    weights_path= '/allen/programs/braintv/workgroups/neuralcoding/Parastoo/Mask_RCNN/samples/vesicles/logs/vesicle20210215T1012/mask_rcnn_vesicle_0175.h5'
   
    # Load weights
    model.load_weights(weights_path, by_name=True)

#########################################################################################################

def gen_ctr_pts(pred_masks):
    
    ctr_pts = []
    for i in range(pred_masks.shape[2]):
        region = regionprops(pred_masks[:, :, i])
        try:
            y, x = region[0].centroid
            ctr_pt = [x, y, 0]
            ctr_pts.append(ctr_pt)
        except IndexError:
            return;
    return ctr_pts

def make_results_df(results, img_id):
    
    # define the dataframe and attributes
    df = pd.DataFrame(columns = ['img_id', 'area_vox', 'ctr_pts', 'p_score'])
    pred_masks = results['masks']
    scores = results['scores']
    masks2 = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    area = np.sum(masks2, axis=0)
    df.p_score = scores
    df.area_vox = area
    df.ctr_pts = gen_ctr_pts(pred_masks.astype(int))
    df.img_id = img_id
    
    return df

    

def save_masks(pred_masks, img_id, dir_path, file_exists=True):
    
    
    if file_exists==False:
        os.mkdir(dir_path + img_id + '/' + 'masks')
        pred_path = dir_path + img_id + '/' + 'masks'
    else:
        pred_path = dir_path + img_id + '/' + 'masks'
        
    for i in range(pred_masks.shape[2]):
        mask = pred_masks[:, :, i]
        pil_mask = Image.fromarray(mask)
        pil_mask.save('{}/{}_{}.png'.format(pred_path,img_id,i))
        
#########################################################################################################

def run_inference():

    dir_path = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo/TestPilot2/'
    ves_path = '/allen/programs/braintv/workgroups/neuralcoding/Parastoo/TestPilot2/meta_data_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d%H%M"))


    LOGS_DIR = setup_logs_dir()

    # Configuration
    config = setup_config()

    # Load test dataset
    dataset = load_dataset()

    # Load Model
    model = load_model(LOGS_DIR, config)

    # Load weights
    load_weights(model)
    
    ves_nglui = pd.DataFrame(columns = ['img_id', 'area_vox', 'ctr_pts', 'p_score'])
    
    for image_id in dataset.image_ids:
        img_id = dataset.image_info[image_id]['id']
        if img_id not in np.unique(ves_nglui.img_id):
            image = dataset.load_image(image_id)

            # Detect objects
            r = model.detect([image], verbose=0)[0]

            try:
                try:
                    df = make_results_df(r, img_id)

                except FileNotFoundError:
                     df = make_results_df(r, img_id, file_exists=False)

                ves_nglui = ves_nglui.append(df)
                ves_nglui.to_csv(ves_path)
                save_masks(r['masks'], img_id, dir_path = dir_path)


            except ValueError:
                pass

if __name__ == '__main__':
    
    run_inference()