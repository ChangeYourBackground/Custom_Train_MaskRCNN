import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

import matplotlib.pyplot as plt

# Keras shit
from keras.preprocessing.image import load_img,img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

# define 81 classes that the coco model knowns about
class_names = ['cockatoo']
 
# define the test configuration
class TestConfig(Config):
    NAME = 'cockatoo'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1
 	
def main():
    # array = sys.argv[1:]
    #
    # if os.path.exists(array[0]):
    #     path_to_weight = array[0]
    #     # sys.exit(0)
    # else:
    #     print('path to weight does not exist')
    #     sys.exit(0)
    # if os.path.exists(array[1]):
    #     path_to_image =array[1]
    # else:
    #     print('path to image does not exist')
    #     sys.exit(0)
    # if 1 >= float(array[2]) >= 0:
    #     conf=array[2]
    # else:
    #     print('confidence must be a float')
    #     sys.exit(0)
    config = TestConfig()
    config.DETECTION_MIN_CONFIDENCE = 0.5

    # define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=config)
    # load coco model weights
    rcnn.load_weights('./logs/cockatoo20210609T2039/mask_rcnn_cockatoo_0010.h5', by_name=True)
    # load photograph
    img = load_img('./dataset/val/images/5v.jpg')
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=1)
    # get dictionary for first prediction
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    display_instances(img, r['rois'], r['masks'], r['class_ids'], 'cockatoo', r['scores'])

if __name__ == '__main__':
    main()

