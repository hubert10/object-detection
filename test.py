import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Set this to True to see more logs details
os.environ["AUTOGRAPH_VERBOSITY"] = "5"
tf.autograph.set_verbosity(3, False)
tf.cast
import warnings

warnings.filterwarnings("ignore")
#from utils.config import CustomConfig

tf.compat.v1.disable_eager_execution()


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
#BASE_DIR = os.path.dirname(os.path.abspath('./test.ipynb'))
#PROJECT_ROOT = BASE_DIR + ""
#print(PROJECT_ROOT)

# Get the project root directory
#project_path = PROJECT_ROOT
#RCNN_ROOT = os.path.abspath(project_path + "Mask_RCNN")
os.chdir("./Mask_RCNN")
print("Printing the current project root dir".format(os.getcwd()))

# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log
from PIL import Image, ImageDraw
import cv2
import os

CLASS_NAMES = ['building']


#class SimpleConfig(Config):
#    NAME = "coco_inference"

#    GPU_COUNT = 1
#   IMAGES_PER_GPU = 1
#   NUM_CLASSES = len(CLASS_NAMES)