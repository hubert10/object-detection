"""
This script reads merged images and annots
and writes them in separate folders: images and annots

Steps
 1.  Download fruits datasets from Kaggle: https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection/download?datasetVersionNumber=1
 2.  Merge both test and train together in one folder named datasets
 3.  Create destination folders: images and annots
"""

import os
import glob
import shutil
import random

data_dir = "fruits/datasets"
images_dir = "fruits/images"
annots_dir = "fruits/annots"

# Copy-pasting images
list_of_files = os.listdir(data_dir)

for file_ in list_of_files:
    # save both images and annotations in separate folders
    if file_.endswith(".jpg"):
        shutil.copy(
            os.path.abspath(data_dir + "/" + file_),
            os.path.abspath(images_dir + "/" + file_),
        )
    else:
        shutil.copy(
            os.path.abspath(data_dir + "/" + file_),
            os.path.abspath(annots_dir + "/" + file_),
        )
