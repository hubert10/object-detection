import os
import xml.etree
from numpy import zeros, asarray

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR + "/"
print(f"Reading the current root dir: {PROJECT_ROOT}.")

# Get the project root directory
project_path = PROJECT_ROOT
RCNN_ROOT = os.path.abspath(project_path)
os.chdir(RCNN_ROOT)

from  mrcnn import utils
from  mrcnn import config
from  mrcnn import model

KANGAROO_ROOT = os.path.join(RCNN_ROOT)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(RCNN_ROOT, "kangaroo-transfer-learning/mask_rcnn_coco.h5")
# Download COCO trainedKANGAROO_ROOT weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory to save logs and trained model while training for backup
DEFAULT_LOGS_DIR = os.path.join(KANGAROO_ROOT, "logs")

def create_dirs(path_str):
    path_dir = os.path.join(path_str)
    if os.path.exists(path_dir):
        print("dir exists?", os.path.exists(path_dir)) 
    else:
        os.makedirs(path_dir)

# Create directory to save logs and trained model if does not exists

create_dirs(DEFAULT_LOGS_DIR)

class KangarooDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "kangaroo")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= 150:
                continue

            if not is_train and int(image_id) < 150:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            print(img_path)
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class KangarooConfig(config.Config):
    NAME = "kangaroo_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 131

# Train
train_dataset = KangarooDataset()
train_dataset.load_dataset(dataset_dir='kangaroo-transfer-learning/kangaroo', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = KangarooDataset()
validation_dataset.load_dataset(dataset_dir='kangaroo-transfer-learning/kangaroo', is_train=False)
validation_dataset.prepare()

# Model Configuration
kangaroo_config = KangarooConfig()

# Build the Mask R-CNN Model Architecture
model = model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=kangaroo_config)

model.load_weights(filepath='kangaroo-transfer-learning/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=kangaroo_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

# After the training the model will be saved in the root folder
model_path = 'Kangaro_mask_rcnn_trained.h5'
print(model_path)
model.keras_model.save_weights(model_path)
