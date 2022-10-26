"""
This script reads merged images and annots
and writes them in separate folders: images and annots
"""

import os, glob, random, shutil

image_dir = "fruits/data/images"
annot_dir = "fruits/data/annots"

renamed_im_dir = "fruits/renamed_to_numbers/images"
renamed_ann_dir = "fruits/renamed_to_numbers/annots"

image_path_list = glob.glob(os.path.join(image_dir, "*.jpg"))
random.Random(42).shuffle(
    image_path_list
)  # setting the seed is important to make sure we
padding = len(str(len(image_path_list)))  # number of digits to add for file number

for n, filepath in enumerate(image_path_list, 1):
    
    # Loop over images and for each image we find
    # the corresponding annotation right before
    # renaming it to numbers, we want to make sure that 
    # the new label assigned to image alighs with its annot
    
    annot_filename = filepath.split("/")[-1].split(".")[0]
    annot_file_path = os.path.join(annot_dir, annot_filename + ".xml")
    shutil.copy(
        os.path.abspath(filepath),
        os.path.join(renamed_im_dir, "{:>0{}}.jpg").format(n, padding),
    )
    shutil.copy(
        os.path.abspath(annot_file_path),
        os.path.join(renamed_annot_dir, "{:>0{}}.xml").format(n, padding),
    )
