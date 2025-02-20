from PIL import Image
import numpy as np
import os
from utils import BasicDataset


img_dir = r"D:\flood-area-segmentation\Image"
mask_dir = r"D:\flood-area-segmentation\Mask"

dataset = BasicDataset(img_dir, mask_dir, mask_format='.png')

for data in dataset:
    np_mask = np.array(data['mask'])
    values = np_mask.flatten()
    print(np.unique(values))
