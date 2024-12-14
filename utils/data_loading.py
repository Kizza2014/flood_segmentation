from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import os
from os.path import splitext, isfile, join
import logging
import numpy as np


class BasicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_format='.jpg', mask_format='.jpg', transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_format = img_format
        self.mask_format = mask_format
        self.transform = transform

        self.img_files = [splitext(file)[0] for file in os.listdir(img_dir) if isfile(join(img_dir, file)) and file.endswith(img_format)] 
        if not self.img_files:
            raise RuntimeError(f'No input file found in {img_dir}, make sure you put your images here.')

        logging.info(f'Creating dataset with {len(self.img_files)} examples.')


    def __len__(self):
        return len(self.img_files)


    @staticmethod
    def preprocess(pil_img, is_mask):
        if is_mask:
            mask = pil_img.convert('1')
            np_mask = np.asarray(mask, dtype=np.float32)
            return np_mask
        else:
            img = pil_img.convert('RGB')
            np_img = np.asarray(img, dtype=np.float32)

            return np_img


    def __getitem__(self, idx):
        name = self.img_files[idx]
        img_file = name + self.img_format
        mask_file = name + self.mask_format

        img = Image.open(join(self.img_dir, img_file))
        mask = Image.open(join(self.mask_dir, mask_file))
        if img.size != mask.size:
            raise RuntimeError(f"Image and mask of {img_file} must be the same size")

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        if self.transform is not None:
            transformations = self.transform(image=img, mask=mask)
            img = transformations['image']
            mask = transformations['mask']

        return {
            'image': img, 
            'mask': mask
        }
    
class FloodDataset(BasicDataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__(img_dir, mask_dir, img_format=".jpg", mask_format=".png", transform=transform)