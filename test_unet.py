#!/usr/bin/env python
# coding: utf-8

import os
import shutil

import albumentations as A
import cv2
import numpy as np
import random
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as U
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Dataset(BaseDataset):
    CLASSES = ['choroid', 'vessel']
    
    def __init__(
            self, 
            rfi_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = [f_name for f_name in os.listdir(rfi_dir) if not f_name.startswith('.')]
        self.rfi_fps = [os.path.join(rfi_dir, image_id) for image_id in self.ids]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        image_id = self.ids[i]
        rfi = cv2.imread(self.rfi_fps[i])
        rfi = cv2.cvtColor(rfi, cv2.COLOR_BGR2RGB)
        
        if self.augmentation:
            sample = self.augmentation(image=rfi)
            rfi = sample['image']
        
        if self.preprocessing:
            sample = self.preprocessing(image=rfi)
            rfi = sample['image']
        return image_id, rfi
        
    def __len__(self):
        return len(self.ids)


def get_validation_augmentation():
    test_transform = [#A.Resize(480, 640, cv2.INTER_NEAREST),
                      #A.CLAHE(always_apply=True),
                     ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn),
                  A.Lambda(image=to_tensor)]
    return A.Compose(_transform)


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


DATA_DIR = './data/segmentation/test_A'
OUT_DIR = './results/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
NUM_WORKERS = 16 if DEVICE == torch.device('cuda') else 0
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['vessel']
BATCH_SIZE = 16

make_dirs(OUT_DIR)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model = torch.load(f'./models/unet_{ENCODER}.pth', map_location=DEVICE).eval()

dataset = Dataset(DATA_DIR,
                      augmentation=get_validation_augmentation(),
                      preprocessing=get_preprocessing(preprocessing_fn),
                      classes=CLASSES)

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS)

for image_ids, batch in tqdm(loader):
    batch = batch.to(DEVICE)
    with torch.no_grad():
        output = model(batch)

    output = output.squeeze().cpu().numpy()
    output = np.where(output > 0.5, 255, 0).astype(int)

    for segmentation, image_id in zip(output, image_ids):
        output_path = f'{OUT_DIR}{image_id.split(".")[0]}.png'
        cv2.imwrite(output_path, segmentation)
