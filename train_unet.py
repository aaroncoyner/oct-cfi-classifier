#!/usr/bin/env python
# coding: utf-8

import os
import ssl

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as U

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context


# def seed_everything(seed):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
    
# seed_everything(1337)


class Dataset(BaseDataset):
    
    CLASSES = ['background', 'vessel']
    
    def __init__(
            self, 
            image_dir, 
            mask_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = [f for f in os.listdir(mask_dir) if not f.startswith('.')]
        self.image_fps = [os.path.join(image_dir, image_id) for image_id in self.ids]
        self.mask_fps = [os.path.join(mask_dir, image_id) for image_id in self.ids]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.image_fps[i])
        mask = cv2.imread(self.mask_fps[i], 0)

        mask = np.where(mask > 0, 1, 0)        
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = np.where(mask > 0, 1, 0)
        return image, mask
        
    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [A.RandomCrop(height=256, width=256, always_apply=True),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=90, shift_limit=0., p=1, border_mode=0)
                      ] 
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [#A.Resize(480, 640, cv2.INTER_NEAREST),
                      #A.CLAHE(always_apply=True)
                     ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """

    _transform = [A.Lambda(image=preprocessing_fn),
                  A.Lambda(image=to_tensor, mask=to_tensor)]
    return A.Compose(_transform)



DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
NUM_WORKERS = 16 if DEVICE == 'cuda' else 0
DATA_DIR = './data/segmentation/'
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['vessel']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 10


x_train = os.path.join(DATA_DIR, 'train_A')
y_train = os.path.join(DATA_DIR, 'train_B')

x_val = os.path.join(DATA_DIR, 'val_A')
y_val = os.path.join(DATA_DIR, 'val_B')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(x_train,
                        y_train,
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES)

valid_dataset = Dataset(x_val,
                        y_val,
                        augmentation=get_validation_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

valid_loader = DataLoader(valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS)


model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=len(CLASSES),
                 activation=ACTIVATION)

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR)])
loss = U.losses.DiceLoss()
metrics = [U.metrics.Fscore(threshold=0.5)]


train_epoch = U.train.TrainEpoch(model,
                                 loss=loss,
                                 metrics=metrics,
                                 optimizer=optimizer,
                                 device=DEVICE,
                                 verbose=True)

valid_epoch = U.train.ValidEpoch(model,
                                 loss=loss, 
                                 metrics=metrics, 
                                 device=DEVICE,
                                 verbose=True)

epochs_since_update = 0
max_score = 0

for i in range(1, EPOCHS + 1):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        epochs_since_update = 0
        torch.save(model, f'./models/unet_{ENCODER}.pth')
        print('Model saved!')
    else:
        epochs_since_update += 1
        
    if epochs_since_update == PATIENCE:
        optimizer.param_groups[0]['lr'] /= 10
        epochs_since_update = 0
        print(f'Decreasing decoder learning rate to {optimizer.param_groups[0]["lr"]}')

print(f'Best F score: {max_score}')
