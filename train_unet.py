#!/usr/bin/env python
# coding: utf-8

import os

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses, train

import config as C


class CustomDataset(Dataset):
    CLASSES = ['background', 'vessel']
    
    def __init__(self, rfi_dir, rvm_dir, classes=None, augmentation=None, preprocessing=None):

        self.ids = [f for f in os.listdir(rvm_dir) if not f.startswith('.')]

        self.rfi_fps = [os.path.join(rfi_dir, image_id) for image_id in self.ids]
        self.rvm_fps = [os.path.join(rvm_dir, image_id) for image_id in self.ids]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else []
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        rfi = np.array(Image.open(self.rfi_fps[i]).convert('RGB'), dtype=np.float32)
        rvm = np.array(Image.open(self.rvm_fps[i]).convert('L'), dtype=np.uint8)
        rvm = np.where(rvm > 0, 1, 0)
        rvm = np.stack([(rvm == v).astype('float') for v in self.class_values], axis=-1)
        
        if self.augmentation:
            sample = self.augmentation(image=rfi, mask=rvm)
            rfi, rvm = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=rfi, mask=rvm)
            rfi, rvm = sample['image'], sample['mask']
            rfi = rfi[0, :, :]
            rfi = np.expand_dims(rfi, axis=0)

        return rfi, rvm
    
    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.RandomCrop(height=C.CROP_SIZE, width=C.CROP_SIZE, always_apply=True),
                       # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                       # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=90, shift_limit=0., p=1, border_mode=0)
                      ]    
    return A.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn),
                  A.Lambda(image=to_tensor, mask=to_tensor)]
    return A.Compose(_transform)


def train_val():
    x_train_dir = os.path.join(C.DATA_DIR, 'train_A')
    y_train_dir = os.path.join(C.DATA_DIR, 'train_B')
    x_val_dir = os.path.join(C.DATA_DIR, 'val_A')
    y_val_dir = os.path.join(C.DATA_DIR, 'val_B')

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(C.ENCODER, C.ENCODER_WEIGHTS)

    train_dataset = CustomDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=C.CLASSES
    )

    val_dataset = CustomDataset(
        x_val_dir,
        y_val_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=C.CLASSES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=C.TRAIN_BATCH,
        shuffle=True,
        num_workers=C.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=C.VAL_BATCH,
        shuffle=False,
        num_workers=C.NUM_WORKERS
    )

    model = smp.Unet(
        encoder_name=C.ENCODER,
        encoder_weights=C.ENCODER_WEIGHTS,
        in_channels=C.IN_CHANNELS,
        classes=len(C.CLASSES),
        activation=C.ACTIVATION
    )

    model = model.to(C.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss = losses.DiceLoss()
    metric = [metrics.Fscore(threshold=0.5)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8)

    max_score = 0
    best_epoch = 0
    for epoch in range(1, C.EPOCHS + 1):
        print(f'\nEpoch {epoch} | learning rate: {scheduler.get_last_lr()[0]:.7f}')
        train_logs = train.TrainEpoch(
            model,
            loss=loss,
            metrics=metric,
            optimizer=optimizer,
            device=C.DEVICE,
            verbose=True
        ).run(train_loader)

        val_logs = train.ValidEpoch(
            model,
            loss=loss,
            metrics=metric,
            device=C.DEVICE,
            verbose=True
        ).run(val_loader)

        scheduler.step()

        if max_score < val_logs['fscore']:
            max_score = val_logs['fscore']
            best_epoch = epoch
            torch.save(model, f'./output/unet_{C.ENCODER}.pth')
            print('Model saved!')

    print(f'Best F score ({max_score}) at epoch {best_epoch}')


if __name__=='__main__':
    train_val()
