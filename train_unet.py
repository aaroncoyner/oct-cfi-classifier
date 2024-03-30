#!/usr/bin/env python
# coding: utf-8

import os

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses, train

import config as C

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CustomDataset(Dataset):
    CLASSES = ['background', 'vessel']
    
    def __init__(self, image_dir, mask_dir, classes=None, augmentation=None, preprocessing=None):

        self.ids = [f for f in os.listdir(mask_dir) if not f.startswith('.')]

        self.image_fps = [os.path.join(image_dir, image_id) for image_id in self.ids]
        self.mask_fps = [os.path.join(mask_dir, image_id) for image_id in self.ids]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else []
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = np.array(Image.open(self.image_fps[i]).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(self.mask_fps[i]).convert('L'), dtype=np.uint8)
        mask = np.where(mask > 0, 1, 0)
        mask = np.stack([(mask == v).astype('float') for v in self.class_values], axis=-1)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            image = image[0, :, :]
            image = np.expand_dims(image, axis=0)

        return image, mask
    
    def __len__(self):
        return len(self.ids)


def get_model(model_name, encoder, weights, channels, classes, activation):
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=channels,
            classes=len(classes),
            activation=activation
        )
        return model
    
    elif model_name == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=channels,
            classes=len(classes),
            activation=activation
        )
        return model

    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=channels,
            classes=len(classes),
            activation=activation
        )
        return model

    elif model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=channels,
            classes=len(classes),
            activation=activation
        )
        return model

    else:
        print('ERROR: Model not implemented.')


def get_training_augmentation():
    train_transform = [A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.ShiftScaleRotate(scale_limit=0.2, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
                       # A.RandomCrop(height=C.CROP_SIZE, width=C.CROP_SIZE, always_apply=True),
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

    model = get_model(
        C.MODEL,
        encoder=C.ENCODER,
        weights=C.ENCODER_WEIGHTS,
        channels=C.IN_CHANNELS,
        classes=C.CLASSES,
        activation=C.ACTIVATION
    )

    model = model.to(C.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss = losses.DiceLoss()
    metric = [metrics.Fscore(threshold=0.5)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=C.T_0, eta_min=C.ETA_MIN)

    max_score = 0
    best_epoch = 0
    for epoch in range(1, C.EPOCHS + 1):
        print(f'\nEpoch {epoch} | learning rate: {scheduler.get_last_lr()[0]:.6f}')
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
            torch.save(model, f'./output/{C.MODEL}_{C.ENCODER}.pth')
            print('Model saved!')

    print(f'Best F score ({max_score}) at epoch {best_epoch}')


if __name__=='__main__':
    train_val()
