import os
import torch

# Paths
DATA_DIR = './data/segmentation/'
OUTPUT_DIR = './output/images/'

# Model
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 1
CLASSES = ['vessel']
ACTIVATION = 'sigmoid'

# Training parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
NUM_WORKERS = 16 if torch.cuda.is_available() else 0
TRAIN_BATCH = 16
VAL_BATCH = 8
LR = 1e-3
EPOCHS = 100
PATIENCE = 10
CROP_SIZE = 448
