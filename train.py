import csv
import os

import cv2
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.transforms import v2
from tqdm import tqdm



TRAIN_MODALITY = 'rfi'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 16
IMAGE_SIZE = (512, 512)
NUM_EPOCHS = 15
NUM_CLASSES = 3
LR = 1e-5


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ImageClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def get_class_weights(dataset):
    class_counts = np.bincount(dataset.targets)
    class_weights = 1. / class_counts
    weights = class_weights[dataset.targets]
    return weights


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(4,4)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        img = np.array(img)
        img = self.clahe.apply(img)
        img = Image.fromarray(img)
        return img


train_transform = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(15),
    v2.RandomPerspective(),
    # v2.RandomZoomOut(side_range=(0.5,2.0)),
    v2.Grayscale(num_output_channels=1),
    CLAHE(),
    v2.Grayscale(num_output_channels=3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    v2.Normalize(mean=[0.465, 0.465, 0.465], std=[0.221, 0.221, 0.221])
])

test_transform = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.Grayscale(num_output_channels=1),
    CLAHE(),
    v2.Grayscale(num_output_channels=3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.Normalize(mean=[0.465, 0.465, 0.465], std=[0.221, 0.221, 0.221])
])

# train_dataset = datasets.ImageFolder(f'data/binary/train_{TRAIN_MODALITY}', transform=train_transform)
# val_dataset = datasets.ImageFolder('data/binary/val', transform=test_transform)
# test_dataset = datasets.ImageFolder('data/binary/test', transform=test_transform)

train_dataset = datasets.ImageFolder(f'data/train_{TRAIN_MODALITY}', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=test_transform)
test_dataset = datasets.ImageFolder('data/test', transform=test_transform)

train_weights = get_class_weights(train_dataset)
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

val_weights = get_class_weights(val_dataset)
val_sampler = WeightedRandomSampler(val_weights, len(val_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = ImageClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Training]')
    for i, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / (i + 1)

        train_progress_bar.set_description(f'Epoch {epoch+1}/{NUM_EPOCHS} [Train] Loss: {loss.item():.4f}, Avg: {avg_train_loss:.4f}')

    model.eval()
    val_loss_sum = 0.0
    val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Validation]')
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_progress_bar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item()
            avg_val_loss = val_loss_sum / (i + 1)
            val_progress_bar.set_description(f'Epoch {epoch+1}/{NUM_EPOCHS} [Val] Loss: {loss.item():.4f}, Avg: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        if not os.path.isdir('out'):
            os.makedirs('out')
        torch.save(model.state_dict(), f'./out/{TRAIN_MODALITY}-model_best.pth')



model.load_state_dict(torch.load(f'./out/{TRAIN_MODALITY}-model_best.pth', map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []
all_probs = []
all_basenames = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

        # Extract and store basenames
        for img_path in test_loader.dataset.samples:
            basename = os.path.basename(img_path[0])
            all_basenames.append(basename)

# Writing to CSV
with open(f'./out/{TRAIN_MODALITY}-test_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'p_normal', 'p_pre_plus', 'p_plus', 'predicted_label', 'actual_label'])
    # writer.writerow(['image_name', 'p_0', 'p_1', 'predicted_label', 'actual_label'])
    for basename, prob, pred, label in zip(all_basenames, all_probs, all_preds, all_labels):
        writer.writerow([basename, prob[0], prob[1], prob[2], pred, label])


cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)

cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=False, plot_lib='matplotlib')
plt.savefig(f'./out/{TRAIN_MODALITY}-conf_mat-raw.png',dpi=600,bbox_inches ='tight')

cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib='matplotlib')
plt.savefig(f'./out/{TRAIN_MODALITY}-conf_mat-normalized.png',dpi=600,bbox_inches ='tight')
