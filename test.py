import csv
import os

from matplotlib import pyplot as plt
import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm



device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)



test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder('data/test', transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


model = ImageClassifier().to(device)
model.load_state_dict(torch.load('./out/model_best.pth', map_location=device))
model.eval()


all_preds = []
all_labels = []
all_probs = []
all_basenames = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
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
with open('./out/test_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'p_normal', 'p_pre_plus', 'p_plus', 'predicted_label', 'actual_label'])
    for basename, prob, pred, label in zip(all_basenames, all_probs, all_preds, all_labels):
        writer.writerow([basename, prob[0], prob[1], prob[2], pred, label])


cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)

cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=False, plot_lib='matplotlib')
plt.savefig('./out/conf_mat-raw.png',dpi=600,bbox_inches ='tight')

cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib='matplotlib')
plt.savefig('./out/conf_mat-normalized.png',dpi=600,bbox_inches ='tight')
