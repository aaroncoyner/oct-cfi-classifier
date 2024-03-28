import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils import metrics
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score

from train_unet import CustomDataset, get_preprocessing
import config as C


os.makedirs(C.OUTPUT_DIR, exist_ok=True)
x_test_dir = os.path.join(C.DATA_DIR, 'test_A')
y_test_dir = os.path.join(C.DATA_DIR, 'test_B')


model = torch.load(f'./output/{C.MODEL}_{C.ENCODER}.pth', map_location=C.DEVICE)
model.eval()

preprocessing_fn = smp.encoders.get_preprocessing_fn(C.ENCODER, C.ENCODER_WEIGHTS)
test_dataset = CustomDataset(
    x_test_dir,
    y_test_dir,  # even though we don't use ground truth here, the class expects this argument
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=C.CLASSES
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

f1_scores = []
for i, (image, gt_mask) in enumerate(test_loader):
    image = image.to(C.DEVICE)
    with torch.no_grad():
        output = model(image)
    pred_mask = (output.cpu().numpy() > 0.5).astype(np.uint8)
    pred_mask = pred_mask[0, 0, :, :]

    gt_mask = gt_mask.cpu().numpy().astype(np.uint8)
    gt_mask = gt_mask[0, 0, :, :]  # Assuming your ground truth is also in [1, Height, Width] format

    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    score = f1_score(gt_flat, pred_flat, average='binary')
    f1_scores.append(score)

    filename = os.path.basename(test_dataset.rfi_fps[i])
    output_image = Image.fromarray(pred_mask * 255)
    output_image.save(os.path.join(C.OUTPUT_DIR, filename))

print(f"Average F1 Score: {np.mean(f1_scores)}")
