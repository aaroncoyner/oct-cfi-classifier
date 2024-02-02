import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset



class PairedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask


def visualize_transformations(dataset, indices, transforms):
    fig, axs = plt.subplots(2, len(indices), figsize=(5*len(indices), 10))
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        # Apply transformation only for visualization
        transformed_image = transforms(image)

        axs[0, i].imshow(image)
        axs[0, i].set_title(f'Original Image {idx}')
        axs[0, i].axis('off')

        axs[1, i].imshow(transformed_image)
        axs[1, i].set_title(f'Transformed Image {idx}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Assuming train_dataset is your dataset object
SRC = '../data/segmentation-data/'
num_samples_to_visualize = 5
indices = list(range(num_samples_to_visualize))

train_transforms = v2.Compose([
    v2.RandomCrop((512,512)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    # v2.GaussianBlur((5,5),3),
    v2.ColorJitter(brightness=0.5, contrast=0.5),
])

train_dataset = PairedDataset(SRC + 'train_A', SRC + 'train_B')


visualize_transformations(train_dataset, indices, train_transforms)
