import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

# Define your dataset and data loader
dataset = ImageFolder(root='./data.nosync/val/', transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Calculate mean and std for normalization
mean = torch.zeros(1)
std = torch.zeros(1)
total_samples = len(dataset)

for data in data_loader:
    image = data[0]  # Assuming the first element of the data tuple is the image
    mean += image.mean()
    std += image.std()

mean /= total_samples
std /= total_samples

print(f"Mean: {mean.item()}")
print(f"Std: {std.item()}")
