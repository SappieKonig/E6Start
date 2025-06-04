import torch
from torchvision.datasets import MNIST
import numpy as np


# Wrapper dataset to apply Albumentations
class AlbumentationsMNIST(torch.utils.data.Dataset):
    def __init__(self, train: bool, transform=None):
        self.mnist_dataset = MNIST(root="./data", train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        img_np = np.array(img)  # (H, W), grayscale

        # Expand dims to (H, W, 1) since Albumentations expects 3D input
        img_np = np.expand_dims(img_np, axis=-1)

        if self.transform:
            img_np = self.transform(image=img_np)["image"]

        return img_np, label
