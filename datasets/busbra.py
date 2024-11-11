import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BusBra(Dataset):
    """
    Dataset class for the BusBra dataset.

    Args:
        dir (str): Name of the directory containing all the images.
        transform (callable, optional): Transforms to be applied to data.
    """
    def __init__(self, dir: str, transform: bool = None) -> Dataset:
        self.transform = transform
        self.dir = dir
        self.classes = ['benign', 'malignant']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.img_paths = []
        self.labels = []

        for category in self.classes:
            category_dir = os.path.join(dir, category)

            for image_name in os.listdir(category_dir):
                image = os.path.join(category_dir, image_name)
                self.img_paths.append(image)
                self.labels.append(self.class_to_idx[category])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label