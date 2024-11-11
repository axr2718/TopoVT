from torch.utils.data import Dataset
import numpy as np
import torch

def class_weight(dataset: Dataset):
        labels = np.array(dataset.labels)
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)

        return torch.tensor(class_weights, dtype=torch.float)