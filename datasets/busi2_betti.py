import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Betti2Dataset(Dataset):
    def __init__(self, filename, dataset_num=None):
        df = pd.read_excel(filename, skiprows=0)
        
        betti_vector = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        labels = np.array(labels)
        
        desired_labels = [1, 2]
        
        mask = np.isin(labels, desired_labels)
        
        self.betti_vector = betti_vector[mask]
        self.labels = labels[mask]
        
        self.labels = self.labels - 1
        
        self.classes = np.unique(self.labels)
        

    def __len__(self):
        return len(self.betti_vector)
    
    def __getitem__(self, idx):
        vector = self.betti_vector[idx]
        label = self.labels[idx]

        # Convert to torch tensors
        vector = torch.tensor(vector, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return vector, label