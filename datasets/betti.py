import torch
from torch.utils.data import Dataset
import pandas as pd

class BettiDataset(Dataset):
    def __init__(self, filename):

        df = pd.read_excel(filename, skiprows=0)

        betti_vector = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        self.betti_vector = betti_vector
        self.labels = labels

    def __len__(self):
        return len(self.betti_vector)
    
    def __getitem__(self, idx):
        vector = self.betti_vector[idx]
        label = self.labels[idx]

        vector = torch.tensor(vector, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return vector, label