from torch.utils.data import Dataset

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len (self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    @property
    def labels(self):
        return self.dataset.labels