from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.subset)