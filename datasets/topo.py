from torch.utils.data import Dataset

class TopoDataset(Dataset):
    def __init__(self, img_dataset, b0_dataset, b1_dataset, transform=None):
        self.img_dataset = img_dataset
        self.b0_dataset = b0_dataset
        self.b1_dataset = b1_dataset
        self.transform = transform

    def __len___(self):
        return len(self.img_dataset)
    
    def __getitem__(self, idx):
        img, label = self.img_dataset[idx]
        b0, _ = self.b0_dataset
        b1, _ = self.b1_dataset

        if self.transform:
            img = self.transform(img)

        return img, b0, b1, label
    