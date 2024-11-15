import torch
import torch.nn as nn
from torchvision import transforms
from datasets import (busi, busbra, mendeley)
from datasets.betti import BettiDataset
from models.toposwin import TopoSwin
import torch.optim as optim
import numpy as np
import random
from topo_experiment.topo_strat_kfold import skfold
from utils.class_weights import class_weight


def set_seed(seed):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dataset = mendeley.Mendeley('./data/mendeley', transform=None)
    b0_dataset = BettiDataset('./data/mendeley/mendeley_betti0.xlsx')
    b1_dataset = BettiDataset('./data/mendeley/mendeley_betti1.xlsx')

    num_classes = len(image_dataset.classes)

    class_weights = class_weight(image_dataset)
    class_weights = class_weights.to(device)

    model = TopoSwin(num_classes=num_classes)
    model = model.to(device)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, 
                                                     contrast=0.2, 
                                                     saturation=0.2, 
                                                     hue=0.1)], p=0.5),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5])
    ])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9)
    epochs = 100
    k = 5

    print('Starting K-Fold Cross Validation for TopoSwin')
    skfold(
        model=model,
        image_dataset=image_dataset,
        b0_dataset=b0_dataset,
        b1_dataset=b1_dataset,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        k=k,
        seed=seed,
        train_transform=train_transform,
        val_transform=val_transform
    )