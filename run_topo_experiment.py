import torch
import torch.nn as nn
from torchvision import transforms
from datasets.busi import BUSI
from datasets.betti import BettiDataset
from models.topovit import TopoViT
import torch.optim as optim
import numpy as np
import random
from topo_experiment.topo_strat_kfold import skfold
from utils.class_weights import class_weight

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    image_dataset = BUSI('./data/busi', transform=None)  # Base dataset without transforms
    b0_dataset = BettiDataset('./data/busi/busi_betti0.xlsx')
    b1_dataset = BettiDataset('./data/busi/busi_betti1.xlsx')

    # Get class weights
    class_weights = class_weight(image_dataset)
    class_weights = class_weights.to(device)

    # Create model
    model = TopoViT(model_name='vit_base_patch16_224', num_classes=3)
    model = model.to(device)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5])
    ])

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 100
    k = 5

    print('Starting K-Fold Cross Validation for TopoViT')
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