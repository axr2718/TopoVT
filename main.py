import torch
import torch.nn as nn
from torchvision import transforms
from datasets.busi import BUSI
from models.vit import ViT
import torch.optim as optim
from timm.data import Mixup
import numpy as np
import random
from experiment.kfold import kfold
from experiment.strat_kfold import skfold

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

    model = ViT(model_name='vit_base_patch16_224', num_classes=3, freeze=False)
    model = model.to(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    ])

    dataset = BUSI('./data', transform=val_transform)
    
    class_weights = BUSI.class_weights(dataset)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    epochs = 100
    k = 5

    mixup_fn = Mixup(mixup_alpha=0.2,
                    cutmix_alpha=0.0,
                    prob=1.0,
                    switch_prob=0.5,
                    mode='batch',
                    label_smoothing=0.0,
                    num_classes=3)

    print('Starting K-Fold Cross Validation')
    skfold(model=model, 
          dataset=dataset, 
          criterion=criterion, 
          optimizer=optimizer, 
          epochs=epochs, 
          device=device, 
          k=k,
          seed=seed,
          train_transform=train_transform,
          val_transform=val_transform,
          scheduler=True)