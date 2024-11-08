import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import v2
from torch.optim.lr_scheduler import OneCycleLR

def train(model: nn.Module,
          train_dataset: torch.utils.data.Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          scheduler=None) -> tuple[nn.Module, float]:
    """
    Trains a model on the dataset given the loss function, optimizer, and number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        train_dataset (Dataset): The dataset being used for training.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs to be trained on.
        device (device): Device that will train.

    Returns:
        tuple[nn.Module, float]: The trained model and the (optional) final loss.
    """
    
    model.train()

    trainloader = DataLoader(train_dataset, 
                             batch_size=128, 
                             shuffle=True, 
                             num_workers=2)
    
    steps_per_epoch = len(trainloader)
    total_steps = epochs * steps_per_epoch

    scheduler = OneCycleLR(optimizer, max_lr=1e-6, total_steps=total_steps)
    
    mixup = v2.MixUp(num_classes=len(train_dataset.dataset.classes))


    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in trainloader:
            images, labels = mixup(images, labels)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        epoch_loss = total_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.15f}')

        if scheduler is not None:
            scheduler.step()
    
    return model, epoch_loss