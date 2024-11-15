import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

def train(model: nn.Module,
          train_dataset: torch.utils.data.Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> tuple[nn.Module, float]:
    """
    Trains a TopoSwin model.

    Args:
        model (nn.Module): The model to be trained.
        train_dataset (Dataset): The dataset being used for training (TopoTransformDataset).
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs to be trained on.
        device (device): Device that will train.

    Returns:
        tuple[nn.Module, float]: The trained model and the final loss.
    """
    
    model.train()

    trainloader = DataLoader(train_dataset, 
                           batch_size=64, 
                           shuffle=True, 
                           num_workers=6,
                           pin_memory=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for img, b0, b1, labels in trainloader:
            img = img.to(device)
            b0 = b0.to(device)
            b1 = b1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(img, b0, b1)
            loss = criterion(outputs, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * img.size(0)
        
        epoch_loss = total_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.15f}')
    
    return model, epoch_loss