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
    
    model.train()

    trainloader = DataLoader(train_dataset, 
                             batch_size=32, 
                             shuffle=True, 
                             num_workers=6)


    for epoch in range(epochs):
        total_loss = 0.0
        for images, b0, b1, labels in trainloader:
            images = images.to(device)
            b0 = b0.to(device)
            b1 = b1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, b0, b1)
            loss = criterion(outputs, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        epoch_loss = total_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.15f}')
    
    return model, epoch_loss