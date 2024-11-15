import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

def train(model, train_dataset, criterion, optimizer, epochs, device):
    model.train()
    trainloader = DataLoader(train_dataset, 
                           batch_size=len(train_dataset), 
                           shuffle=True, 
                           num_workers=6)

    for epoch in range(epochs):
        total_loss = 0.0
        for vectors, labels in trainloader:
            vectors = vectors.to(device)    
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(vectors)       
            loss = criterion(outputs, labels)

            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * vectors.size(0)
        
        epoch_loss = total_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.15f}')
    
    return model, epoch_loss