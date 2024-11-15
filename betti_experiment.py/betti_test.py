import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (accuracy_score,
                     precision_score,
                     recall_score,
                     f1_score,
                     roc_auc_score)

def test(model, test_dataset, device):
    model.eval()
    testloader = DataLoader(dataset=test_dataset, 
                          batch_size=len(test_dataset), 
                          shuffle=False, 
                          num_workers=6)
    
    all_labels = []
    all_probabilities = []
    all_predictions = []
    
    with torch.no_grad():
        for vectors, labels in testloader:  
            vectors = vectors.to(device)    
            labels = labels.to(device)

            outputs = model(vectors)        
            probabilities = F.softmax(outputs, dim=1)

            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_predictions = np.array(all_predictions)

    num_classes = all_probabilities.shape[1]

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    if num_classes == 2:
        roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return metrics