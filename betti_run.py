import torch
import torch.nn as nn
from datasets.betti import BettiDataset
from models.betti_encoder import BettiClassifier
import torch.optim as optim
import numpy as np
import random
from swin_experiment.strat_kfold import skfold
from utils.class_weights import class_weight


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b0_dataset = BettiDataset('./data/busbra/busbra_betti0.xlsx')
    b1_dataset = BettiDataset('./data/busbra/busbra_betti1.xlsx')
    class_weights = class_weight(b0_dataset)
    class_weights = class_weights.to(device)

    num_classes = len(b0_dataset.classes)

    epochs = 100
    k = 5

    print('\nTraining Betti-0 Classifier')
    
    b0_model = BettiClassifier(num_classes=num_classes)
    b0_model = b0_model.to(device)
    
    b0_criterion = nn.CrossEntropyLoss(weight=class_weights)
    b0_optimizer = optim.Adam(b0_model.parameters(), lr=1e-3)
    
    skfold(model=b0_model, 
           dataset=b0_dataset, 
           criterion=b0_criterion, 
           optimizer=b0_optimizer, 
           epochs=epochs, 
           device=device, 
           k=k,
           seed=seed)

    print('\nTraining Betti-1 Classifier')
    
    b1_model = BettiClassifier(num_classes=num_classes)
    b1_model = b1_model.to(device)
    
    b1_criterion = nn.CrossEntropyLoss(weight=class_weights)
    b1_optimizer = optim.Adam(b1_model.parameters(), lr=1e-4)
    
    skfold(model=b1_model, 
           dataset=b1_dataset, 
           criterion=b1_criterion, 
           optimizer=b1_optimizer, 
           epochs=epochs, 
           device=device, 
           k=k,
           seed=seed)