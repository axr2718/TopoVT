from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch
import numpy as np
import copy
from datasets.busi import BUSI
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .test import test
from .train import train

def compute_mean_std_err(metric_list: list) -> tuple[float, float]:
    """
    Computes the mean and standard error of a metric list.

    Args:
        metric_list (list): A list containing the metrics of each fold.

    Returns:
        tuple[float, float]: The mean and standard error of the aggregated list.

    """
    mean = np.mean(metric_list)
    std_err = np.std(metric_list, ddof=1) / np.sqrt(len(metric_list))

    return mean, std_err

def skfold(model: nn.Module,
          dataset: torch.utils.data.Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          k: int,
          seed: int,
          train_transform=None,
          val_transform=None,
          scheduler=False):
    """
    Trains and evaluates a model using k-fold cross-validation.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): The dataset being used for training.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs to be trained on.
        device (device): Device that will train.
    """
    labels = np.array(dataset.labels)
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    for fold, (train_idx, val_idx) in enumerate(skfold.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold + 1}")

        # Create separate dataset instances for training and validation
        train_full_dataset = BUSI('./data', transform=train_transform)
        val_full_dataset = BUSI('./data', transform=val_transform)

        # Create subsets using the indices
        train_dataset = Subset(dataset=train_full_dataset, indices=train_idx)
        val_dataset = Subset(dataset=val_full_dataset, indices=val_idx)

        if scheduler:
            scheduler = scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        trained_model, _ = train(model=model,
                                train_dataset=train_dataset,
                                criterion=criterion,
                                optimizer=optimizer,
                                epochs=epochs,
                                device=device)

        metrics = test(model=trained_model,
                        test_dataset=val_dataset,
                        device=device)

        accuracy_list.append(metrics['accuracy'])
        precision_list.append(metrics['precision'])
        recall_list.append(metrics['recall'])
        f1_list.append(metrics['f1'])
        roc_auc_list.append(metrics['roc_auc'])

        print(f"Fold {fold + 1} Metrics: Accuracy={metrics['accuracy']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)

    accuracy_mean, accuracy_std_err = compute_mean_std_err(accuracy_list)
    precision_mean, precision_std_err = compute_mean_std_err(precision_list)
    recall_mean, recall_std_err = compute_mean_std_err(recall_list)
    f1_mean, f1_std_err = compute_mean_std_err(f1_list)
    roc_auc_mean, roc_auc_std_err = compute_mean_std_err(roc_auc_list)


    print("\nFinal Metrics Across All Folds:")
    print(f"Accuracy: Mean={accuracy_mean:.4f}, StdErr={accuracy_std_err:.4f}")
    print(f"Precision: Mean={precision_mean:.4f}, StdErr={precision_std_err:.4f}")
    print(f"Recall: Mean={recall_mean:.4f}, StdErr={recall_std_err:.4f}")
    print(f"F1 Score: Mean={f1_mean:.4f}, StdErr={f1_std_err:.4f}")
    print(f"ROC-AUC: Mean={roc_auc_mean:.4f}, StdErr={roc_auc_std_err:.4f}")
