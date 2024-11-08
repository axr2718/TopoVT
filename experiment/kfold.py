import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
import copy
import numpy as np
from .train import train
from .test import test

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

def _kfolds(dataset: Dataset, len_dataset: int, k: int) -> list[Dataset]:
    """
    Returns a list of k-flolds.

    Args:
        dataset (Dataset): Dataset to be split.
        len_dataset (int): Length of the dataset.

    Returns:
        list[Dataset]: A list of all the k-folds in the form of PyTorch Dataset.
    """

    length = len_dataset
    indices = torch.randperm(length).tolist()

    fold_sizes = [length // k] * k
    for i in range(length % k):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(Subset(dataset, indices[start:stop]))
        current = stop

    return folds

def kfold(model: nn.Module,
          dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          k: int):
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

    len_dataset = len(dataset)

    kfolds = _kfolds(dataset=dataset, len_dataset=len_dataset, k=k)

    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    for fold in range(len(kfolds)):

        print(f"Fold = {fold + 1}")
        val_fold = kfolds[fold]

        train_folds = kfolds[:fold] + kfolds[fold + 1:]
        train_fold = ConcatDataset(train_folds)

        trained_model, _ = train(model=model, 
                                 train_dataset=train_fold, 
                                 criterion=criterion, 
                                 optimizer=optimizer, 
                                 epochs=epochs, 
                                 device=device)
        
        metrics = test(model=trained_model, 
                       test_dataset=val_fold, 
                       device=device)
        
        accuracy_list.append(metrics['accuracy'])
        precision_list.append(metrics['precision'])
        recall_list.append(metrics['recall'])
        f1_list.append(metrics['f1'])
        roc_auc_list.append(metrics['roc_auc'])

        print(f"Fold {fold + 1} Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

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