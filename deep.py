from torch import Tensor, nn
from tqdm import tqdm
from data import *
from model import Model
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def make_mini_batch(data_size, batch_size, shuffle=True) -> torch.Tensor: 
    if shuffle: return torch.randperm(data_size).split(batch_size)
    else: return torch.arange(data_size).split(batch_size)

def evaluate(yhat, y, n_classe):
    if n_classe is None: # regression
        return F.mse_loss(yhat, y).item()
    elif n_classe == 2: 
        return (
            F.binary_cross_entropy(F.sigmoid(yhat), y.float()).item(),
            torch.where((yhat>=0.5) == y, 1., 0.).mean().item()
        )
    else:
        return (
            F.cross_entropy(yhat, y).item(),
            torch.where(yhat == y, 1., 0.).mean().item()
        )
    
def get_task_loss(n_classe):
    if n_classe is None: # regression
        return nn.MSELoss()
    elif n_classe == 2: 
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
    
def get_features(dataset):
    n_num_features = dataset['train'].get('num', 0)
    n_bin_features = dataset['train'].get('bin', 0)
    cat_features =  dataset['train'].get('cat', 0)
    if 'num' in dataset['train']: 
        n_num_features = n_num_features.shape[1]
    if 'bin' in dataset['train']: 
        n_bin_features = n_bin_features.shape[1]
    if 'cat' in dataset['train']: 
        cat_features = cat_features.shape[1]
    return n_num_features, n_bin_features, cat_features


def get_patience(best_value, current_value, patience, delta=0):
    """
    Patience défnint dans l'article, pour la minimisation.
    retourne un tuple (best_value, patience)
    """
    if current_value <= best_value - delta:
        return (current_value, 0)
    else:
        return (best_value, patience+1)

        
    