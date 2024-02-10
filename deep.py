from torch import Tensor, nn
from tqdm import tqdm
from data import *
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def make_mini_batch(data_size, batch_size, shuffle=True): 
    if shuffle: return torch.randperm(data_size).split(batch_size)
    else: return torch.arange(data_size).split(batch_size)

def evaluate(yhat, y, n_classe):
    if n_classe is None: # regression
        return F.mse_loss(yhat, y).item()
    elif n_classe == 2: 
        return (
            F.binary_cross_entropy(F.sigmoid(yhat), y.float()).item(),
            torch.where((yhat>=0.5).unsqueeze(-1) == y, 1., 0.).mean().item()
        )
    else:
        return (
            F.cross_entropy(yhat, y).item(),
            torch.where(yhat.argmax(1) == y, 1., 0.).mean().item()
        )

def get_task_loss(n_classe):
    """
    Donne la loss selon la tâche:
    - regression: Mean Square Error
    - classification binaire: Binary Cross Entropy 
    - classification multi-classe: Cross Entropy 
    """
    if n_classe is None or n_classe == 1: # regression
        return nn.MSELoss()
    elif n_classe == 2: return nn.BCEWithLogitsLoss()
    else: return nn.CrossEntropyLoss()
    
def get_features(dataset):
    """
    Donnne les dimensions des données:
    - numérique
    - binaire
    - catégoriel
    """
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
    Patience définit dans l'article, pour la minimisation.
    Pour faire du early stopping, on regarde loss la plus petite sur les donnnées de validation durant l'apprentissage.
    Si elle s'est amélioré, patience = 0, sinon patience augmente de 1 
    retourne un tuple (best_value, patience)
    """
    if current_value <= best_value - delta:
        return (current_value, 0)
    else:
        return (best_value, patience+1)



################ INSPIRED BY AUTHORS #############################
import torch.optim as optim
from typing import Any, Callable, Optional, Union, cast
from torch.nn.parameter import Parameter

#=============== Optimization ===============
def make_parameter_groups(
    model: nn.Module,
    zero_weight_decay_condition,
    custom_groups: dict[tuple[str], dict],  # [(fullnames, options), ...]
) -> list[dict[str, Any]]:
    custom_fullnames = set()
    custom_fullnames.update(*custom_groups)
    assert sum(map(len, custom_groups)) == len(
        custom_fullnames
    ), 'Custom parameter groups must not intersect'

    parameters_info = {}  # fullname -> (parameter, needs_wd)
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters():
            fullname = f'{module_name}.{name}' if module_name else name
            parameters_info.setdefault(fullname, (parameter, []))[1].append(
                not zero_weight_decay_condition(module_name, module, name, parameter)
            )
    parameters_info = {k: (v[0], all(v[1])) for k, v in parameters_info.items()}

    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    custom_params = {k: {'params': []} | v for k, v in custom_groups.items()}

    for fullname, (parameter, needs_wd) in parameters_info.items():
        for fullnames, group in custom_params.items():
            if fullname in fullnames:
                custom_fullnames.remove(fullname)
                group['params'].append(parameter)
                break
        else:
            (params_with_wd if needs_wd else params_with_wd)['params'].append(parameter)
    assert (
        not custom_fullnames
    ), f'Some of the custom parameters were not found in the model: {custom_fullnames}'
    return [params_with_wd, params_without_wd] + list(custom_params.values())


def zero_wd_condition(
    module_name: str,
    module: nn.Module,
    parameter_name: str,
    parameter: nn.parameter.Parameter,
):
    return (
        'Y' in module_name
        or 'Y' in parameter_name
        or default_zero_weight_decay_condition(
            module_name, module, parameter_name, parameter
        )
    )

def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
        ),
    )

def make_optimizer(
    module: nn.Module,
    type: str,
    *,
    zero_weight_decay_condition=zero_wd_condition,
    custom_parameter_groups: Optional[dict[tuple[str], dict]] = None,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if custom_parameter_groups is None:
        custom_parameter_groups = {}
    Optimizer = getattr(optim, type)
    parameter_groups = make_parameter_groups(
        module, zero_weight_decay_condition, custom_parameter_groups
    )
    return Optimizer(parameter_groups, **optimizer_kwargs)
###################################
