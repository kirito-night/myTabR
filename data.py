import numpy as np
from os.path import isfile, join
import sklearn.preprocessing
import torch


#========= Preprocessing dataset ==============
def process_num(dataset):
    """
    Pré-traitement pour données numériques
    On fait une normalisation quantile en utilisant Scikit-learn
    """
    X_train = dataset['train']['num']
    normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles = max(min(X_train.shape[0] // 30, 1000), 10),
            subsample=1_000_000_000,  # i.e. no subsampling
        )
    noise = 1e-3
    # l'auteur ajoute du bruit dans le cas où il y a  peu de valeurs uniques
    stds = np.std(X_train, axis=0, keepdims=True)
    noise_std = noise / np.maximum(stds, noise) 
    X_train = X_train + noise_std * np.random.default_rng().standard_normal(X_train.shape)
    normalizer.fit(X_train)
    for key in ['train', 'test', 'val']:
        dataset[key]['num'] = normalizer.transform(dataset[key]['num'])

def process_cat(dataset):
    """
    Pré-traitement pour données catégorielles
    On fait un one-hot encoding sur les données
    """
    transformer = sklearn.preprocessing.OneHotEncoder(
        handle_unknown='ignore', 
        sparse=False, 
        dtype=np.float32
    )
    transformer.fit(dataset['train']['cat'])
    for key in ['train', 'test', 'val']:
        dataset[key]['cat'] = transformer.transform(dataset[key]['cat'])

def process_y(Y):
    """
    Pré-traitement sur les labels pour la regression
    On fait une normalisation centrée réduite
    """
    assert len(Y['train'].shape) == 2
    normalizer = sklearn.preprocessing.StandardScaler()
    normalizer.fit(Y['train'])
    for key in ['train', 'test', 'val']:
        Y[key] = normalizer.transform(Y[key])
 

def cat_features(dataset):
    return [len(np.unique(x)) for x in dataset['train']['cat'].T]

def load_data(path, mode = 'quantile', is_regression = False):
    """
    Charge les données d'entrainement, de validation, de test
    path: chemin du dossier avec les données
    mode: 
    """
    if path[-1] == '/': path = path[:-1]
    X = {part:
         {
             key: np.load(f_path) 
             for key in ['num', 'bin', 'cat']
             if isfile(f_path := f'{path}/X_{key}_{part}.npy')
         }
        for part in ['train', 'test', 'val']
    }
    Y = {part:np.load(f'{path}/Y_{part}.npy')[..., np.newaxis] for part in ['train', 'test', 'val']}
    if 'num' in  X['train']: process_num(X)
    if 'cat' in  X['train']: process_cat(X)
    if is_regression: process_y(Y)
    return X, Y

def to_torch(dataset, labels):
    for data in dataset.values():
        for key, x in data.items():
            data[key] = torch.from_numpy(x)
    for part, y in labels.items():
        labels[part] = torch.from_numpy(y).squeeze()

def to(data, labels, device=None):
    return (
        {k: v.to(device) for k,v in data.items()},
        labels.to(device)
    )
