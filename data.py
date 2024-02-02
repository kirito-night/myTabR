import numpy as np
from os.path import isfile, join
import sklearn.preprocessing
import torch


#========= Preprocessing dataset ==============
def process_num(dataset, mode='quantile'):
    """
    preprocessing for numeric featuress
    """
    X_train = dataset['train']['num']
    if mode == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=1_000_000_000,  # i.e. no subsampling
            )
        noise = 1e-3
        if noise > 0:
            # Noise is added to get a bit nicer transformation
            # for features with few unique values.
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train = X_train + noise_std * np.random.default_rng().standard_normal(X_train.shape)
    elif mode == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    normalizer.fit(X_train)
    for key in ['train', 'test', 'val']:
        dataset[key]['num'] = normalizer.transform(dataset[key]['num'])

def process_cat(dataset):
    """
    preprocessing for categorial featuress
    """
    transformer = sklearn.preprocessing.OneHotEncoder(
        handle_unknown='ignore', sparse=False, dtype=np.float32  # type: ignore[code]
    )
    transformer.fit(dataset['train']['cat'])
    for key in ['train', 'test', 'val']:
        dataset[key]['cat'] = transformer.transform(dataset[key]['cat'])

def process_y(Y):
    """
    preprocessing for target values
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
    Load Train, Test, Val Dataset from folder
    return X, Y
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
    if 'num' in  X['train']: process_num(X, mode)
    if 'cat' in  X['train']: process_cat(X)
    if is_regression: process_y(Y)
    return X, Y

def to_torch(dataset, labels, device=None):
    for data in dataset.values():
        for key, x in data.items():
            data[key] = torch.from_numpy(x).to(device)
    for part, y in labels.items():
        labels[part] = torch.from_numpy(y).squeeze().to(device)