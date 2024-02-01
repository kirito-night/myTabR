import torch.nn.functional as F
# from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm
from data import *
from model import Model
import numpy as np
import torch
from deep import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_folder = '/users/Etu5/28601285/Documents/AMAL/myTabR/data'
    data_name = 'churn'
    is_regression = False

    # paramètres d'apprentissage
    n_epochs = 20
    BATCHSIZE = 100
    learning_rate = 0.0003121273641315169
    weight_decay = 0.0000012260352006404615

    # paramètres du modèle
    context_size = 10


    dataset, Y = load_data(
        path= f"{data_folder}/{data_name}",
        mode = 'quantile',
        is_regression=is_regression
    )
    to_torch(dataset, Y, device=device)

    X_train = dataset['train']
    Y_train = Y['train']

    if is_regression: n_classe = None
    else: n_classe = len(torch.unique(Y_train))
    loss_fn = get_task_loss(n_classe)
    n_num_features, n_bin_features, cat_features = get_features(dataset)

    model = Model(
        n_num_features=n_num_features,
        n_bin_features=n_bin_features,
        cat_cardinalities=cat_features,
        n_classes=n_classe,
        #
        d_main =  265,
        d_multiplier = 2,
        encoder_n_blocks = 0,
        predictor_n_blocks = 1,
        context_dropout= 0.389,
        dropout0= 0.389,
        normalization= nn.LayerNorm,
        activation= nn.ReLU,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
    )

    def get_Xy(part, idx):
        return (
            {
                key: values[idx] 
                for key, values in dataset[part].items()
            },
            Y[part][idx]
        )
    
    train_size = Y_train.size()[0]
    test_size = Y['test'].size()[0]
    val_size = Y['val'].size()[0]
    
    best_score = np.inf
    patience = 0
    max_patience = 16
    epoch = 0
    while patience < max_patience:
        losses = []
        print(f"======= {epoch = } =======")
        print(f'{patience = }/{max_patience}')
        for idx in tqdm(make_mini_batch(train_size, BATCHSIZE, shuffle=True)):
            optim.zero_grad()
            x, y = get_Xy('train', idx)
            yhat = model(x, X_train, Y_train, training=True)
            l = loss_fn(yhat, y.float())
            l.backward()
            losses.append(l.item())
            optim.step()
        print('train | loss: ', np.mean(losses).round(4))
        with torch.no_grad():
            # Données test
            log = []
            for idx in make_mini_batch(test_size, BATCHSIZE, shuffle=True):
                x, y = get_Xy('test', idx)
                yhat = model(x, X_train, Y_train, training=False)
                log.append(evaluate(yhat, y, n_classe))
            if is_regression: 
                print('test | loss: ', np.mean(log).round(4))
            else: 
                l, acc = np.mean(log,0).round(4)
                print(f'test | acc: {acc} | loss: {l}')
            # Données validation
            log = []
            for idx in tqdm(make_mini_batch(val_size, BATCHSIZE, shuffle=False)):
                x, y = get_Xy('val', idx)
                yhat = model(x, X_train, Y_train, training=False)
                log.append(loss_fn(yhat, y.float()).item())
            best_score, patience = get_patience(best_score,np.mean(log), patience)
        epoch += 1
        
                






if __name__ == '__main__':
    main()