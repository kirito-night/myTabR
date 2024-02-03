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
    data_folder = '/Vrac/weather-big'
    batch_size = 1024
    is_regression = True

    # 0 à 7 (inclus)
    train_perc_i = 0 
    percs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]

    # paramètres d'apprentissage
    learning_rate = 0.0003121273641315169
    weight_decay = 0.0000012260352006404615

    # paramètres du modèle
    dataset, Y = load_data(
        path= '/Vrac/weather-big',
        mode = 'quantile',
        is_regression=is_regression
    )
    to_torch(dataset, Y, device=device)

    if is_regression: n_classe = None
    else: n_classe = len(torch.unique(Y['train']))
    loss_fn = get_task_loss(n_classe)
    n_num_features, n_bin_features, cat_features = get_features(dataset)

    if n_classe is None:
        for k,v in Y.items():
            Y[k] = v.float().unsqueeze(-1)
    if n_classe == 2:
        for k,v in Y.items():
            Y[k] = v.unsqueeze(-1)



    X_train = dataset['train']
    Y_train = Y['train']

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
        context_dropout= 0.38920071545944357,
        dropout0= 0.38852797479169876,
        normalization= nn.LayerNorm,
        activation= nn.ReLU,
        segmentation_batch_size = 10240
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
    order = torch.randperm(train_size)

    def make_candidate(order, i):
        size = Y_train.size()[0]
        cut = int(train_size*percs[i])
        candidate_index = order[:cut]
        return (
            {
                key: values[candidate_index]
                for key, values in X_train.items()
            },
            Y_train[candidate_index]
        )

    def get_Xy_train(data, labels, idx):
        return (
            {
                key: values[idx] 
                for key, values in data.items()
            },
            labels[idx]
        )

    candidat_train, labels_train = make_candidate(order, train_perc_i)
    candidat_size = labels_train.shape[0]

    best_score = np.inf
    patience = 0
    max_patience = 16

    # Apprentissage du Modèle
    epoch = 0
    while patience < max_patience:
        losses = []
        break
        print(f"======= {epoch = } =======")
        print(f'{patience = }/{max_patience}')
        model.train()
        for idx in tqdm(make_mini_batch(candidat_size, batch_size, shuffle=True)):
            optim.zero_grad()
            x, y = get_Xy_train(candidat_train, labels_train, idx)
            yhat = model(x, candidat_train, labels_train, training=True)
            l = loss_fn(yhat, y)
            l.backward()
            losses.append(l.item())
            optim.step()
        print('train | loss: ', np.mean(losses).round(4))
        log = []
        epoch += 1
        model.eval()
        break
        with torch.no_grad():
            for idx in make_mini_batch(val_size, batch_size, shuffle=False):
                x, y = get_Xy('val', idx)
                yhat = model(x, candidat_train, labels_train, training=False)
                if n_classe == 2: y = y.float()
                log.append(loss_fn(yhat, y).item())
            best_score, patience = get_patience(best_score,np.mean(log), patience)

    # exp
    resultat = []
    ############ A CHANGER ###################
    for i in range(5, len(percs)):
        torch.cuda.empty_cache()
        candidat_train, labels_train = make_candidate(order, i)
        model.eval()
        log = []
        with torch.no_grad():
            print("===================")
            print(model.segmentation_batch_size, labels_train.shape[0])
            for idx in tqdm(make_mini_batch(test_size, batch_size, shuffle=False)):
                x, y = get_Xy('test', idx)
                yhat = model(x, candidat_train, labels_train, training=False)
                log.append(loss_fn(yhat, y).item())
            print(f'{percs[i]} | {labels_train.shape} | {np.mean(log)}')
            resultat.append(np.mean(log))
    resultat = np.array(resultat)
    np.savetxt(f'log/{train_perc_i}.log', resultat)

if __name__ == '__main__':
    main()