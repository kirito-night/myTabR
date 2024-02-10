import torch.nn.functional as F
# from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm
from data import *
from model import Model
import numpy as np
import torch
from deep import *


# paramètres du modèle
dataset, Y = load_data(
    path= '/Vrac/weather-big',
    is_regression=is_regression
)
to_torch(dataset, Y)

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


def main(train_perc_i, number=0):
    data_folder = '/Vrac/weather-big'
    batch_size = 1024
    is_regression = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#####################################")
    print(f"########## DEVICE {device} #############")
    print("#####################################")


    # 0 à 6 (inclus)
    percs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.42]

    model = Model(
        n_num_features=n_num_features,
        n_bin_features=n_bin_features,
        n_cat_features=cat_features,
        n_classes=n_classe,
    ).to(device)

    print("Number of cuda device", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optim_params = {
        "type": "AdamW",
        "lr": 0.0003121273641315169,
        "weight_decay": 0.0000012260352006404615,
    }

    optim = make_optimizer(model, **optim_params)

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

    def make_candidate(order, i, device=None):
        size = Y_train.size()[0]
        cut = int(train_size*percs[i])
        candidate_index = order[:cut]
        return (
            {
                key: values[candidate_index].to(device)
                for key, values in X_train.items()
            },
            Y_train[candidate_index].to(device)
        )

    def get_Xy_train(data, labels, idx):
        return (
            {
                key: values[idx] 
                for key, values in data.items()
            },
            labels[idx]
        )

    candidat_train, labels_train = make_candidate(order, train_perc_i, device)
    candidat_size = labels_train.shape[0]

    best_score = np.inf
    patience = 0
    max_patience = 16

    # Apprentissage du Modèle
    epoch = 0
    print('######### Training ################')
    dataset['val'], Y['val'] = to(dataset['val'], Y['val'], device)
    while patience < max_patience:
        losses = []
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
        model.reset_memory()
        with torch.no_grad():
            for idx in make_mini_batch(val_size, batch_size, shuffle=False):
                x, y = get_Xy('val', idx)
                yhat = model(x, candidat_train, labels_train, training=False, memory=True)
                if n_classe == 2: y = y.float()
                log.append(loss_fn(yhat, y).item())
            rmse = np.sqrt(np.mean(log))
            best_score, patience = get_patience(best_score, rmse, patience)

    del dataset['val']
    del Y['val']
    torch.cuda.empty_cache()
    
    # exp
    resultat = []



    dataset['test'], Y['test'] = to(dataset['test'], Y['test'], device)
    print('\n######### Evaluate ################')
    ############ A CHANGER ###################
    for i in range(train_perc_i, len(percs)):
        torch.cuda.empty_cache()
        candidat_train, labels_train = make_candidate(order, i, device)
        model.eval()
        model.reset_memory()
        log = []
        with torch.no_grad():
            print("===================")
            print('size: ',labels_train.shape[0])
            for idx in tqdm(make_mini_batch(test_size, batch_size, shuffle=False)):
                x, y = get_Xy('test', idx)
                yhat = model(x, candidat_train, labels_train, training=False, memory = True)
                log.append(loss_fn(yhat, y).item())
            
            rmse = np.sqrt(np.mean(log))
            print(f'{percs[i]} | {labels_train.shape} | {rmse}')
            resultat.append(rmse)
        del candidat_train
        del labels_train
        np.savetxt(f'log/{train_perc_i}_{number}.log', np.array(resultat))



if __name__ == '__main__':
    for perc_i in range(5):
        main(perc_i, log_num)