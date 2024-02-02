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
    data_folder = './data'
    
    data_name_list = [
        'black-friday',
        #'otto', 
        #'california', 
        #'adult', 
        'churn']
    tasks = [
        True, 
        #False, 
        #True, 
        #False, 
        False]
    batch_list = [512,512,256,256,128]


    # paramètres d'apprentissage
    learning_rate = 0.0003121273641315169
    weight_decay = 0.0000012260352006404615

    # paramètres du modèle

    for data_name, is_regression, BATCHSIZE in tqdm(zip(data_name_list, tasks, batch_list)):
        print('Dataset: ', data_name)
        print(f'{is_regression = }')
        print(f'{BATCHSIZE = }')
        torch.cuda.empty_cache()
        dataset, Y = load_data(
            path= f"{data_folder}/{data_name}",
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
            # print(f"======= {epoch = } =======")
            # print(f'{patience = }/{max_patience}')
            for idx in make_mini_batch(train_size, BATCHSIZE, shuffle=True):
                optim.zero_grad()
                x, y = get_Xy('train', idx)
                yhat = model(x, X_train, Y_train, training=True)
                #print(yhat.shape, y.shape)
                if n_classe == 2: y = y.float()
                l = loss_fn(yhat, y)
                l.backward()
                losses.append(l.item())
                optim.step()
            print('train | loss: ', np.mean(losses).round(4))
            log = []
            with torch.no_grad():
                for idx in make_mini_batch(val_size, BATCHSIZE, shuffle=False):
                    x, y = get_Xy('val', idx)
                    yhat = model(x, X_train, Y_train, training=False)
                    if n_classe == 2: y = y.float()
                    log.append(loss_fn(yhat, y).item())
                best_score, patience = get_patience(best_score,np.mean(log), patience)
            epoch += 1

            with torch.no_grad():
                # Données test
                log = []
                for idx in make_mini_batch(test_size, BATCHSIZE, shuffle=False):
                    x, y = get_Xy('test', idx)
                    yhat = model(x, X_train, Y_train, training=False)
                    log.append(evaluate(yhat, y, n_classe))
                if is_regression: 
                    print(f'test {data_name} | loss: ', np.mean(log).round(4))
                else: 
                    l, acc = np.mean(log,0).round(4)
                    print(f'test {data_name} | acc: {acc} | loss: {l}')
                log = []

if __name__ == '__main__':
    main()