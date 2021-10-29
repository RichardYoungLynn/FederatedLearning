import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from data_process import *
from Nets import MLP
from options import args_parser
from Party import Party
from VFL import *


def select_party(args,all_parties):
    parties=[]
    budget=args.budget
    party_num=args.party_num
    party_feature_num=args.party_feature_num
    dp=np.zeros((party_num,budget))
    w=[i for i in party_feature_num]
    v=[1*budget]
    for i in range(party_num):
        j=budget
        while j>=w[i]:
            k=j-w[i]
            if k>=0:
                if i==0:
                    dp[i][j] = v[i]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][k] + v[i])
    for i in range(party_num-1,0,-1):
        if dp[i][budget-1]!=dp[i-1][budget-1]:
            parties.append(all_parties[i])
    return parties


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train_filename = "dataset/default of credit card clients dataset/train.npz"
    test_filename = "dataset/default of credit card clients dataset/test.npz"
    train_arr = load_data(train_filename)
    test_arr = load_data(test_filename)
    all_parties = create_party(args)
    parties = select_party(args,all_parties)
    dataloader_train = make_dataloader(train_arr, args, shuffle=True, drop_last=True)
    dataloader_test = make_dataloader(test_arr, args, shuffle=True, drop_last=True)
    loss = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        train_loss, train_auc = train(dataloader_train, parties, args, loss)
        print(f"epoch: {epoch}, train loss: {train_loss}, train_auc: {train_auc}")

        if epoch % 5 == 0:
            test_loss, test_auc = test(dataloader_test, parties, args, loss)
            print(f"epoch: {epoch}, test_loss: {test_loss}, test_auc: {test_auc}")
