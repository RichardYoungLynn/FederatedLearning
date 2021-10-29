import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from data_process import *
from Nets import MLP
from options import args_parser


def train(args,dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module, optimizer: optim.Optimizer):
    model.train()
    count = len(dataloader)
    total_loss = 0
    true_num = 0.0
    for batch_idx, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(args.device), label.to(args.device)
        predict = model(feature).to(args.device)
        predict = torch.squeeze(predict)
        loss_val = loss(predict, label)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        true_num += ((predict.ge(0.5) == label).sum().item() / args.batch_size)
        total_loss += loss_val.item()

    return total_loss / count, true_num / count


def test(args,dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module):
    model.eval()
    count = len(dataloader)
    total_loss = 0
    true_num=0.0
    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(dataloader):
            feature, label = feature.to(args.device), label.to(args.device)
            predict = model(feature).to(args.device)
            predict=torch.squeeze(predict)
            loss_val = loss(predict, label)

            true_num += ((predict.ge(0.5) == label).sum().item() / args.batch_size)
            total_loss += loss_val.item()

    return total_loss / count, true_num / count


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train_filename = "dataset/default of credit card clients dataset/train.npz"
    test_filename = "dataset/default of credit card clients dataset/test.npz"
    train_arr = load_data(train_filename)
    test_arr = load_data(test_filename)
    train_dataloader = make_dataloader(train_arr, args, shuffle=True, drop_last=False)
    test_dataloader = make_dataloader(test_arr, args, shuffle=True, drop_last=False)
    model = MLP(dim_in=23, dim_out=1).to(args.device)
    loss = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        train_loss, auc = train(args,train_dataloader,model,loss,optimizer)
        print(f"epoch: {epoch}, train loss: {train_loss}")

        if epoch % 5 == 0:
            test_loss, auc = test(args,test_dataloader,model,loss)
            print(f"epoch: {epoch}, test_loss: {test_loss}, auc: {auc}")

    train_x=train_arr[:,:-1]
    train_y=train_arr[:,-1]
    test_x = test_arr[:, :-1]
    test_y = test_arr[:, -1]
    # LogisticRegression
    reg = LogisticRegression()
    reg.fit(train_x, train_y)
    print('LogisticRegression train',reg.score(train_x, train_y))
    print('LogisticRegression test', reg.score(test_x, test_y))
    # KNeighborsClassifier
    knn = KNeighborsClassifier(5)
    knn.fit(train_x, train_y)
    print('KNeighborsClassifier train',knn.score(train_x, train_y))
    print('KNeighborsClassifier test', knn.score(test_x, test_y))
    # RandomForestClassifier
    rfc = RandomForestClassifier(100)
    rfc.fit(train_x, train_y)
    print('RandomForestClassifier train',rfc.score(train_x, train_y))
    print('RandomForestClassifier test', rfc.score(test_x, test_y))