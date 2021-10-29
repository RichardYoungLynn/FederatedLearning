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


def load_data(filename: str) -> np.ndarray:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    return arr


def train(dataloader: DataLoader, parties, args, loss: torch.nn.Module):
    party_num = args.party_num
    count = len(dataloader)
    total_loss = 0
    true_num = 0.0
    for batch_idx, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(args.device), label.to(args.device)
        predict=torch.zeros(args.batch_size,1).to(args.device)
        for i in range(party_num):
            parties[i].model.train()
            party_feature=feature[:, parties[i].feature]
            predict+=parties[i].model(party_feature)
            parties[i].optimizer.zero_grad()
        predict = torch.div(predict, party_num)
        predict = torch.squeeze(predict)
        loss_val = loss(predict, label)
        loss_val.backward()
        for i in range(party_num):
            parties[i].optimizer.step()
        true_num += ((predict.ge(0.5) == label).sum().item() / args.batch_size)
        total_loss += loss_val.item()
    return total_loss / count, true_num / count


def test(dataloader: DataLoader, parties, args, loss: torch.nn.Module):
    party_num = args.party_num
    count = len(dataloader)
    total_loss = 0
    true_num = 0.0
    for batch_idx, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(args.device), label.to(args.device)
        predict = torch.zeros(args.batch_size, 1).to(args.device)
        for i in range(party_num):
            parties[i].model.eval()
            party_feature = feature[:, parties[i].feature]
            predict += parties[i].model(party_feature)
        predict = torch.div(predict, party_num)
        predict = torch.squeeze(predict)
        loss_val = loss(predict, label)
        true_num += ((predict.ge(0.5) == label).sum().item() / args.batch_size)
        total_loss += loss_val.item()
    return total_loss / count, true_num / count


def create_party(args):
    party_num = args.party_num
    party_feature_num = args.party_feature_num
    total_feature_num=args.total_feature_num
    feature_list=[i for i in range(total_feature_num)]
    parties = []
    for i in range(party_num):
        random_set=set(np.random.choice(feature_list, party_feature_num[i], replace=False))
        feature_list=list(set(feature_list)-random_set)
        party=Party(args,list(random_set))
        parties.append(party)
    return parties


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train_filename = "dataset/default of credit card clients dataset/train.npz"
    test_filename = "dataset/default of credit card clients dataset/test.npz"
    train_arr = load_data(train_filename)
    test_arr = load_data(test_filename)
    parties=create_party(args)
    dataloader_train=make_dataloader(train_arr,args,shuffle=True, drop_last=True)
    dataloader_test = make_dataloader(test_arr, args, shuffle=True, drop_last=True)
    loss = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        train_loss, train_auc = train(dataloader_train, parties, args, loss)
        print(f"epoch: {epoch}, train loss: {train_loss}, train_auc: {train_auc}")

        if epoch % 5 == 0:
            test_loss, test_auc = test(dataloader_test,parties, args, loss)
            print(f"epoch: {epoch}, test_loss: {test_loss}, test_auc: {test_auc}")
