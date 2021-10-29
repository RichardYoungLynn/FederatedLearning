import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import random
from scipy import stats

attr_classes = ["ID",
                "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",
                "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20",
                "X21", "X22", "X23",
                "Y"]


def load_data(filename: str) -> np.ndarray:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    return arr


def data_cleaning(odata):
    odata = odata.drop(['ID'], axis=1)
    odata.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
    # EDUCATION为0，5，6的人数很少 可以去掉
    odata = odata[(odata.EDUCATION!=0) & (odata.EDUCATION!=5) & (odata.EDUCATION!=6)]
    # MARRIAGE为0的人数很少 可以去掉
    odata = odata[(odata.MARRIAGE != 0)]
    # PAY是PAY_AMT和BILL_AMT信息的一种转化 可以去掉PAY_AMT和BILL_AMT 只保留PAY
    odata = odata.drop(
        ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
         'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis=1)
    return odata


def df_to_arr(df: pd.DataFrame) -> np.ndarray:
    names = df.columns.tolist()
    rows = len(df)
    arr = []
    for i in range(rows):
        arr_row = []
        for col in names:
            raw_val = df.iloc[i][col]
            arr_row.append(raw_val)
        arr.append(arr_row)
    res = np.array(arr, dtype=np.float32)
    return res


def load_excel(filename: str) -> pd.DataFrame:
    df = pd.read_excel(io=filename, header=1, names=attr_classes)
    return df


def get_mean_std(df: pd.DataFrame):
    res = {}
    for col in attr_classes[1:-1]:
        mean = df[col].mean()
        std = df[col].std()
        res[col] = {"mean": mean, "std": std}
    return res


def convert_excel_to_arr(filename: str, mean_std: Dict[str, Dict[str, float]]) -> np.ndarray:
    df = load_excel(filename)
    df.drop(columns=["ID"], inplace=True)
    for col in mean_std:
        mean = mean_std[col]["mean"]
        std = mean_std[col]["std"]
        df[col] = (df[col] - mean) / std
    arr = df_to_arr(df)
    return arr


def split_feature(arr: np.ndarray) -> Tuple[np.ndarray, ...]:
    feature, label = arr[:, :-1], arr[:, -1:]
    a_feature_size = 12
    a_feature = feature[:, :a_feature_size]
    b_feature = feature[:, a_feature_size:]
    return a_feature, b_feature, label


def split_train_test_dataset(arr: np.ndarray) -> Tuple[np.ndarray, ...]:
    train_arr, test_arr = arr[:22500, :], arr[22500:, :]
    return train_arr, test_arr


def make_dataloader(arr: np.ndarray, args, shuffle: bool = False,drop_last: bool = False) -> DataLoader:
    tensor = torch.tensor(arr)
    feature= tensor[:, :-1].float()
    label = tensor[:, -1].float()
    dataset = TensorDataset(feature, label)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


if __name__ == '__main__':
    dataset_dir="dataset/default of credit card clients dataset/default of credit card clients.xls"
    df = load_excel(dataset_dir)
    mean_std = get_mean_std(df)
    arr = convert_excel_to_arr(dataset_dir, mean_std)
    train_arr, test_arr = split_train_test_dataset(arr)
    np.savez("dataset/default of credit card clients dataset/train.npz", train_arr)
    np.savez("dataset/default of credit card clients dataset/test.npz", test_arr)

    # dataset_dir = "dataset/default of credit card clients dataset/default of credit card clients.xls"
    # odata = pd.read_excel(dataset_dir, header=[1], sheet_name="Data")
    # odata = data_cleaning(odata)
    # print(odata.head())
    # print(odata.info())
    # data_arr = df_to_arr(odata)
    # train_arr, test_arr = split_train_test_dataset(data_arr)
    # np.savez("dataset/default of credit card clients dataset/train.npz", train_arr)
    # np.savez("dataset/default of credit card clients dataset/test.npz", test_arr)
