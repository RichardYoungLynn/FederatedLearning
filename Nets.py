import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 27)
        self.relu = nn.ReLU()
        self.layer_hidden_1 = nn.Linear(27, 14)
        self.layer_hidden_2 = nn.Linear(14, 7)
        self.layer_output = nn.Linear(7, dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden_1(x)
        x = self.relu(x)
        x = self.layer_hidden_2(x)
        x = self.relu(x)
        x = self.layer_output(x)
        return self.sigmoid(x)