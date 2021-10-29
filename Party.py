import numpy as np
import torch
import torch.optim as optim

from Nets import MLP


class Party(object):
    def __init__(self, args, feature=[]):
        self.args = args
        self.feature = feature
        self.feature_num = len(feature)
        self.model = MLP(dim_in=self.feature_num, dim_out=1).to(args.device)
        self.optimizer = optim.SGD(self.model.parameters(), args.lr)