# -*- coding: utf-8 -*-
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from torch import nn as nn
from torch.nn import Dropout


class MLP(nn.Module):
    """
    A module for the multi-layer perceptron (MLP)
    """
    def __init__(self, n_in, n_hidden, dropout=0, activation=nn.GELU()):
        super(MLP, self).__init__()
        self.linear = nn.Linear(n_in, n_hidden)
        self.f = activation
        self.dropout = Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.f(self.linear(x))
        x = self.dropout(x)
        return x
