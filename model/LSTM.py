# -*- coding: utf-8 -*-
# time: 2024/5/31 15:28
# file: LSTM.py
# author: Shuai

import torch
import torch.nn as nn
import numpy as np


# LSTM
class LSTM(nn.Module):
    def __init__(self, inputs=32, outputs=1, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=inputs, hidden_size=16, num_layers=num_layers, batch_first=True,
                             bidirectional=False)
        self.reg1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.Linear(8, outputs)
        )

    def forward(self, x):
        # [b_s, 366, 32, 64, 64]
        shapes = x.shape
        x = self.reshape(x)
        # [b_s*64*64, 366, 32]
        out, _ = self.lstm1(x)
        out = self.reg1(out)
        out = self.return_shape(out, shapes)
        return out

    def reshape(self, data):
        data = data.clone().detach().requires_grad_(True)
        data = torch.transpose(data, 1, 3)
        data = torch.transpose(data, 2, 4)
        data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4])
        return data

    def return_shape(self, data, shapes):
        data = data.reshape(shapes[0], shapes[3], shapes[4], data.shape[1])
        data = torch.transpose(data, 2, 3)
        data = torch.transpose(data, 1, 2)
        return data
