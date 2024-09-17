# -*- coding: utf-8 -*-
# time: 2024/6/19 12:05
# file: mamba_reshape.py
# author: Shuai


import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm import Mamba2


class mamba_reshape(nn.Module):
    def __init__(self, inputs=32, model_name='mamba', outputs=1):
        super(mamba_reshape, self).__init__()
        if model_name == 'mamba':
            self.mamba = Mamba(d_model=inputs,  # Model dimension d_model
                               d_state=16,  # SSM state expansion factor, typically 64 or 128
                               d_conv=4,  # Local convolution width
                               expand=2,  # Block expansion factor
                               )
            self.linear = nn.Sequential(
                nn.Linear(32, 16),
                nn.Linear(16, outputs),
            )
        if model_name == 'mamba2':
            self.mamba = Mamba2(d_model=inputs,  # Model dimension d_model
                                d_state=64,  # SSM state expansion factor, typically 64 or 128
                                d_conv=4,  # Local convolution width
                                expand=2,  # Block expansion factor
                                headdim=8,  # just make sure d_model * expand / headdim = multiple of 8
                                )
            self.linear = nn.Sequential(
                nn.Linear(32, 1)
            )

    def forward(self, x):
        # [b_s, 366, 32, 64, 64]
        shapes = x.shape
        x = self.reshape(x)
        # [b_s*64*64, 366, 32]
        out = self.mamba(x)
        out = self.linear(out)
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
