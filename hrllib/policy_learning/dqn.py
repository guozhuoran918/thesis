
import torch
import torch.nn.functional
import os
import numpy as np
from collections import namedtuple

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(MLP, self).__init__()
        self.params = parameter
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )
        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        q_values = self.policy_layer(x)
        return q_values

