import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_3(hidden)
        return output
