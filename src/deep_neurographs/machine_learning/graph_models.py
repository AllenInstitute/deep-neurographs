"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

from torch.nn import ELU, Linear
from torch_geometric.nn import GCNConv

import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, input_channels // 2)
        self.conv2 = GCNConv(input_channels // 2, 1)
        self.ELU = ELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.linear1 = Linear(input_channels, input_channels // 2)
        self.linear2 = Linear(input_channels // 2, 1)
        self.ELU = ELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)
        x = self.linear2(x)
        return x
