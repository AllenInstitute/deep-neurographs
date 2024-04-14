"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import torch
import torch.nn.functional as F
from torch.nn import ELU, Linear
from torch_geometric.nn import GATConv, GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, input_channels)
        self.conv2 = GCNConv(input_channels, input_channels // 2)
        self.conv3 = GCNConv(input_channels // 2, 1)
        self.ELU = ELU()

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)

        # Layer 3
        x = self.conv3(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = GATConv(input_channels, input_channels)
        self.conv2 = GATConv(input_channels, input_channels // 2)
        self.conv3 = GATConv(input_channels // 2, 1)
        self.ELU = ELU()

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        # x = self.ELU(x)
        # x = F.dropout(x, p=0.25)

        # Layer 2
        # x = self.conv2(x, edge_index)
        # x = self.ELU(x)
        # x = F.dropout(x, p=0.25)

        # Layer 3
        # x = self.conv3(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.linear1 = Linear(input_channels, input_channels)
        self.linear2 = Linear(input_channels, input_channels // 2)
        self.linear3 = Linear(input_channels // 2, 1)
        self.ELU = ELU()

    def forward(self, x, edge_index):
        x = self.linear1(x)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)
        x = self.linear2(x)
        x = self.ELU(x)
        x = F.dropout(x, p=0.25)
        x = self.linear3(x)
        return x
