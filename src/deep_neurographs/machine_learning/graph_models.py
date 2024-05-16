"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import torch
import torch.nn.init as init
from torch.nn import Dropout, LeakyReLU, Linear
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # Linear layers
        self.input = Linear(input_channels, input_channels)
        self.output = Linear(input_channels // 2, 1)

        # Convolutional layers
        self.conv1 = GCNConv(input_channels, input_channels // 2)
        self.conv2 = GCNConv(input_channels // 2, input_channels // 2)

        # Activation
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        layers = [self.conv1, self.conv2, self.input, self.output]
        for layer in layers:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x, edge_index):
        # Input
        x = self.input(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Output
        x = self.output(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # Linear layers
        self.input = Linear(input_channels, input_channels)
        self.output = Linear(input_channels // 2, 1)

        # Convolutional layers
        self.conv1 = GATConv(input_channels, input_channels)
        self.conv2 = GATConv(input_channels, input_channels // 2)

        # Activation
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        layers = [self.conv1, self.conv2, self.input, self.output]
        for layer in layers:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x, edge_index):
        # Input
        x = self.input(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Output
        x = self.output(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # Linear layers
        self.input = Linear(input_channels, input_channels)
        self.linear1 = Linear(input_channels, input_channels // 2)
        self.linear2 = Linear(input_channels // 2, input_channels // 2)
        self.linear3 = Linear(input_channels, input_channels // 2)
        self.output = Linear(input_channels // 2, 1)

        # Activation
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        layers = [
            self.linear1,
            self.linear2,
            self.linear3,
            self.input,
            self.output,
        ]
        for layer in layers:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x, edge_index):
        # Input
        x = self.input(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 1
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Output
        x = self.output(x)
        return x
