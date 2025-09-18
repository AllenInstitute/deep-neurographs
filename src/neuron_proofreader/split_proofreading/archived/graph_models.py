"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

NOTE: THIS SCRIPT IS NO LONGER USED!

"""

import torch
import torch.nn.init as init
from torch.nn import Dropout, LeakyReLU, Linear
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import GCNConv

CONV_TYPES = ["GATConv", "GCNConv"]
DROPOUT = 0.3
HEADS_1 = 1
HEADS_2 = 1


class GNN(torch.nn.Module):
    """
    Class of graph neural networks.

    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        conv_type="GATConv",
        dropout=DROPOUT,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
    ):
        """
        Constructs a graph neural network.

        """
        super().__init__()
        # Linear layers
        attn_dim = heads_1 * heads_2 * hidden_dim
        output_dim = attn_dim if conv_type == "GATConv" else hidden_dim
        self.input = Linear(input_dim, hidden_dim)
        self.output = Linear(output_dim, 1)

        # Convolutional layers
        assert conv_type in CONV_TYPES, "conv_type is not supported"
        self.conv_type = conv_type
        self.conv1 = self.init_conv_layer(
            hidden_dim, hidden_dim, dropout, heads_1
        )
        if self.conv_type == "GATConv":
            hidden_dim = heads_1 * hidden_dim

        self.conv2 = self.init_conv_layer(
            hidden_dim, hidden_dim, dropout, heads_2
        )
        if self.conv_type == "GATConv":
            hidden_dim = heads_2 * hidden_dim

        # Activation
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

    def init_conv_layer(self, input_dim, output_dim, dropout, heads):
        """
        Initializes a graph convolutional layer.

        Parameters
        ----------
        input_dim : int
            Dimension of input feature vector.
        output_dim : int
            Dimension of output feature vector.
        dropout : float
            Dropout probability.
        heads : int
            Number of attention heads.

        Returns
        -------
        torch.nn.Module
            Graph convolutional layer.

        """
        if self.conv_type == "GCNConv":
            return GCNConv(input_dim, output_dim)
        elif self.conv_type == "GATConv":
            return GATConv(input_dim, output_dim, dropout=dropout, heads=heads)

    def init_weights(self):
        """
        Initializes linear and convolutional layers.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
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
        if self.conv_type == "GCNConv":
            x = self.leaky_relu(x)
            x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        if self.conv_type == "GCNConv":
            x = self.leaky_relu(x)
            x = self.dropout(x)

        # Output
        x = self.output(x)
        return x


class MLP(torch.nn.Module):
    """
    Class of multi-layer perceptrons.

    """

    def __init__(self, input_dim):
        super().__init__()
        # Linear layers
        self.input = Linear(input_dim, input_dim)
        self.linear1 = Linear(input_dim, input_dim // 2)
        self.linear2 = Linear(input_dim // 2, input_dim // 2)
        self.linear3 = Linear(input_dim, input_dim // 2)
        self.output = Linear(input_dim // 2, 1)

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
