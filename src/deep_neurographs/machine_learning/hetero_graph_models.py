"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import torch
import torch.nn.init as init
from torch import nn
from torch.nn import Dropout, LeakyReLU
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import GCNConv, HANConv, HeteroConv, Linear

CONV_TYPES = ["GATConv", "GCNConv"]
DROPOUT = 0.3
HEADS_1 = 1
HEADS_2 = 1


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network.

    """

    def __init__(
        self,
        node_dict,
        n_edge_features,
        hidden_dim,
        conv_type="GATConv",
        dropout=DROPOUT,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Linear layers
        attn_dim = heads_1 * heads_2 * hidden_dim
        output_dim = attn_dim if conv_type == "GATConv" else hidden_dim
        self.input = nn.ModuleDict(
            {key: nn.Linear(d, hidden_dim) for key, d in node_dict.items()}
        )
        self.output = Linear(output_dim, 1)

        # Convolutional layers
        assert conv_type in CONV_TYPES, "conv_type is not supported"
        self.conv_type = conv_type
        self.conv1 = HeteroConv(
            {
                ("proposal", "edge", "proposal"): self.init_conv_layer(
                    -1, hidden_dim, dropout, heads_1
                ),
                ("branch", "edge", "branch"): self.init_conv_layer(
                    -1, hidden_dim, dropout, heads_1
                ),
                ("branch", "edge", "proposal"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    add_self_loops=False,
                    heads=heads_1,
                ),
            },
            aggr="sum",
        )
        if conv_type == "GATConv":
            hidden_dim = heads_1 * hidden_dim

        self.conv2 = HeteroConv(
            {
                ("proposal", "edge", "proposal"): self.init_conv_layer(
                    -1, hidden_dim, dropout, heads_2
                ),
                ("branch", "edge", "branch"): self.init_conv_layer(
                    -1, hidden_dim, dropout, heads_2
                ),
                ("branch", "edge", "proposal"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    add_self_loops=False,
                    heads=heads_2,
                ),
            },
            aggr="sum",
        )
        if conv_type == "GATConv":
            hidden_dim = heads_2 * hidden_dim

        # Nonlinear activation
        self.dropout = Dropout(dropout)
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
        Initializes linear layers.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for layer in [self.input, self.output]:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x_dict, edge_index_dict):
        # Input
        x_dict = {key: lin(x_dict[key]) for key, lin in self.input.items()}
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Convolutional layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        if self.conv_type == "GCNConv":
            x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        if self.conv_type == "GCNConv":
            x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return self.output(x_dict["proposal"])


class HAN(torch.nn.Module):
    """
    Heterogeneous graph neural network.

    """

    def __init__(
        self,
        hidden_dim,
        metadata,
        node_dict,
        dropout=DROPOUT,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Linear layers
        self.input = nn.ModuleDict(
            {key: nn.Linear(d, hidden_dim) for key, d in node_dict.items()}
        )
        self.output = Linear(heads_1 * heads_2 * hidden_dim)

        # Convolutional layers
        self.conv1 = HANConv(
            hidden_dim,
            hidden_dim,
            heads=heads_1,
            dropout=dropout,
            metadata=metadata,
        )
        hidden_dim = heads_1 * hidden_dim

        self.conv2 = HANConv(
            hidden_dim,
            hidden_dim,
            heads=heads_2,
            dropout=dropout,
            metadata=metadata,
        )
        hidden_dim = heads_2 * hidden_dim

        # Nonlinear activation
        self.dropout = Dropout(dropout)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

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
        layers = [self.input, self.conv1, self.conv2, self.output]
        for layer in layers:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)
