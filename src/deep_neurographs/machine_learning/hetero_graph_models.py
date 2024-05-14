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
from torch_geometric.nn import GCNConv, HeteroConv, Linear


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network.

    """

    def __init__(
        self,
        n_branch_features,
        n_proposal_features,
        hidden_channels,
        num_layers=2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Linear layers
        self.input_branches = Linear(n_branch_features, hidden_channels)
        self.input_proposals = Linear(n_proposal_features, hidden_channels)
        self.output = Linear(hidden_channels // 2, 1)

        # Convolutional layers
        self.conv1 = HeteroConv(
            {
                ("proposal", "to", "proposal"): GATConv(-1, hidden_channels),
                ("branch", "to", "branch"): GATConv(-1, hidden_channels),
                ("branch", "to", "proposal"): GATConv(
                    (-1, -1), hidden_channels, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        hidden_channels = hidden_channels // 2
        self.conv2 = HeteroConv(
            {
                ("proposal", "to", "proposal"): GATConv(-1, hidden_channels),
                ("branch", "to", "branch"): GATConv(-1, hidden_channels),
                ("branch", "to", "proposal"): GATConv(
                    (-1, -1), hidden_channels // 2, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        # Nonlinear activation
        self.dropout = Dropout(0.3)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        #self.init_weights()

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
        layers = [
            self.input_branches,
            self.input_proposals,
            self.conv1,
            self.conv2,
            self.output,
        ]
        for layer in layers:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x_dict, edge_index_dict):
        # Input
        x_dict = self.input_branches(x_dict["branch"], edge_index_dict)
        x_dict = self.input_proposals(x_dict["proposal"], edge_index_dict)
        x_dict = {key: self.leaky_relu(x_dict) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x_dict) for key, x in x_dict.items()}

        # Convolutional layers
        stop
        return self.output(x_dict["proposal"])


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
