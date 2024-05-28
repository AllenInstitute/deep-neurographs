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
    Heterogeneous graph neural network that utilizes edge features.

    """

    def __init__(
        self,
        node_dict,
        edge_dict,
        hidden_dim,
        dropout=DROPOUT,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Linear layers
        output_dim = heads_1 * heads_2 * hidden_dim
        self.input_nodes = nn.ModuleDict(
            {key: nn.Linear(d, hidden_dim) for key, d in node_dict.items()}
        )
        self.input_edges = {
            key: nn.Linear(d, hidden_dim) for key, d in edge_dict.items()
        }
        self.output = Linear(output_dim, 1)

        # Convolutional layers
        self.conv1 = HeteroConv(
            {
                ("proposal", "edge", "proposal"): GATConv(
                    -1,
                    hidden_dim,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    heads=heads_1,
                ),
                ("branch", "edge", "branch"): GATConv(
                    -1,
                    hidden_dim,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    heads=heads_1,
                ),
                ("branch", "edge", "proposal"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    add_self_loops=False,
                    edge_dim=hidden_dim,
                    heads=heads_1,
                ),
            },
            aggr="sum",
        )
        edge_dim = hidden_dim
        hidden_dim = heads_1 * hidden_dim

        self.conv2 = HeteroConv(
            {
                ("proposal", "edge", "proposal"): GATConv(
                    -1,
                    hidden_dim,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    heads=heads_2,
                ),
                ("branch", "edge", "branch"): GATConv(
                    -1,
                    hidden_dim,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    heads=heads_2,
                ),
                ("branch", "edge", "proposal"): GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                    heads=heads_2,
                ),
            },
            aggr="sum",
        )
        hidden_dim = heads_2 * hidden_dim

        # Nonlinear activation
        self.dropout = Dropout(dropout)
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

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
        for layer in [self.input_nodes, self.output]:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Input - Nodes
        x_dict = {key: f(x_dict[key]) for key, f in self.input_nodes.items()}
        x_dict = self.activation(x_dict)

        # Input - Edges
        edge_attr_dict = {
            key: f(edge_attr_dict[key]) for key, f in self.input_edges.items()
        }
        edge_attr_dict = self.activation(edge_attr_dict)

        # Convolutional layers
        x_dict = self.conv1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.conv2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )

        # Output
        x_dict = self.output(x_dict["proposal"])
        return x_dict

    def activation(self, x_dict):
        """
        Applies nonlinear activation

        Parameters
        ----------
        x_dict : dict
            Dictionary that maps node/edge types to feature matrices.

        Returns
        -------
        dict
            Feature matrices with activation applied.

        """
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return x_dict


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
