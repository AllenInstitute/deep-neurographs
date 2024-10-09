"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import numpy as np
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import Dropout, LeakyReLU
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import HeteroConv, Linear


class HeteroGNN(torch.nn.Module):  # change to HGAT
    """
    Heterogeneous graph attention network that classifies proposals.

    """

    def __init__(
        self,
        node_dict,
        edge_dict,
        device=None,
        dropout=0.3,
        heads_1=1,
        heads_2=1,
        scale_hidden_dim=2,
    ):
        """
        Constructs a heterogeneous graph attention network.

        Parameters
        ----------
        ...

        Returns
        -------
        None

        """
        super().__init__()
        # Instance attributes
        self.device = device
        self.dropout = dropout

        # Feature vector sizes
        hidden_dim = scale_hidden_dim * np.max(list(node_dict.values()))
        output_dim = heads_1 * heads_2 * hidden_dim

        # Linear layers
        self.input_nodes = self.init_linear_layer(hidden_dim, node_dict)
        self.input_edges = self.init_linear_layer(hidden_dim, edge_dict)
        self.output = Linear(output_dim, 1).to(device)

        # Message passing layers
        self.conv1 = self.init_gat_layer(hidden_dim, hidden_dim, heads_1)  # change name
        edge_dim = hidden_dim
        hidden_dim = heads_1 * hidden_dim

        self.conv2 = self.init_gat_layer(hidden_dim, edge_dim, heads_2)  # change name

        # Nonlinear activation
        self.dropout = Dropout(dropout)  # change name
        self.leaky_relu = LeakyReLU()

        # Initialize weights
        self.init_weights()

    # --- Initialize architecture ---
    def init_linear_layer(self, hidden_dim, my_dict):
        linear_layer = dict()
        for key, dim in my_dict.items():
            linear_layer[key] = nn.Linear(dim, hidden_dim, device=self.device)
        return linear_layer

    def init_gat_layer(self, hidden_dim, edge_dim, heads):
        gat_layers = HeteroConv(
            {
                ("proposal", "edge", "proposal"): self.init_gat_layer_same(
                    hidden_dim, edge_dim, heads
                ),
                ("branch", "edge", "branch"): self.init_gat_layer_same(
                    hidden_dim, edge_dim, heads
                ),
                ("branch", "edge", "proposal"): self.init_gat_layer_mixed(
                    hidden_dim, edge_dim, heads
                ),
            },
            aggr="sum",
        )
        return gat_layers

    def init_gat_layer_same(self, hidden_dim, edge_dim, heads):
        gat_layer = GATConv(
            -1,
            hidden_dim,
            dropout=self.dropout,
            edge_dim=edge_dim,
            heads=heads,
        )
        return gat_layer

    def init_gat_layer_mixed(self, hidden_dim, edge_dim, heads):
        gat_layer = GATConv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            add_self_loops=False,
            edge_dim=edge_dim,
            heads=heads,
        )
        return gat_layer

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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Input - Nodes
        x_dict = {key: f(x_dict[key]) for key, f in self.input_nodes.items()}
        x_dict = self.activation(x_dict)

        # Input - Edges
        for key, f in self.input_edges.items():
            edge_attr_dict[key] = f(edge_attr_dict[key])
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


class MultiModalHGAT(HeteroGNN):
    pass
