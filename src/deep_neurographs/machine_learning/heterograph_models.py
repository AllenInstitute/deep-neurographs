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
from torch_geometric.nn import HEATConv, HeteroConv, Linear

from deep_neurographs import machine_learning as ml

CONV_TYPES = ["GATConv", "GCNConv"]
DROPOUT = 0.3
HEADS_1 = 1
HEADS_2 = 1


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous graph attention network that classifies proposals.

    """
    # Class attributes
    relation_types = [
        ("proposal", "edge", "proposal"),
        ("branch", "edge", "proposal"),
        ("branch", "edge", "branch"),
    ]

    def __init__(
        self,
        device=None,
        scale_hidden=2,
        dropout=DROPOUT,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Feature vector sizes
        node_dict = ml.feature_generation.get_node_dict()
        edge_dict = ml.feature_generation.get_edge_dict()
        hidden_dim = scale_hidden * np.max(list(node_dict.values()))
        output_dim = heads_1 * heads_2 * hidden_dim

        # Nonlinear activation
        self.dropout = dropout
        self.dropout_layer = Dropout(dropout)
        self.leaky_relu = LeakyReLU()

        # Linear layers        
        self.input_nodes = nn.ModuleDict()
        self.input_edges = dict()
        for key, d in node_dict.items():
            self.input_nodes[key] = nn.Linear(d, hidden_dim, device=device)
        for key, d in edge_dict.items():
            self.input_edges[key] = nn.Linear(d, hidden_dim, device=device)
        self.output = Linear(output_dim, 1).to(device)

        # Message passing layers
        self.gat1 = self.init_gat_layer(hidden_dim, hidden_dim, heads_1)
        self.gat2 = self.init_gat_layer(hidden_dim * heads_2, hidden_dim, heads_2)

        # Initialize weights
        self.init_weights()

    # --- Class methods ---
    @classmethod
    def get_relation_types(cls):
        return cls.relation_types

    # --- Architecture ---
    def init_gat_layer(self, hidden_dim, edge_dim, heads):
        gat_dict = dict()
        for r in self.get_relation_types():
            is_same = True if r[0] == r[2] else False
            init_gat = self.init_gat_same if is_same else self.init_gat_mixed
            gat_dict[r] = init_gat(hidden_dim, edge_dim, heads)
        return HeteroConv(gat_dict, aggr="sum")

    def init_gat_same(self, hidden_dim, edge_dim, heads):
        gat_layer = GATConv(
            -1,
            hidden_dim,
            dropout=self.dropout,
            edge_dim=edge_dim,
            heads=heads,
        )
        return gat_layer

    def init_gat_mixed(self, hidden_dim, edge_dim, heads):
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
        x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
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
        x_dict = self.gat1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.gat2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )

        # Output
        x_dict = self.output(x_dict["proposal"])
        return x_dict
