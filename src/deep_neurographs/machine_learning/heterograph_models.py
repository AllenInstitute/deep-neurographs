"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import re

import torch
import torch.nn.init as init
from torch import nn
from torch.nn import Dropout, LeakyReLU
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn import HeteroConv, Linear

from deep_neurographs.machine_learning.models import ConvNet


class HGAT(torch.nn.Module):
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
        node_dict,
        edge_dict,
        device=None,
        hidden_dim=96,
        dropout=0.3,
        heads_1=2,
        heads_2=2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Nonlinear activation
        self.dropout = dropout
        self.dropout_layer = Dropout(dropout)
        self.leaky_relu = LeakyReLU()

        # Initial Embedding
        self.input_nodes = nn.ModuleDict()
        for key, d in node_dict.items():
            self.input_nodes[key] = nn.Linear(d, hidden_dim, device=device)

        self.input_edges = nn.ModuleDict()
        for key, d in edge_dict.items():
            self.input_edges[str(key)] = nn.Linear(d, hidden_dim, device=device)

        # Layer dimensions
        hidden_dim_1 = hidden_dim
        hidden_dim_2 = hidden_dim_1 * heads_2
        output_dim = hidden_dim_1 * heads_1 * heads_2

        # Message passing layers
        self.gat1 = self.init_gat_layer(hidden_dim_1, hidden_dim_1, heads_1)
        self.gat2 = self.init_gat_layer(hidden_dim_2, hidden_dim_1, heads_2)
        self.output = Linear(output_dim, 1).to(device)

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
            key = reformat_edge_key(key)
            edge_attr_dict[key] = f(edge_attr_dict[key])
        edge_attr_dict = self.activation(edge_attr_dict)

        # Message passing layers
        x_dict = self.gat1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.gat2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )

        # Output
        x_dict = self.output(x_dict["proposal"])
        return x_dict


class MultiModalHGAT(HGAT):
    """
    Heterogeneous graph attention network that uses multimodal features which
    includes an image patch of the proposal and a vector of geometric and
    graphical features.

    """

    def __init__(
        self,
        node_dict,
        edge_dict,
        device=None,
        hidden_dim=64,
        dropout=0.3,
        heads_1=2,
        heads_2=2,
    ):
        # Call super constructor
        super().__init__(
            node_dict,
            edge_dict,
            device,
            hidden_dim * 2,
            dropout,
            heads_1,
            heads_2,
        )

        # Patch Embedding
        self.input_patches = ConvNet((48, 48, 48), hidden_dim)

        # Node Embedding
        proposal_dim = node_dict["proposal"]
        branch_dim = node_dict["branch"]
        self.input_nodes = nn.ModuleDict({
            "proposal": nn.Linear(proposal_dim, hidden_dim, device=device),
            "branch": nn.Linear(branch_dim, hidden_dim * 2, device=device),
        })

        # Edge Embedding
        self.input_edges = dict()
        for key, d in edge_dict.items():
            self.input_edges[key] = nn.Linear(
                d, hidden_dim * 2, device=device
            )

        # Initialize weights
        self.init_weights()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Input - Patches
        x_patch = self.input_patches(x_dict["patch"])
        del x_dict["patch"]

        # Input - Nodes
        for key, f in self.input_nodes.items():
            x_dict[key] = f(x_dict[key])
        x_dict = self.activation(x_dict)

        # Input - Edges
        for key, f in self.input_edges.items():
            key = tuple(key)
            edge_attr_dict[key] = f(edge_attr_dict[key])
        edge_attr_dict = self.activation(edge_attr_dict)

        # Concatenate multimodal embeddings
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_patch), dim=1)

        # Message passing layers
        x_dict = self.gat1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.gat2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )

        # Output
        x_dict = self.output(x_dict["proposal"])
        return x_dict


# --- Utils ---
def reformat_edge_key(key):
    if type(key) is str:
        return tuple([rm_non_alphanumeric(s) for s in key.split(",")])
    else:
        return key


def rm_non_alphanumeric(s):
    return re.sub(r'\W+', '', s)
