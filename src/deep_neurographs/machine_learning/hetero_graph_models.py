"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

import torch
import torch.nn.init as init
from torch import nn
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
        n_edge_features,
        hidden_dim,
        num_layers=2,
    ):
        """
        Constructs a heterogeneous graph neural network.

        """
        super().__init__()
        # Linear layers
        self.input = nn.ModuleDict({
            "branch": nn.Linear(n_branch_features, hidden_dim),
            "proposal": nn.Linear(n_proposal_features, hidden_dim),
            "to": nn.Linear(n_edge_features, hidden_dim),
            
        })
        self.output = Linear(hidden_dim // 2, 1)

        # Convolutional layers
        self.conv1 = HeteroConv(
            {
                ("proposal", "to", "proposal"): GATConv(-1, hidden_dim), #nn.Linear(hidden_dim, hidden_dim),
                ("branch", "to", "branch"): GATConv(-1, hidden_dim), # nn.Linear(hidden_dim, hidden_dim),
                ("branch", "to", "proposal"): GATConv(
                    (-1, -1), hidden_dim, add_self_loops=False
                ),
            },
            aggr="sum",
        )

        hidden_dim = hidden_dim // 2
        self.conv2 = HeteroConv(
            {
                ("proposal", "to", "proposal"): GATConv(-1, hidden_dim),
                ("branch", "to", "branch"): GATConv(-1, hidden_dim),
                ("branch", "to", "proposal"): GATConv(
                    (-1, -1), hidden_dim // 2, add_self_loops=False
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
        x = torch.cat([self.linear_layers[key](x_dict[key]) for key in x_dict], dim=0)
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Convolutional layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        return self.output(x_dict["proposal"])
