"""
Created on Wed July 19 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of graph convolutional network (GCN).

"""

import torch
from torch_geometric.data import Data


def init_training_data(supergraph, node_features, edge_features):
    """
    """
    x = torch.tensor(node_features, dtype=torch.long)
    edge_index = torch.tensor(list(supergraph.edges()), dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.long)
    data = Data(
        x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_features
    )
    return data
