"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training graph neural networks.

# explain branches vs edges terminology

"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data import HeteroData as HeteroGraphData

from deep_neurographs.machine_learning import feature_generation


# Wrapper
def init(neurograph, branch_features, proposal_features, heterogeneous=False):
    """
    Initializes a dataset that can be used to train a graph neural network.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that dataset is built from.
    branch_features : dict
        Feature vectors corresponding to branches such that the keys are a
        frozenset of the node pair and values are the corresponding feature
        vectors.
    proposal_features : dict
        Feature vectors corresponding to proposals such that the keys are a
        frozenset of the node pair and values are the corresponding feature
        vectors.
    heterogeneous : bool
        Indication of whether dataset should be stored as a heterogeneous
        graph.

    Returns
    -------
    GraphDataset, HeteroGraphDataset
        Custom dataset.

    """
    # Extract features
    x_branches, _, idxs_branches = feature_generation.get_matrix(
        neurograph, branch_features, "GraphNeuralNet"
    )
    x_proposals, y_proposals, idxs_proposals = feature_generation.get_matrix(
        neurograph, proposal_features, "GraphNeuralNet"
    )

    # Initialize data
    if heterogeneous:
        graph_dataset = HeteroGraphDataset(
            neurograph, x_branches, x_proposals, idxs_branches, idxs_proposals
        )
    else:
        graph_dataset = GraphDataset(
            neurograph,
            x_branches,
            x_proposals,
            y_proposals,
            idxs_branches,
            idxs_proposals,
        )

    return graph_dataset


# Datasets
class GraphDataset:
    """
    Custom dataset for homogenous graphs.

    """
    def __init__(
        self,
        neurograph,
        x_branches,
        x_proposals,
        y_proposals,
        idxs_branches,
        idxs_proposals,
    ):
        # Combine feature matrices
        x = np.vstack([x_proposals, x_branches]).astype(np.float32)
        x = torch.tensor(x)
        y = torch.tensor(y_proposals.astype(np.float32))
        idxs_branches = upd_idxs(idxs_branches, x_proposals.shape[0])
        self.idxs_branches = add_edge_to_idx(idxs_branches)
        self.idxs_proposals = add_edge_to_idx(idxs_proposals)
        self.n_proposals = len(y_proposals)

        # Initialize data
        edge_index = set_edge_index(neurograph, idxs_branches, idxs_proposals)
        self.data = GraphData(x=x, y=y, edge_index=edge_index)


class HeteroGraphDataset:
    """
    Custom dataset for heterogenous graphs.

    """
    def __init__(
        self,
        neurograph,
        x_branches,
        x_proposals,
        y_proposals,
        idxs_branches,
        idxs_proposals,
    ):
        # Update idxs
        idxs_branches = add_edge_to_idx(idxs_branches)
        idxs_proposals = add_edge_to_idx(idxs_proposals)

        # Init dataset
        data = HeteroGraphData()
        data["branch"].x = x_branches
        data["proposal"].x = x_proposals
        data["proposal", "to", "proposal"] = None
        data["proposal", "to", "branch"] = None
        data["branch", "to", "branch"] = None


# -- utils --
def upd_idxs(idxs, shift):
    """
    Updates index transform dictionary "idxs" by shifting each index by
    "shift".

    idxs : dict
        ...
    shift : int
        ...

    Returns
    -------
    idxs : dict
        Updated index transform dictinoary.

    """
    idxs["block_to_idxs"] = upd_set(idxs["block_to_idxs"], shift)
    idxs["idx_to_edge"] = upd_dict(idxs["idx_to_edge"], shift)
    return idxs


def upd_set(my_set, shift):
    shifted_set = set()
    for element in my_set:
        shifted_set.add(element + shift)
    return shifted_set


def upd_dict(my_dict, shift):
    shifted_dict = dict()
    for key, value in my_dict.items():
        shifted_dict[key + shift] = value
    return shifted_dict


def add_edge_to_idx(idxs):
    idxs["edge_to_idx"] = dict()
    for idx, edge in idxs["idx_to_edge"].items():
        idxs["edge_to_idx"][edge] = idx
    return idxs


def set_edge_index(neurograph, idxs_branches, idxs_proposals):
    # Initializations
    branches_line_graph = nx.line_graph(neurograph)
    proposals_line_graph = init_proposals_line_graph(neurograph)

    # Compute edges
    edge_index = branch_to_branch(branches_line_graph, idxs_branches)
    edge_index.extend(
        proposal_to_proposal(proposals_line_graph, idxs_proposals)
    )
    edge_index.extend(
        branch_to_proposal(neurograph, idxs_branches, idxs_proposals)
    )

    # Reshape
    edge_index = np.array(edge_index, dtype=np.int64).tolist()
    edge_index = torch.Tensor(edge_index).t().contiguous()
    return edge_index.long()


def init_proposals_line_graph(neurograph):
    proposals_graph = nx.Graph()
    proposals_graph.add_edges_from(list(neurograph.proposals.keys()))
    return nx.line_graph(proposals_graph)


def branch_to_branch(branches_line_graph, idxs_branches):
    edge_index = []
    for e1, e2 in branches_line_graph.edges:
        v1 = idxs_branches["edge_to_idx"][frozenset(e1)]
        v2 = idxs_branches["edge_to_idx"][frozenset(e2)]
        edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index


def proposal_to_proposal(proposals_line_graph, idxs_proposals):
    edge_index = []
    for e1, e2 in proposals_line_graph.edges:
        v1 = idxs_proposals["edge_to_idx"][frozenset(e1)]
        v2 = idxs_proposals["edge_to_idx"][frozenset(e2)]
        edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index


def branch_to_proposal(neurograph, idxs_branches, idxs_proposals):
    edge_index = []
    for e in neurograph.proposals.keys():
        i, j = tuple(e)
        v1 = idxs_proposals["edge_to_idx"][frozenset(e)]
        for k in neurograph.neighbors(i):
            v2 = idxs_branches["edge_to_idx"][frozenset((i, k))]
            edge_index.extend([[v1, v2], [v2, v1]])
        for k in neurograph.neighbors(j):
            v2 = idxs_branches["edge_to_idx"][frozenset((j, k))]
            edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index
