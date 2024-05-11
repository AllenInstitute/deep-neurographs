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
from torch_geometric.data import HeteroData as HeteroGraphData
from deep_neurographs.machine_learning import feature_generation

DTYPE = torch.float32


# Wrapper
def init(neurograph, features):
    """
    Initializes a dataset that can be used to train a graph neural network.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that dataset is built from.
    features : dict
        Feature vectors corresponding to branches such that the keys are
        "proposals" and "branches". The values are a dictionary containing
        different types of features for edges and branches.

    Returns
    -------
    GraphDataset
        Custom dataset.

    """
    # Extract features
    x_branches, _, idxs_branches = feature_generation.get_matrix(
        neurograph, features["branches"], "GraphNeuralNet"
    )
    x_proposals, y_proposals, idxs_proposals = feature_generation.get_matrix(
        neurograph, features["proposals"], "GraphNeuralNet"
    )
    # get node feature matrix

    # Initialize data
    proposals = list(features["proposals"]["skel"].keys())
    graph_dataset = HeteroGraphDataset(
        neurograph,
        proposals,
        x_branches,
        x_proposals,
        y_proposals,
        idxs_branches,
        idxs_proposals,
    )
    return graph_dataset


# Datasets
class HeteroGraphDataset:
    """
    Custom dataset for heterogenous graphs.

    """

    def __init__(
        self,
        neurograph,
        proposals,
        x_branches,
        x_proposals,
        y_proposals,
        idxs_branches,
        idxs_proposals,
    ):
        """
        Constructs a HeteroGraphDataset object.

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.
        proposals : list
            List of edge proposals.
        x_branches : numpy.ndarray
            Feature matrix generated from branches in "neurograph".
        x_proposals : numpy.ndarray
            Feature matrix generated from "proposals" in "neurograph".
        y_proposals : numpy.ndarray
            Ground truth of proposals.
        idxs_branches : dict
            Dictionary that maps edges in "neurograph" to an index that
            represents the edge's position in "x_branches".
        idxs_proposals : dict
            Dictionary that maps "proposals" to an index that represents the
            edge's position in "x_proposals".

        Returns
        -------
        None

        """
        # Conversion idxs
        self.idxs_branches = init_idxs(idxs_branches)
        self.idxs_proposals = init_idxs(idxs_proposals)
        self.proposals = proposals

        # Features
        self.data = HeteroGraphData()
        self.data["branch"].x = torch.tensor(x_branches, dtype=DTYPE)
        self.data["proposal"].x = torch.tensor(x_proposals, dtype=DTYPE)
        self.data["proposal"].y =  torch.tensor(y_proposals, dtype=DTYPE)

        # Edges
        self.init_edges(neurograph)

    def init_edges(self, neurograph):
        """
        Initializes edge index for a graph dataset.

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.

        Returns
        -------
        None

        """
        # Compute edges
        proposal_edges = self.proposal_to_proposal()
        branch_edges = self.branch_to_branch(neurograph)
        branch_proposal_edges = self.branch_to_proposal(neurograph)

        # Store edges
        self.data["proposal", "to", "proposal"] = proposal_edges
        self.data["branch", "to", "branch"] = branch_edges
        self.data["branch", "to", "proposal"] = branch_proposal_edges

    def proposal_to_proposal(self):
        """
        Generates edge indices between nodes corresponding to proposals.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Edges generated from line_graph generated from proposals which are
            part of an edge_index for a graph dataset.

        """
        edge_index = []
        line_graph = init_line_graph(self.proposals)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_proposals["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return to_tensor(edge_index)

    def branch_to_branch(self, neurograph):
        """
        Generates edge indices between nodes corresponding to branches
        (i.e. edges in some neurograph).

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.

        Returns
        -------
        torch.Tensor
            Edges generated from "branches_line_graph" which are a subset of
            an edge_index for a graph dataset.

        """
        edge_index = []
        line_graph = nx.line_graph(neurograph)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_branches["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_branches["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return to_tensor(edge_index)

    def branch_to_proposal(self, neurograph):
        """
        Generates edge indices between nodes that correspond to proposals and
        edges.

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.

        Returns
        -------
        torch.Tensor
            Edges generated from proposals which are a subset of an edge_index
            for a graph dataset.

        """
        edge_index = []
        for e in self.proposals:
            i, j = tuple(e)
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e)]
            for k in neurograph.neighbors(i):
                v2 = self.idxs_branches["edge_to_idx"][frozenset((i, k))]
                edge_index.extend([[v2, v1]])
            for k in neurograph.neighbors(j):
                v2 = self.idxs_branches["edge_to_idx"][frozenset((j, k))]
                edge_index.extend([[v2, v1]])
        return to_tensor(edge_index)


# -- utils --
def init_idxs(idxs):
    """
    Adds dictionary item called "edge_to_index" which maps an edge in a
    neurograph to an that represents the edge's position in the feature
    matrix.

    Parameters
    ----------
    idxs : dict
        Dictionary that maps indices to edges in some neurograph.

    Returns
    -------
    dict
        Updated dictionary.

    """
    idxs["edge_to_idx"] = dict()
    for idx, edge in idxs["idx_to_edge"].items():
        idxs["edge_to_idx"][edge] = idx
    return idxs


def init_line_graph(edges):
    """
    Initializes a line graph from a list of edges.

    Parameters
    ----------
    edges : list
        List of edges.

    Returns
    -------
    networkx.Graph
        Line graph generated from a list of edges.

    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.line_graph(graph)


def to_tensor(my_list):
    """
    Converts a list to a tensor with contiguous memory.

    Parameters
    ----------
    my_list : list
        List to be converted into a tensor.

    Returns
    -------
    torch.Tensor
        Tensor.

    """
    arr = np.array(my_list, dtype=np.int64).tolist()
    return torch.Tensor(arr).t().contiguous().long()
