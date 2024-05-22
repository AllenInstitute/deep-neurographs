"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training graph neural networks.

# explain branches vs edges terminology

"""

from random import sample

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData as HeteroGraphData

from deep_neurographs.machine_learning import feature_generation, gnn_utils

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
    x_nodes = feature_generation.combine_features(features["nodes"])

    # Initialize data
    proposals = list(features["proposals"]["skel"].keys())
    graph_dataset = HeteroGraphDataset(
        neurograph,
        proposals,
        x_nodes,
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
        x_nodes,
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
        x_nodes : numpy.ndarray
            Feature matrix generated from nodes in "neurograph".
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

        # Types
        self.node_types = ["branch", "proposal"]
        self.edge_types = [
            ("proposal", "edge", "proposal"),
            ("branch", "edge", "branch"),
            ("branch", "edge", "proposal"),
        ]

        # Features
        self.data = HeteroGraphData()
        self.data["branch"].x = torch.tensor(x_branches, dtype=DTYPE)
        self.data["proposal"].x = torch.tensor(x_proposals, dtype=DTYPE)
        self.data["proposal"].y = torch.tensor(y_proposals, dtype=DTYPE)

        # Edges
        self.init_edges(neurograph)
        self.init_edge_attrs(x_nodes)
        self.n_edge_attrs = n_edge_features(x_nodes)

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
        self.data["proposal", "edge", "proposal"].edge_index = proposal_edges
        self.data["branch", "edge", "branch"].edge_index = branch_edges
        self.data[
            "branch", "edge", "proposal"
        ].edge_index = branch_proposal_edges

    def init_edge_attrs(self, x_nodes):
        """
        Initializes edge attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Proposal edges
        edge_type = ("proposal", "edge", "proposal")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_proposals)

        # Branch edges
        edge_type = ("branch", "edge", "branch")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_branches)

        # Branch-Proposal edges
        edge_type = ("branch", "edge", "proposal")
        self.set_hetero_edge_attrs(
            x_nodes, edge_type, self.idxs_branches, self.idxs_proposals
        )

    # -- Getters --
    def n_branch_features(self):
        """
        Gets the dimension of feature vector for branches.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Dimension of feature vector for branches.

        """
        return self.data["branch"]["x"].size(1)

    def n_proposal_features(self):
        """
        Gets the dimension of feature vector for proposals.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Dimension of feature vector for proposals.

        """
        return self.data["proposal"]["x"].size(1)

    def n_edge_features(self):
        """
        Gets the dimension of feature vector for edges.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Dimension of feature vector for edges.

        """
        edge_type = ("proposal", "edge", "proposal")
        return self.data[edge_type]["x"].size(1)

    # -- Set Edges --
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
        line_graph = gnn_utils.init_line_graph(self.proposals)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_proposals["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return gnn_utils.to_tensor(edge_index)

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
        for e1, e2 in nx.line_graph(neurograph).edges:
            v1 = self.idxs_branches["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_branches["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return gnn_utils.to_tensor(edge_index)

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
        return gnn_utils.to_tensor(edge_index)

    # Set Edge Attributes
    def set_edge_attrs(self, x_nodes, edge_type, idx_map):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with the same type.

        Parameters
        ----------
        ...

        Returns
        -------
        None

        """
        attrs = []
        for i in range(self.data[edge_type].edge_index.size(1)):
            e1, e2 = self.data[edge_type].edge_index[:, i]
            v = node_intersection(idx_map, e1, e2)
            attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs), dtype=DTYPE)
        self.data[edge_type].x = arrs

    def set_hetero_edge_attrs(self, x_nodes, edge_type, idx_map_1, idx_map_2):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with different types.

        Parameters
        ----------
        ...

        Returns
        -------
        None
        """
        attrs = []
        for i in range(self.data[edge_type].edge_index.size(1)):
            e1, e2 = self.data[edge_type].edge_index[:, i]
            v = hetero_node_intersection(idx_map_1, idx_map_2, e1, e2)
            attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs), dtype=DTYPE)
        self.data[edge_type].x = arrs


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


def node_intersection(idx_map, e1, e2):
    """
    Computes the common node between "e1" and "e2" in the case where these
    edges connect nodes of the same type.

    Parameters
    ----------
    e1 : torch.Tensor
        Edge to be checked.
    e2 : torch.Tensor
        Edge to be checked.

    Returns
    -------
    int
        Common node between "e1" and "e2".
    """
    hat_e1 = idx_map["idx_to_edge"][int(e1)]
    hat_e2 = idx_map["idx_to_edge"][int(e2)]
    node = list(hat_e1.intersection(hat_e2))
    assert len(node) == 1, "Node intersection is not unique!"
    return node[0]


def hetero_node_intersection(idx_map_1, idx_map_2, e1, e2):
    """
    Computes the common node between "e1" and "e2" in the case where these
    edges connect nodes of different types.

    Parameters
    ----------
    e1 : torch.Tensor
        Edge to be checked.
    e2 : torch.Tensor
        Edge to be checked.

    Returns
    -------
    int
        Common node between "e1" and "e2".
    """
    hat_e1 = idx_map_1["idx_to_edge"][int(e1)]
    hat_e2 = idx_map_2["idx_to_edge"][int(e2)]
    node = list(hat_e1.intersection(hat_e2))
    assert len(node) == 1, "Node intersection is empty or not unique!"
    return node[0]


def n_edge_features(x):
    """
    Gets the number of edge features.

    Parameters
    ----------
    x : dict
        Dictionary that maps node (from a neurograph) to feature vectors.

    Returns
    -------
    int
        Number of edge features.

    """
    key = sample(list(x.keys()), 1)[0]
    return x[key].shape[0]
