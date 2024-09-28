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

from deep_neurographs.machine_learning import datasets, feature_generation
from deep_neurographs.utils import gnn_util

DTYPE = torch.float32


# Wrapper
def init(neurograph, features, computation_graph):
    """
    Initializes a dataset that can be used to train a graph neural network.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that dataset is built from.
    features : dict
        Dictionary that contains different types of feature vectors for nodes,
        edges, and proposals.
    computation_graph : networkx.Graph
        Graph used by gnn to classify proposals.

    Returns
    -------
    HeteroGraphDataset
        Custom dataset.

    """
    # Extract features
    x_branches, _, idxs_branches = feature_generation.get_matrix(
        neurograph, features["edges"]
    )
    x_proposals, y_proposals, idxs_proposals = feature_generation.get_matrix(
        neurograph, features["proposals"]
    )
    x_nodes = feature_generation.combine_features(features["nodes"])

    # Initialize dataset
    proposals = list(features["proposals"]["skel"].keys())
    heterograph_dataset = HeteroGraphDataset(
        computation_graph,
        proposals,
        x_nodes,
        x_branches,
        x_proposals,
        y_proposals,
        idxs_branches,
        idxs_proposals,
    )
    return heterograph_dataset


# Datasets
class HeteroGraphDataset:
    """
    Custom dataset for heterogenous graphs.

    """

    def __init__(
        self,
        computation_graph,
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
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.
        proposals : list
            List of proposals to be classified.
        x_nodes : numpy.ndarray
            Feature matrix generated from nodes in "computation_graph".
        x_branches : numpy.ndarray
            Feature matrix generated from branches in "computation_graph".
        x_proposals : numpy.ndarray
            Feature matrix generated from "proposals" in "computation_graph".
        y_proposals : numpy.ndarray
            Ground truth of proposals.
        idxs_branches : dict
            Dictionary that maps edges in "computation_graph" to an index that
            represents the edge's position in "x_branches".
        idxs_proposals : dict
            Dictionary that maps "proposals" to an index that represents the
            edge's position in "x_proposals".

        Returns
        -------
        None

        """
        # Conversion idxs
        self.idxs_branches = datasets.init_idxs(idxs_branches)
        self.idxs_proposals = datasets.init_idxs(idxs_proposals)
        self.computation_graph = computation_graph
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
        self.init_edges()
        self.check_missing_edge_type()
        self.init_edge_attrs(x_nodes)
        self.n_edge_attrs = n_edge_features(x_nodes)

    def init_edges(self):
        """
        Initializes edge index for a graph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Compute edges
        proposal_edges = self.proposal_to_proposal()
        branch_edges = self.branch_to_branch()
        branch_proposal_edges = self.branch_to_proposal()

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

    def check_missing_edge_type(self):
        edge_type = ("branch", "edge", "branch")
        if len(self.data[edge_type].edge_index) == 0:
            # Add dummy features
            dtype = self.data["branch"].x.dtype
            zeros = torch.zeros(2, self.n_branch_features(), dtype=dtype)
            self.data["branch"].x = torch.cat(
                (self.data["branch"].x, zeros), dim=0
            )

            # Update edge_index
            n = self.data["branch"]["x"].size(0)
            edge_index = [[n - 1, n - 2], [n - 2, n - 1]]
            self.data[edge_type].edge_index = gnn_util.to_tensor(edge_index)
            self.idxs_branches["idx_to_edge"][n - 1] = frozenset({-1, -2})
            self.idxs_branches["idx_to_edge"][n - 2] = frozenset({-2, -3})

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
        line_graph = gnn_util.init_line_graph(self.proposals)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_proposals["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return gnn_util.to_tensor(edge_index)

    def branch_to_branch(self):
        """
        Generates edge indices between nodes corresponding to branches
        (i.e. edges in some neurograph).

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            Edges generated from "branches_line_graph" which are a subset of
            an edge_index for a graph dataset.

        """
        edge_index = []
        for e1, e2 in nx.line_graph(self.computation_graph).edges:
            e1_edge_bool = frozenset(e1) not in self.proposals
            e2_edge_bool = frozenset(e2) not in self.proposals
            if e1_edge_bool and e2_edge_bool:
                v1 = self.idxs_branches["edge_to_idx"][frozenset(e1)]
                v2 = self.idxs_branches["edge_to_idx"][frozenset(e2)]
                edge_index.extend([[v1, v2], [v2, v1]])
        return gnn_util.to_tensor(edge_index)

    def branch_to_proposal(self):
        """
        Generates edge indices between nodes that correspond to proposals and
        edges.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            Edges generated from proposals which are a subset of an edge_index
            for a graph dataset.

        """
        edge_index = []
        for p in self.proposals:
            i, j = tuple(p)
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(p)]
            for k in self.computation_graph.neighbors(i):
                if frozenset((i, k)) not in self.proposals:
                    v2 = self.idxs_branches["edge_to_idx"][frozenset((i, k))]
                    edge_index.extend([[v2, v1]])
            for k in self.computation_graph.neighbors(j):
                if frozenset((j, k)) not in self.proposals:
                    v2 = self.idxs_branches["edge_to_idx"][frozenset((j, k))]
                    edge_index.extend([[v2, v1]])
        return gnn_util.to_tensor(edge_index)

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
        if self.data[edge_type].edge_index.size(0) > 0:
            for i in range(self.data[edge_type].edge_index.size(1)):
                e1, e2 = self.data[edge_type].edge_index[:, i]
                v = node_intersection(idx_map, e1, e2)
                if v < 0:
                    attrs.append(torch.zeros(self.n_branch_features() + 1))
                else:
                    attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs), dtype=DTYPE)
        self.data[edge_type].edge_attr = arrs

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
        self.data[edge_type].edge_attr = arrs


# -- util --
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
