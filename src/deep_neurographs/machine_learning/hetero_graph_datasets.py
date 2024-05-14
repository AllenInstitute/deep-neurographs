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
from random import sample
from torch_geometric.data import HeteroData as HeteroGraphData

from deep_neurographs.machine_learning import feature_generation

DEVICE = "cuda:0"
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
            ("proposal", "to", "proposal"),
            ("branch", "to", "branch"),
            ("branch", "to", "proposal")
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
        self.data["proposal", "to", "proposal"].edge_index = proposal_edges
        self.data["branch", "to", "branch"].edge_index = branch_edges
        self.data["branch", "to", "proposal"].edge_index = branch_proposal_edges

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
        edge_type = ("proposal", "to", "proposal")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_proposals)

        # Branch edges
        edge_type = ("branch", "to", "branch")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_branches)

        # Branch-Proposal edges
        edge_type = ("branch", "to", "proposal")

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

    def toGPU(self):
        """
        Moves "data" from CPU to GPU.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Node features
        for n_type in self.node_types:
            self.data[n_type]["x"] = self.data[n_type]["x"].to(DEVICE, dtype=DTYPE)

        # Edge features
        for e_type in self.edge_types:
            if e_type != ("branch", "to", "proposal"):
                self.data[e_type]["x"] = self.data[e_type]["x"].to(DEVICE, dtype=DTYPE)

        # Labels
        self.data[n_type]["y"] = self.data[n_type]["y"].to(DEVICE, dtype=DTYPE)

    def toCPU(self):
        """
        Moves "data" from GPU to CPU.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Node features
        for n_type in self.node_types:
            self.data[n_type] = self.data[n_type]["x"].detach().cpu()

        # Edge features
        for e_type in self.edge_types:
            self.data[e_type] = self.data[e_type]["x"].detach().cpu()

    # Set Edge Attributes
    def set_edge_attrs(self, x_nodes, edge_type, idx_mapping):
        """
        Generate proposal edge attributes

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        attrs = []
        for i in range(self.data[edge_type].edge_index.size(1)):
            e1, e2 = self.data[edge_type].edge_index[:, i]
            v = node_intersection(idx_mapping, e1, e2)
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


def node_intersection(idx_mapping, e1, e2):
    """
    Computes the common node between "e1" and "e2".

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
    hat_e1 = idx_mapping["idx_to_edge"][int(e1)]
    hat_e2 = idx_mapping["idx_to_edge"][int(e2)]
    node = list(hat_e1.intersection(hat_e2))
    assert len(node) == 1, "Node intersection is not unique!"
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
