"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training graph neural networks.

NOTE: THIS SCRIPT IS NO LONGER USED!

"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data as GraphData

from deep_neurographs.machine_learning import datasets, feature_generation
from deep_neurographs.utils import gnn_util


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

    # Initialize dataset
    proposals = list(features["proposals"]["skel"].keys())
    graph_dataset = GraphDataset(
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
class GraphDataset:
    """
    Custom dataset for homogenous graphs.

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
        Constructs a GraphDataset object.

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
            proposals's position in "x_proposals".

        Returns
        -------
        None

        """
        # Combine feature matrices
        x = np.vstack([x_proposals, x_branches]).astype(np.float32)
        x = torch.tensor(x)
        y = torch.tensor(y_proposals.astype(np.float32))

        # Set edges
        idxs_branches = shift_idxs(idxs_branches, x_proposals.shape[0])
        self.idxs_branches = datasets.init_idxs(idxs_branches)
        self.idxs_proposals = datasets.init_idxs(idxs_proposals)
        self.proposals = proposals

        # Initialize data
        edge_index = self.init_edge_index(neurograph)
        self.data = GraphData(x=x, y=y, edge_index=edge_index)

    def init_edge_index(self, neurograph):
        """
        Initializes edge index for a graph dataset.

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.

        Returns
        -------
        list
            List of edges in a graph dataset.

        """
        # Compute edges
        branch_edges = self.branch_to_branch(neurograph)
        branch_proposal_edges = self.branch_to_proposal(neurograph)
        proposal_edges = self.proposal_to_proposal()

        # Compile edge_index
        self.dropout_edges = proposal_edges
        edge_index = branch_proposal_edges
        edge_index.extend(branch_edges)
        edge_index.extend(proposal_edges)
        return gnn_util.to_tensor(edge_index)

    def proposal_to_proposal(self):
        """
        Generates edge indices between nodes corresponding to proposals.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Edges generated from proposal line graph that are a subset of
            edges in edge_index.

        """
        edge_index = []
        line_graph = gnn_util.init_line_graph(self.proposals)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_proposals["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return edge_index

    def branch_to_branch(self, neurograph):
        """
        Generates edge indices between nodes corresponding to branches
        (i.e. edges in neurograph).

        Parameters
        ----------
        neurograph : neurograph.NeuroGraph
            Graph that represents a predicted segmentation.

        Returns
        -------
        list
            Edges generated from "branches_line_graph" that are a subset of
            edges in edge_index.

        """
        edge_index = []
        for e1, e2 in nx.line_graph(neurograph).edges:
            v1 = self.idxs_branches["edge_to_idx"][frozenset(e1)]
            v2 = self.idxs_branches["edge_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return edge_index

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
        list
            List of edges between proposals and branches.

        """
        edge_index = []
        for e in self.proposals:
            i, j = tuple(e)
            v1 = self.idxs_proposals["edge_to_idx"][frozenset(e)]
            for k in neurograph.neighbors(i):
                v2 = self.idxs_branches["edge_to_idx"][frozenset((i, k))]
                edge_index.extend([[v1, v2], [v2, v1]])
            for k in neurograph.neighbors(j):
                v2 = self.idxs_branches["edge_to_idx"][frozenset((j, k))]
                edge_index.extend([[v1, v2], [v2, v1]])
        return edge_index


# -- util --
def shift_idxs(idxs, shift):
    """
    Shifts every key in "idxs["idx_to_edge"]" by "shift".

    Parameters
    ----------
    idxs : dict
        Dictionary that maps indices to edges in some neurograph.
    shift : int
        Magnitude of shift.

    Returns
    -------
    dict
        Updated dictinoary where keys are shifted by value "shift".

    """
    shifted_idxs = dict()
    for key, value in idxs["idx_to_edge"].items():
        shifted_idxs[key + shift] = value
    idxs["idx_to_edge"] = shifted_idxs
    return idxs
