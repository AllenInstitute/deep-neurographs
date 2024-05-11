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

from deep_neurographs.machine_learning import feature_generation


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

    # Initialize data
    proposals = features["proposals"]["skel"].keys()
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
            edge's position in "x_proposals".

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
        self.idxs_branches = init_idxs(idxs_branches)
        self.idxs_proposals = init_idxs(idxs_proposals)
        self.n_proposals = len(y_proposals)

        # Initialize data
        edge_index, proposal_edges = init_edge_index(
            neurograph, proposals, idxs_branches, idxs_proposals
        )
        self.data = GraphData(x=x, y=y, edge_index=edge_index)
        self.dropout_edges = proposal_edges


# -- utils --
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
    return shifted_idxs


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


def init_edge_index(neurograph, proposals, idxs_branches, idxs_proposals):
    """
    Initializes edge index for a graph dataset.

    Parameters
    ----------
    neurograph : neurograph.NeuroGraph
        Graph that represents a predicted segmentation.
    proposals : list
        List of edge proposals.
    idxs_branches : dict
        Dictionary that maps edges in "neurograph" to an index that represents
        an edge's position in "x_branches".
    idxs_proposals : dict
        Dictionary that maps "proposals" to an index that represents an edge's
        position in "x_proposals".

    Returns
    -------
    list
        List of edges in a graph dataset.
    list
        List of edges in a graph dataset that correspond to "proposals".

    """
    # Initializations
    branches_line_graph = nx.line_graph(neurograph)
    proposals_line_graph = init_line_graph(neurograph, proposals)
    proposal_edges = proposal_to_proposal(proposals_line_graph, idxs_proposals)

    # Compute edges
    edge_index = branch_to_branch(branches_line_graph, idxs_branches)
    edge_index.extend(proposal_edges)
    edge_index.extend(
        branch_to_proposal(
            neurograph, proposals, idxs_branches, idxs_proposals
        )
    )

    # Reshape
    edge_index = np.array(edge_index, dtype=np.int64).tolist()
    edge_index = torch.Tensor(edge_index).t().contiguous()
    return edge_index.long(), proposal_edges


def init_line_graph(neurograph, edges):
    """
    Initializes a line graph from a list of edges.

    Parameters
    ----------
    neurograph : neurograph.NeuroGraph
        Graph that represents a predicted segmentation.
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


def branch_to_branch(branches_line_graph, idxs_branches):
    """
    Generates edge indices between nodes corresponding to branches (i.e. edges
    in some neurograph).

    Parameters
    ----------
    branches_line_graph : networkx.Graph
        Line graph where each node represents an edge from some neurograph.
    idxs_branches : dict
        Dictionary that maps edges to vertices in "branches_line_graph".

    Returns
    -------
    list
        List of edges generated from "branches_line_graph" that are a subset
        of edges in some edge_index for a graph dataset.

    """
    edge_index = []
    for e1, e2 in branches_line_graph.edges:
        v1 = idxs_branches["edge_to_idx"][frozenset(e1)]
        v2 = idxs_branches["edge_to_idx"][frozenset(e2)]
        edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index


def proposal_to_proposal(proposals_line_graph, idxs_proposals):
    """
    Generates edge indices between nodes corresponding to proposals.

    Parameters
    ----------
    proposals_line_graph : networkx.Graph
        Line graph where each node represents a proposal from some neurograph.
    idxs_proposals : dict
        Dictionary that maps proposals to vertices in "proposals_line_graph".

    Returns
    -------
    list
        List of edges generated from "proposal_line_graph" that are a subset
        of edges in some edge_index for a graph dataset.

    """
    edge_index = []
    for e1, e2 in proposals_line_graph.edges:
        v1 = idxs_proposals["edge_to_idx"][frozenset(e1)]
        v2 = idxs_proposals["edge_to_idx"][frozenset(e2)]
        edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index


def branch_to_proposal(neurograph, proposals, idxs_branches, idxs_proposals):
    """
    Generates edge indices between nodes that correspond to proposals and
    edges.

    Parameters
    ----------
    neurograph : neurograph.NeuroGraph
        Graph that represents a predicted segmentation.
    proposals : list
        List of edge proposals.
    idxs_branches : dict
        Dictionary that maps edges to vertices in "branches_line_graph".
    idxs_proposals : dict
        Dictionary that maps proposals to vertices in "proposals_line_graph".

    Returns
    -------
    list
        List of edges generated from "proposal_line_graph" that are a subset
        of edges in some edge_index for a graph dataset.
    """
    edge_index = []
    for e in proposals:
        i, j = tuple(e)
        v1 = idxs_proposals["edge_to_idx"][frozenset(e)]
        for k in neurograph.neighbors(i):
            v2 = idxs_branches["edge_to_idx"][frozenset((i, k))]
            edge_index.extend([[v1, v2], [v2, v1]])
        for k in neurograph.neighbors(j):
            v2 = idxs_branches["edge_to_idx"][frozenset((j, k))]
            edge_index.extend([[v1, v2], [v2, v1]])
    return edge_index
