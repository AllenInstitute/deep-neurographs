"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

from collections import deque

import networkx as nx
import numpy as np
import torch

from deep_neurographs.utils import util

GNN_DEPTH = 2


# --- Batch Generation ---
def get_batch(graph, proposals, batch_size, flagged_proposals=set()):
    """
    Gets a batch for training that consist of a computation graph and list of
    proposals. Note: queue contains tuples that consist of a node id and
    distance from proposal.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that contains proposals to be classified.
    proposals : list
        Proposals to be classified as accept or reject.
    batch_size : int
        Maximum number of proposals in the computation graph.
    flagged_proposals : List[frozenset], optional
        List of proposals that are part of a large connected component in the
        proposal induced subgraph of "graph". The default is None

    Returns
    -------
    dict
        Batch that consists of set of proposals and the computation graph.

    """
    # Helpers
    def visit_proposal(p):
        batch["graph"].add_edge(i, j)
        batch["proposals"].add(p)
        proposals.remove(p)
        queue.append((j, 0))

    # Main
    batch = {"graph": nx.Graph(), "proposals": set()}
    visited = set()
    while len(proposals) > 0 and len(batch["proposals"]) < batch_size:
        root = tuple(util.sample_once(proposals))
        queue = deque([(root[0], 0), (root[1], 0)])
        while len(queue) > 0:
            # Visit node's nbhd
            i, d = queue.pop()
            for j in graph.neighbors(i):
                if (i, j) not in batch["graph"].edges:
                    batch["graph"].add_edge(i, j)
            visited.add(i)

            # Visit node's proposals
            for j in graph.nodes[i]["proposals"]:
                p = frozenset({i, j})
                if p in proposals and p in flagged_proposals:
                    for q in graph.extract_proposal_component(p):
                        q_0, q_1 = tuple(q)
                        if q_0 not in visited:
                            queue.append((q_0, 0))
                        if q_1 not in visited:
                            queue.append((q_1, 0))
                        visit_proposal(q)
                elif p in proposals:
                    visit_proposal(p)

            # Update queue
            if len(batch["proposals"]) < batch_size:
                for j in [j for j in graph.neighbors(i) if j not in visited]:
                    d_j = min(d + 1, -len(graph.nodes[j]["proposals"]))
                    if d_j <= GNN_DEPTH:
                        queue.append((j, d + 1))
    return batch


# --- Miscellaneous ---
def get_inputs(data, device="cpu"):
    """
    Extracts input data for a graph-based model and optionally moves it to a
    GPU.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        A data object with the following attributes:
            - x_dict: Dictionary of node features for different node types.
            - edge_index_dict: Dictionary of edge indices for edge types.
            - edge_attr_dict: Dictionary of edge attributes for edge types.
    device : str, optional
        Target device for the data, 'cuda' for GPU and 'cpu' for CPU. The
        default is "cpu".

    Returns
    --------
    tuple:
        Tuple containing the following:
            - x (dict): Node features dictionary.
            - edge_index (dict): Edge indices dictionary.
            - edge_attr (dict): Edge attributes dictionary.
    """
    data.to(device)
    return data.x_dict, data.edge_index_dict, data.edge_attr_dict


def line_graph(edges):
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


def load_model(model, model_path, device="cuda"):
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


def to_cpu(tensor, to_numpy=False):
    """
    Move PyTorch tensor to the CPU and optionally convert it to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be moved to CPU.
    to_numpy : bool, optional
        If True, converts the tensor to a NumPy array. Default is False.

    Returns
    -------
    torch.Tensor or np.ndarray
        Tensor or array on CPU.
    """
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()


def to_tensor(arr):
    """
    Converts a numpy array to a tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to tensor.
    """
    return torch.tensor(arr, dtype=torch.float32)
