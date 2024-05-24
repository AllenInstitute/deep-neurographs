"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training graph neural networks.

"""

import networkx as nx
import numpy as np
import torch


def toCPU(tensor):
    """
    Moves tensor from GPU to CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor.

    Returns
    -------
    None

    """
    return tensor.detach().cpu().tolist()


def get_inputs(data, model_type):
    if "Hetero" in model_type:
        x = data.x_dict
        edge_index = data.edge_index_dict
        #edge_attr_dict = data.edge_attr_dict
        return x, edge_index #, edge_attr_dict
    else:
        x = data.x
        edge_index = data.edge_index
        return x, edge_index


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
