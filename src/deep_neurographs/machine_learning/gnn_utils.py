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


def toGPU(data, is_dict=False):
    """
    Moves "data" from CPU to GPU.

    Parameters
    ----------
    data : GraphDataset
        Dataset to be moved to GPU.
    is_dict : bool, optional
        Indication of whether tensor is a dictionary containing tensors. The
        default is False.

    Returns
    -------
    torch.Tensor
        Feature vectors on GPU
    torch.Tensor
        Edge indices on GPU.
    torch.Tensor, optional
        Edge features on GPU.

    """
    if is_dict:
        x = toGPU_dict(data.x_dict)
        edge_index = toGPU_dict(data.edge_index_dict)
        edge_attrs = toGPU_dict(data.edge_attrs)
        return x, edge_index, edge_attrs
    else:
        x = data.x  # .to("cuda:0", dtype=torch.float32)
        edge_index = data.edge_index  # .to("cuda:0")
        return x, edge_index


def toGPU_dict(data_dict):
    """
    Moves a dictionary containing tensors to from the CPU to GPU.

    Parameters
    ----------
    data_dict : dict
        Dictionary to be moved to GPU.

    Returns
    -------
    dict
        Dictionary containing tensors on GPU.

    """
    for key in data_dict.keys():
        data_dict[key] = data_dict[key]  # .cuda()
    return data_dict


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
