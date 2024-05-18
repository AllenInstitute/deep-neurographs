"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training graph neural networks.

"""

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

    """
    if is_dict:
        x = toGPU_dict(data.x_dict)
        edge_index = toGPU_dict(data.edge_index_dict)
    else:
        x = data.x.to("cuda:0", dtype=torch.float32)
        edge_index = data.edge_index.to("cuda:0")
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
