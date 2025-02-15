"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

from random import sample

import numpy as np
import torch

SUPPORTED_MODELS = ["RandomForest", "GraphNeuralNet"]


# --- model utils ---
def load_model(path):
    """
    Loads the parameters of a machine learning model.

    Parameters
    ----------
    path : str
        Path to the model parameters.

    Returns
    -------
    ...

    """
    model = torch.load(path, weights_only=False)
    model.eval()
    return model


def toGPU(tensor_dict):
    """
    Moves dictionary of tensors from CPU to GPU.

    Parameters
    ----------
    tensor_dict : dict
        Tensor to be moved to GPU.

    Returns
    -------
    None

    """
    return {k: tensor.to("cuda") for k, tensor in tensor_dict.items()}


def toCPU(tensor):
    """
    Moves tensor from GPU to CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be moved to CPU.

    Returns
    -------
    list
        Tensor moved to CPU and converted into a list.

    """
    return tensor.detach().cpu().tolist()


# --- miscellaneous ---
def sigmoid(x):
    """
    Sigmoid function.

    Parameters
    ----------
    x : numpy.ndarray
        Input to sigmoid.

    Return
    ------
    numpy.ndarray
        Sigmoid applied to "x".

    """
    return 1.0 / (1.0 + np.exp(-x))


def get_kfolds(filenames, k):
    """
    Partitions "filenames" into k-folds to perform cross validation.

    Parameters
    ----------
    filenames : list[str]
        List of filenames of samples for training.
    k : int
        Number of folds to be used in k-fold cross validation.

    Returns
    -------
    folds : list[list[str]]
        Partition of "filesnames" into k-folds.

    """
    folds = list()
    samples = set(filenames)
    n_samples = int(np.floor(len(filenames) / k))
    assert n_samples > 0, "Sample size is too small for {}-folds".format(k)
    for i in range(k):
        samples_i = sample(samples, n_samples)
        samples = samples.difference(samples_i)
        folds.append(samples_i)
        if n_samples > len(samples):
            break
    return folds


def toTensor(my_list):
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
