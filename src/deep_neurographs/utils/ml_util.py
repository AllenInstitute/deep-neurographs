"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

from random import sample

import joblib
import numpy as np
import torch

from deep_neurographs.machine_learning import datasets, heterograph_datasets

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
    if ".joblib" in path:
        model = joblib.load(path)
    else:
        model = torch.load(path)
        model.eval()
    return model


def save_model(path, model, model_type):
    """
    Saves a machine learning model.

    Parameters
    ----------
    path : str
        Path that model parameters will be written to.
    model : object
        Model to be saved.

    Returns
    -------
    None

    """
    print("Model saved!")
    if "Net" in model_type:
        torch.save(model, path)
    else:
        joblib.dump(model, path)


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


# --- dataset utils ---
def init_dataset(
    fragments_graph,
    features,
    is_gnn=True,
    is_multimodal=False,
    computation_graph=None
):
    """
    Initializes a dataset given features generated from some set of proposals
    and fragments_graph.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that "features" were generated from.
    features : dict
        Feaures generated from some set of proposals and "fragments_graph".
    model_type : str
        Type of machine learning model used to perform inference.
    computation_graph : networkx.Graph, optional
        Computation graph used by gnn if the "model_type" is either
        "GraphNeuralNet" or "HeteroGraphNeuralNet". The default is None.

    Returns
    -------
    custom dataset type
        Dataset that stores features.

    """
    if is_gnn:
        assert computation_graph is not None, "Must input computation graph!"
        dataset = heterograph_datasets.init(
            fragments_graph, features, computation_graph
        )
    else:
        dataset = datasets.init(fragments_graph, features)
    return dataset


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
