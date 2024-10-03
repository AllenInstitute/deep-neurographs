"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training machine learning models.

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
    return joblib.load(path) if ".joblib" in path else torch.load(path)


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


# --- dataset utils ---
def init_dataset(
    neurograph, features, model_type, computation_graph=None, sample_ids=None
):
    """
    Initializes a dataset given features generated from some set of proposals
    and neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "features" were generated from.
    features : dict
        Feaures generated from some set of proposals and "neurograph".
    model_type : str
        Type of machine learning model used to perform inference.
    computation_graph : networkx.Graph, optional
        Computation graph used by gnn if the "model_type" is either
        "GraphNeuralNet" or "HeteroGraphNeuralNet". The default is None.
    sample_ids : list[str], optional
        List of ids of samples if features were generated from distinct
        predictions. The default is None.

    Returns
    -------
    custom dataset type
        Dataset that stores features.

    """
    if model_type == "GraphNeuralNet":
        assert computation_graph is not None, "Must input computation graph!"
        dataset = heterograph_datasets.init(
            neurograph, features, computation_graph
        )
    else:
        dataset = datasets.init(
            neurograph, features, sample_ids=sample_ids
        )
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
