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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from deep_neurographs.machine_learning import (
    feature_generation,
    graph_datasets,
    heterograph_datasets,
)
from deep_neurographs.machine_learning.datasets import (
    MultiModalDataset,
    ProposalDataset,
)
from deep_neurographs.machine_learning.models import (
    FeedForwardNet,
    MultiModalNet,
)

SUPPORTED_MODELS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
    "GraphNeuralNet",
]


# --- model utils ---
def init_model(model_type):
    """
    Initializes a machine learning model.

    Parameters
    ----------
    model_type : str
        Type of machine learning model.

    Returns
    -------
    ...

    """
    assert model_type in SUPPORTED_MODELS, f"{model_type} not supported!"
    if model_type == "AdaBoost":
        return AdaBoostClassifier()
    elif model_type == "RandomForest":
        return RandomForestClassifier()
    elif model_type == "FeedForwardNet":
        n_features = feature_generation.count_features(model_type)
        return FeedForwardNet(n_features)
    elif model_type == "MultiModalNet":
        n_features = feature_generation.count_features(model_type)
        return MultiModalNet(n_features)


def load_model(model_type, path):
    """
    Loads the parameters of a machine learning model.

    Parameters
    ----------
    model_type : str
        Type of machine learning model.
    path : str
        Path to the model parameters.

    Returns
    -------
    ...
    """
    if model_type in ["AdaBoost", "RandomForest"]:
        return joblib.load(path)
    else:
        return torch.load(path)


# --- dataset utils ---
def init_dataset(
    neurograph,
    features,
    model_type,
    computation_graph=None,
    sample_ids=None,
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
    if "Hetero" in model_type:
        assert computation_graph, "Must provide computation graph!"
        dataset = heterograph_datasets.init(
            neurograph, computation_graph, features
        )
    elif "Graph" in model_type:
        dataset = graph_datasets.init(neurograph, features)
    else:
        dataset = init_proposal_dataset(
            neurograph,
            features,
            model_type,
            sample_ids=sample_ids,
        )
    return dataset


def init_proposal_dataset(
    neurographs, features, model_type, sample_ids=None,
):
    # Extract features
    inputs, targets, idx_transforms = feature_generation.get_matrix(
        neurographs, features, model_type, sample_ids=sample_ids
    )
    dataset = {
        "dataset": get_dataset(inputs, targets, model_type),
        "block_to_idxs": idx_transforms["block_to_idxs"],
        "idx_to_edge": idx_transforms["idx_to_edge"],
    }
    return dataset


def get_dataset(inputs, targets, model_type):
    """
    Gets classification model to be fit.

    Parameters
    ----------
    inputs : ...
        ...
    targets : ...
        ...
    model_type : str
        Type of machine learning model, see "SUPPORTED_MODEL_TYPES" for
        options.
    transform : bool
        Indication of whether to apply data augmentation

    Returns
    -------
    ...

    """
    if model_type == "FeedForwardNet":
        dataset = ProposalDataset(inputs, targets)
    elif model_type == "MultiModalNet":
        dataset = MultiModalDataset(inputs, targets)
    else:
        dataset = {"inputs": inputs, "targets": targets}
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
    folds = []
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


def get_batches(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start: min(start + batch_size, len(iterable))]
