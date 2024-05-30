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
    hetero_graph_datasets,
)
from deep_neurographs.machine_learning.datasets import (
    ImgProposalDataset,
    MultiModalDataset,
    ProposalDataset,
)
from deep_neurographs.machine_learning.models import (
    ConvNet,
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
    elif model_type == "ConvNet":
        return ConvNet()
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


def get_dataset(inputs, targets, model_type, transform, lengths):
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
        return ProposalDataset(
            inputs, targets, transform=transform, lengths=lengths
        )
    elif model_type == "ConvNet":
        return ImgProposalDataset(inputs, targets, transform=transform)
    elif model_type == "MultiModalNet":
        return MultiModalDataset(inputs, targets, transform=transform)
    else:
        return {"inputs": inputs, "targets": targets}


def init_dataset(
    neurographs, features, model_type, block_ids=None, transform=False
):
    if "Hetero" in model_type:
        dataset = hetero_graph_datasets.init(neurographs, features)
    elif "Graph" in model_type:
        dataset = graph_datasets.init(neurographs, features)
    else:
        dataset = init_proposal_dataset(
            neurographs,
            features,
            model_type,
            block_ids=block_ids,
            transform=transform,
        )
    return dataset


def init_proposal_dataset(
    neurographs, features, model_type, block_ids=None, transform=False
):
    # Extract features
    inputs, targets, idx_transforms = feature_generation.get_matrix(
        neurographs, features, model_type, block_ids=block_ids
    )
    lens = []
    if transform:
        for block_id in block_ids:
            lens.extend(get_lengths(neurographs[block_id]))

    dataset = {
        "dataset": get_dataset(inputs, targets, model_type, transform, lens),
        "block_to_idxs": idx_transforms["block_to_idxs"],
        "idx_to_edge": idx_transforms["idx_to_edge"],
    }
    return dataset


def get_lengths(neurograph):
    lengths = []
    for edge in neurograph.proposals.keys():
        lengths.append(neurograph.proposal_length(edge))
    return lengths


def sigmoid(x):
    """
    Sigmoid function.

    Parameters
    ----------
    x : numpy.ndarray
        Input to sigmoid.

    Return
    ------
    Sigmoid applied to "x".

    """
    return 1.0 / (1.0 + np.exp(-x))
