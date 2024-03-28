"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training machine learning models.

"""

from random import sample

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from deep_neurographs.machine_learning import feature_extraction as extracter
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
    assert model_type in SUPPORTED_MODELS, f"{model_type} not supported!"
    if model_type == "AdaBoost":
        return AdaBoostClassifier()
    elif model_type == "RandomForest":
        return RandomForestClassifier()
    elif model_type == "FeedForwardNet":
        n_features = extracter.count_features(model_type)
        return FeedForwardNet(n_features)
    elif model_type == "ConvNet":
        return ConvNet()
    elif model_type == "MultiModalNet":
        n_features = extracter.count_features(model_type)
        return MultiModalNet(n_features)


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
    neurographs, features, model_type, block_ids, transform=False
):
    # Extract features
    inputs, targets, block_to_idx, idx_to_edge = extracter.get_feature_matrix(
        neurographs, features, model_type, block_ids=block_ids
    )
    lens = []
    if transform:
        for block_id in block_ids:
            lens.extend(get_lengths(neurographs[block_id]))

    dataset = {
        "dataset": get_dataset(inputs, targets, model_type, transform, lens),
        "block_to_idxs": block_to_idx,
        "idx_to_edge": idx_to_edge,
    }
    return dataset


def get_lengths(neurograph):
    lengths = []
    for edge in neurograph.proposals.keys():
        lengths.append(neurograph.proposal_length(edge))
    return lengths
