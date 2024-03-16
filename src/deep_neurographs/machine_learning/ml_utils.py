"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training machine learning models.

"""

import numpy as np
from random import sample
from deep_neurographs.machine_learning.models import ConvNet, FeedForwardNet, MultiModalNet
from deep_neurographs import feature_extraction as extracter

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


def get_model_type(model):
    # Set model_type
    assert model in SUPPORTED_MODELS, "Model not supported!"
    if type(model) == FeedForwardNet:
        return "FeedForwardNet"
    elif type(model) == ConvNet:
        return "ConvNet"
    elif type(model) == MultiModalNet:
        return "MultiModalNet"
    else:
        print("Input model instead of model_type")


def init_model(model_type):
    assert model_type in SUPPORTED_MODELS, "Model not supported!"
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


def init_dataloader(model_type, augmentation=False):
    """
    Gets classification model to be fit.

    Parameters
    ----------
    model_type : str
        Indication of type of model. Options are "AdaBoost",
        "RandomForest", "FeedForwardNet", "ConvNet", and
        "MultiModalNet".
    data : dict, optional
        Training data used to fit model. This dictionary must contain the keys
        "inputs" and "labels" which correspond to the feature matrix and
        target labels to be learned. The default is None.

    Returns
    -------
    ...

    """
    if model_type == "FeedForwardNet":
        dataset = ds.ProposalDataset(data["inputs"], data["labels"], transform=augmentation)
    elif model_type == "ConvNet":
        dataset = ds.ImgProposalDataset(
            data["inputs"], data["labels"], transform=True
        )
    elif model_type == "MultiModalNet":
        models.init_weights(net)
        dataset = ds.MultiModalDataset(
            data["inputs"], data["labels"], transform=True
        )
    return net, dataset
