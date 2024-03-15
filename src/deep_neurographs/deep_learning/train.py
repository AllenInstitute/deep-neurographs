"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training models that classify edge proposals.

"""

import logging
from random import sample

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    binary_f1_score,
    binary_precision,
    binary_recall,
)

from deep_neurographs import feature_extraction as extracter
from deep_neurographs.deep_learning import datasets as ds
from deep_neurographs.deep_learning import loss, models
#from deep_neurographs.deep_learning.datasets import ConvNet, FeedForwardNet, MultiModalNet

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

BATCH_SIZE = 32
SHUFFLE = True
SUPPORTED_MODELS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
]


# -- Cross Validation --
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


# -- Training --
def run_on_blocks(neurographs, features, dataset, model, block_ids=None):
    # Set model_type
    if type(model) == FeedForwardNet:
        model_type = "FeedForwardNet"
    elif type(model) == ConvNet:
        model_type = "ConvNet"
    elif type(model) == MultiModalNet:
        model_type = "MultiModalNet"
    else:
        print("Input model instead of model_type")

    # Initialize data
    X_train, y_train, _, _ = extracter.get_feature_matrix(
        neurographs,
        features,
        model_type,
        block_ids=train_blocks,
    )


def fit_model(
    model_type, X, y, augmentation=False, lr=1e-3, logger=False, max_epochs=50, n_estimators=100
):
    """
    Fits a model to a training dataset.

    Parameters
    ----------
    model_type : str
        Indication of type of model. Options are "AdaBoost",
        "RandomForest", "FeedForwardNet", "ConvNet", and
        "MultiModalNet".
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Labels to be learned.
    lr : float, optional
        Learning rate to be used if model is a neural network. The default is
        1e-3.
    logger : bool, optional
        Indication of whether to log performance stats while neural network
        trains. The default is False.
    max_epochs : int, optional
        Maximum number of epochs used to train neural network. The default is
        50.

    Returns
    -------
    ...
    """
    if model_type in ["FeedForwardNet", "ConvNet", "MultiModalNet"]:
        data = {"inputs": X, "labels": y}
        net, dataset = get_model(model_type, augmentation=augmentation, data=data)
        model = fit_network(
            net, dataset, logger=logger, lr=lr, max_epochs=max_epochs
        )
    else:
        model = get_model(model_type, n_estimators=n_estimators)
        model.fit(X, y)
    return model


def get_model(model_type, augmentation=False, data=None, n_estimators=100):
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
    assert model_type in SUPPORTED_MODELS
    if model_type == "AdaBoost":
        return AdaBoostClassifier()
    elif model_type == "RandomForest":
        return RandomForestClassifier(n_estimators=n_estimators)
    elif model_type == "FeedForwardNet":
        n_features = extracter.count_features(model_type)
        net = models.FeedForwardNet(n_features)
        dataset = ds.ProposalDataset(data["inputs"], data["labels"], transform=augmentation)
    elif model_type == "ConvNet":
        net = models.ConvNet()
        models.init_weights(net)
        dataset = ds.ImgProposalDataset(
            data["inputs"], data["labels"], transform=True
        )
    elif model_type == "MultiModalNet":
        n_features = extracter.count_features(model_type)
        net = models.MultiModalNet(n_features)
        models.init_weights(net)
        dataset = ds.MultiModalDataset(
            data["inputs"], data["labels"], transform=True
        )
    return net, dataset


def fit_network(
    net, dataset, logger=False, lr=1e-3, max_epochs=50
):
    # Load data
    train_set, valid_set = random_split(dataset)
    train_loader = DataLoader(train_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)

    # Configure trainer
    model = LitNeuralNet(net=net, lr=lr)
    ckpt_callback = ModelCheckpoint(save_top_k=1, monitor="val_f1", mode="max")

    # Fit model
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[ckpt_callback],
        devices=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=max_epochs,
    )
    trainer.fit(model, train_loader, valid_loader)

    # Return best model
    ckpt = torch.load(ckpt_callback.best_model_path)
    model.net.load_state_dict(ckpt["state_dict"])
    return model


def random_split(train_set, train_ratio=0.8):
    train_set_size = int(len(train_set) * train_ratio)
    valid_set_size = len(train_set) - train_set_size
    return torch_data.random_split(train_set, [train_set_size, valid_set_size])


def eval_network(X, model):
    # Prep data
    if type(X) == dict:
        X = [
            torch.tensor(X["features"], dtype=torch.float32),
            torch.tensor(X["imgs"], dtype=torch.float32),
        ]
    else:
        X = torch.tensor(X, dtype=torch.float32)

    # Run model
    model.eval()
    with torch.no_grad():
        y_pred = sigmoid(model.net(X))
    return np.array(y_pred)


# -- Lightning Module --
class LitNeuralNet(pl.LightningModule):
    def __init__(self, net=None, lr=1e-3):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.net = net
        self.lr = lr

    def forward(self, batch):
        x = self.get_example(batch, "inputs")
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "labels")
        y_hat = self.net(X)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.compute_stats(y_hat, y, prefix="train_")
        return loss

    def validation_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "labels")
        y_hat = self.net(X)
        self.compute_stats(y_hat, y, prefix="val_")

    def compute_stats(self, y_hat, y, prefix=""):
        y_hat = torch.flatten(sigmoid(y_hat))
        y = torch.flatten(y).to(torch.int)
        self.log(prefix + "precision", binary_precision(y_hat, y))
        self.log(prefix + "recall", binary_recall(y_hat, y))
        self.log(prefix + "f1", binary_f1_score(y_hat, y))

    def get_example(self, batch, key):
        return batch[key]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.net.state_dict(destination, prefix + "", keep_vars)
