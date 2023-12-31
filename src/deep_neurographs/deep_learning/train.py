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
import torch.nn.functional as F
import torch.utils.data as torch_data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler
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
from deep_neurographs.deep_learning import models

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


BATCH_SIZE = 32
NUM_WORKERS = 0
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
    num_samples = int(np.floor(len(filenames) / k))
    assert num_samples > 0, "Sample size is too small for {}-folds".format(k)
    for i in range(k):
        samples_i = sample(samples, num_samples)
        samples = samples.difference(samples_i)
        folds.append(samples_i)
        if num_samples > len(samples):
            break
    return folds


# -- Training --
def fit_model(
    model_type, X, y, lr=1e-3, logger=False, max_epochs=50, profile=False
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
    profile : bool, optional
        Indication of whether to profile runtime of training neural network.
        The default is False.

    Returns
    -------
    ...
    """
    if model_type in ["FeedForwardNet", "ConvNet", "MultiModalNet"]:
        data = {"inputs": X, "labels": y}
        net, dataset = get_model(model_type, data=data)
        model = train_network(
            net, dataset, logger=logger, lr=lr, max_epochs=max_epochs
        )
    else:
        model = get_model(model_type)
        model.fit(X, y)
    return model


def evaluate_model():
    pass


def get_model(model_type, data=None):
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
        return RandomForestClassifier()
    elif model_type == "FeedForwardNet":
        n_features = extracter.count_features(model_type)
        net = models.FeedForwardNet(n_features)
        dataset = ds.ProposalDataset(data["inputs"], data["labels"])
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


def train_network(
    net, dataset, logger=False, lr=1e-3, max_epochs=50, profile=False
):
    # Load data
    train_set, valid_set = random_split(dataset)
    train_loader = DataLoader(
        train_set,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=SHUFFLE,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Configure trainer
    model = LitNeuralNet(net=net, lr=lr)
    ckpt_callback = ModelCheckpoint(save_top_k=1, monitor="val_f1", mode="max")
    profiler = PyTorchProfiler() if profile else None

    # Fit model
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[ckpt_callback],
        devices=1,
        enable_model_summary=True,
        enable_progress_bar=False,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        profiler=profiler,
    )
    trainer.fit(model, train_loader, valid_loader)

    # Return best model
    ckpt = torch.load(ckpt_callback.best_model_path)
    model.net.load_state_dict(ckpt["state_dict"])
    return model


def random_split(train_set, train_ratio=0.85):
    train_set_size = int(len(train_set) * train_ratio)
    valid_set_size = len(train_set) - train_set_size
    return torch_data.random_split(train_set, [train_set_size, valid_set_size])


def eval_network(X, model):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_pred = sigmoid(model.net(X))
    return np.array(y_pred)


# -- Lightning Module --
class LitNeuralNet(pl.LightningModule):
    def __init__(self, net=None, lr=1e-3):
        super().__init__()
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
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
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
