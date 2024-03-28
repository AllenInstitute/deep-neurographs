"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training models that classify edge proposals.

"""

import logging

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    binary_f1_score,
    binary_precision,
    binary_recall,
)

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


def fit_model(model, dataset):
    inputs = dataset["dataset"]["inputs"]
    targets = dataset["dataset"]["targets"]
    model.fit(inputs, targets)
    return model


def fit_deep_model(
    model,
    dataset,
    batch_size=BATCH_SIZE,
    criterion=None,
    logger=False,
    lr=1e-3,
    max_epochs=1000,
):
    """
    Fits a neural network to a dataset.

    Parameters
    ----------
    model : ...
        ...
    dataset : ...
        ...
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
    # Load data
    dataset = dataset["dataset"]
    train_set, valid_set = random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # Configure trainer
    lit_model = LitModel(criterion=criterion, model=model, lr=lr)
    ckpt_callback = ModelCheckpoint(save_top_k=1, monitor="val_f1", mode="max")

    # Fit model
    pylightning_trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[ckpt_callback],
        devices=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=max_epochs,
    )
    pylightning_trainer.fit(lit_model, train_loader, valid_loader)

    # Return best model
    print(ckpt_callback.best_model_path)
    ckpt = torch.load(ckpt_callback.best_model_path)
    lit_model.model.load_state_dict(ckpt["state_dict"])
    return lit_model.model


def random_split(train_set, train_ratio=0.8):
    train_set_size = int(len(train_set) * train_ratio)
    valid_set_size = len(train_set) - train_set_size
    return torch_data.random_split(train_set, [train_set_size, valid_set_size])


# -- Lightning Module --
class LitModel(pl.LightningModule):
    def __init__(self, criterion=None, model=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        if criterion:
            self.criterion = criterion
        else:
            pos_weight = torch.tensor([0.75], device=0)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, batch):
        x = self.get_example(batch, "inputs")
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "targets")
        y_hat = self.model(X)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.compute_stats(y_hat, y, prefix="train_")
        return loss

    def validation_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "targets")
        y_hat = self.model(X)
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
        return self.model.state_dict(destination, prefix + "", keep_vars)
