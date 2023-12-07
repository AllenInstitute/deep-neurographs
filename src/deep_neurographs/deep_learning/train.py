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

from deep_neurographs.deep_learning import datasets as ds
from deep_neurographs.deep_learning import models

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


BATCH_SIZE = 32
NUM_WORKERS = 0
SHUFFLE = True
SUPPORTED_CLFS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
]


# Training
def get_kfolds(train_data, k):
    folds = []
    samples = set(train_data)
    num_samples = int(np.floor(len(train_data) / k))
    assert num_samples > 0, "Sample size is too small for {}-folds".format(k)
    for i in range(k):
        samples_i = sample(samples, num_samples)
        samples = samples.difference(samples_i)
        folds.append(samples_i)
        if num_samples > len(samples):
            break
    return folds


def get_clf(key, data=None, num_features=None):
    assert key in SUPPORTED_CLFS
    if key == "AdaBoost":
        return AdaBoostClassifier()
    elif key == "RandomForest":
        return RandomForestClassifier()
    elif key == "FeedForwardNet":
        net = models.FeedForwardNet(num_features)
        train_data = ds.ProposalDataset(data["inputs"], data["labels"])
    elif key == "ConvNet":
        net = models.ConvNet()
        models.init_weights(net)
        train_data = ds.ImgProposalDataset(
            data["inputs"], data["labels"], transform=True
        )
    elif key == "MultiModalNet":
        net = models.MultiModalNet(num_features)
        models.init_weights(net)
        train_data = ds.MultiModalDataset(
            data["inputs"], data["labels"], transform=True
        )
    return net, train_data


def train_network(
    net,
    dataset,
    logger=True,
    lr=10e-3,
    max_epochs=100,
    model_summary=True,
    profile=False,
    progress_bar=True,
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
        enable_model_summary=model_summary,
        enable_progress_bar=progress_bar,
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


# Lightning Module
class LitNeuralNet(pl.LightningModule):
    def __init__(self, net=None, lr=10e-3):
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
