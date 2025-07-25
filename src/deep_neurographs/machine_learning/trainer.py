"""
Created on Wed July 25 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for a custom class for training neural networks to perform classification
tasks within the GraphTrace pipeline.

"""

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from deep_neurographs.utils import ml_util, util


class Trainer:
    """
    Trainer class for training a model to perform binary classifcation.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch during training.
    best_f1 : float
        Best F1 score achieved so far on valiation dataset.
    criterion : torch.nn.BCEWithLogitsLoss
        Loss function used during training.
    log_dir : str
        Path to directory that tensorboard and checkpoints are saved to.
    max_epochs : int
        Maximum number of training epochs.
    model : torch.nn.Module
        Model that is trained to perform binary classification.
    model_name : str
        Name of model used for logging and checkpointing.
    optimizer : torch.optim.AdamW
        Optimizer that is used during training.
    scheduler : torch.optim.lr_scheduler.CosineAnnealingLR
        Scheduler used to the adjust learning rate.
    writer : torch.utils.tensorboard.SummaryWriter
        Writer object that writes to a tensorboard.
    """

    def __init__(
        self,
        model,
        model_name,
        output_dir,
        batch_size=32,
        lr=1e-3,
        max_epochs=200,
    ):
        """
        Instantiates a Trainer object.

        Parameters
        ----------
        model : torch.nn.Module
            Model that is trained to perform binary classification.
        model_name : str
            Name of model used for logging and checkpointing.
        output_dir : str
            Directory that tensorboard and model checkpoints are written to.
        batch_size : int, optional
            Number of samples per batch during training. Default is 32.
        lr : float
            Learning rate.
        max_epochs : int
            Maximum number of training epochs.
        """
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.best_f1 = 0
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.model_name = model_name

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = model.to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

    # --- Core Routines ---
    def run(self, train_dataloader, val_dataloader):
        """
        Run the full training and validation loop.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Dataloader used for training.
        val_dataset : torch.utils.data.Dataset
            Dataloader used for validation.

        Returns
        -------
        None
        """
        exp_name = os.path.basename(os.path.normpath(self.log_dir))
        print("\nExperiment:", exp_name)
        for epoch in range(self.max_epochs):
            # Train-Validate
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats, new_best = self.validate_step(val_dataloader, epoch)

            # Report reuslts
            print(f"\nEpoch {epoch}: " + ("New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Perform a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        dict
            Dictionary of aggregated training metrics.
        """
        self.model.train()
        loss, y, hat_y = list(), list(), list()
        for x_i, y_i in train_dataloader:
            # Forward pass
            hat_y_i, loss_i = self.forward_pass(x_i, y_i)

            # Backward pass
            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            # Store results
            y.extend(ml_util.to_cpu(y_i, True).flatten().tolist())
            hat_y.extend(ml_util.to_cpu(hat_y_i, True).flatten().tolist())
            loss.append(float(ml_util.to_cpu(loss_i)))

        # Write stats to tensorboard
        stats = self.compute_stats(y, hat_y)
        stats["loss"] = np.mean(loss)
        self.update_tensorboard(stats, epoch, "train_")
        return stats

    def validate_step(self, val_dataloader, epoch):
        """
        Perform a full validation loop over the given dataloader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        tuple
            stats : dict
                Dictionary of aggregated validation metrics.
            is_best : bool
                True if the current F1 score is the best so far.
        """
        loss, y, hat_y = list(), list(), list()
        with torch.no_grad():
            self.model.eval()
            for x_i, y_i in val_dataloader:
                # Run model
                hat_y_i, loss_i = self.forward_pass(x_i, y_i)

                # Store results
                y.extend(ml_util.to_cpu(y_i, True).flatten().tolist())
                hat_y.extend(ml_util.to_cpu(hat_y_i, True).flatten().tolist())
                loss.append(float(ml_util.to_cpu(loss_i)))

        # Write stats to tensorboard
        stats = self.compute_stats(y, hat_y)
        stats["loss"] = np.mean(loss)
        self.update_tensorboard(stats, epoch, "val_")

        # Check for new best
        if stats["f1"] > self.best_f1:
            self.save_model(epoch)
            self.best_f1 = stats["f1"]
            return stats, True
        else:
            return stats, False

    def forward_pass(self, x, y):
        """
        Perform a forward pass through the model and compute loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, C, D, H, W).

        Returns
        -------
        tuple
            hat_y : torch.Tensor
                Model predictions.
            loss : torch.Tensor
                Computed loss value.
        """
        x = x.to("cuda", dtype=torch.float32)
        y = y.to("cuda", dtype=torch.float32)
        hat_y = self.model(x)
        loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers
    def compute_stats(self, y, hat_y):
        """
        Compute F1 score, precision, and recall for each sample in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels of shape (B, 1, D, H, W) or (B, 1, H, W).
        hat_y : torch.Tensor
            Model predictions of the same shape as ground truth.

        Returns
        -------
        dict
            Dictionary containing lists of per-sample metrics.
        """
        # Reformat predictions
        hat_y = (np.array(hat_y) > 0).astype(int)
        y = np.array(y, dtype=int)

        # Compute stats
        avg_prec = precision_score(y, hat_y, zero_division=np.nan)
        avg_recall = recall_score(y, hat_y, zero_division=np.nan)
        avg_f1 = 2 * avg_prec * avg_recall / max((avg_prec + avg_recall), 1)
        avg_acc = accuracy_score(y, hat_y)
        stats = {
            "f1": avg_f1,
            "precision": avg_prec,
            "recall": avg_recall,
            "accuracy": avg_acc
        }
        return stats

    def report_stats(self, stats, is_train=True):
        """
        Print a summary of training or validation statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names to values.
        is_train : bool, optional
            Indication of whether stats were computed during training.

        Returns
        -------
        None
        """
        summary = "   Train: " if is_train else "   Val: "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def save_model(self, epoch):
        """
        Save the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        None
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"{self.model_name}-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        """
        Log scalar statistics to TensorBoard.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names (str) to lists of values.
        epoch : int
            Current training epoch.
        prefix : str
            Prefix to prepend to each metric name when logging.

        Returns
        -------
        None
        """
        for key, value in stats.items():
            self.writer.add_scalar(prefix + key, stats[key], epoch)
