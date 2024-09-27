"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training heterogeneous graph neural networks that classify
edge proposals.

"""

from copy import deepcopy
from random import shuffle

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from deep_neurographs.utils import gnn_util, ml_util
from deep_neurographs.utils.gnn_util import toCPU

LR = 1e-3
N_EPOCHS = 200
SCHEDULER_GAMMA = 0.5
SCHEDULER_STEP_SIZE = 1000
WEIGHT_DECAY = 1e-3


class Trainer:
    """
    Custom class that trains graph neural networks.

    """

    def __init__(
        self,
        model,
        criterion,
        lr=LR,
        n_epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
    ):
        """
        Constructs a GraphTrainer object.

        Parameters
        ----------
        model : torch.nn.Module
            Graph neural network.
        criterion : torch.nn.Module._Loss
            Loss function.
        lr : float, optional
            Learning rate. The default is the global variable LR.
        n_epochs : int
            Number of epochs. The default is the global variable N_EPOCHS.
        weight_decay : float
            Weight decay used in optimizer. The default is the global variable
            WEIGHT_DECAY.

        Returns
        -------
        None.

        """
        # Training
        self.model = model  # .to("cuda:0")
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.init_scheduler()
        self.writer = SummaryWriter()

    def init_scheduler(self):
        self.scheduler = StepLR(
            self.optimizer,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA,
        )

    def run(self, train_dataset_list, validation_dataset_list):
        """
        Trains a graph neural network in the case where "datasets" is a
        dictionary of datasets such that each corresponds to a distinct graph.

        Parameters
        ----------
        ...

        Returns
        -------
        torch.nn.Module
            Graph neural network that has been fit onto "datasets".

        """
        best_score = -np.inf
        best_ckpt = None
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            self.model.train()
            for dataset in train_dataset_list:
                # Forward pass
                hat_y_i, y_i = self.predict(dataset.data)
                loss = self.criterion(hat_y_i, y_i)
                self.writer.add_scalar("loss", loss, epoch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store predictions
                y.extend(toCPU(y_i))
                hat_y.extend(toCPU(hat_y_i))

            self.compute_metrics(y, hat_y, "train", epoch)
            self.scheduler.step()

            # Validate
            if epoch % 10 == 0:
                y, hat_y = [], []
                self.model.eval()
                for dataset in validation_dataset_list:
                    hat_y_i, y_i = self.predict(dataset.data)
                    y.extend(toCPU(y_i))
                    hat_y.extend(toCPU(hat_y_i))
                test_score = self.compute_metrics(y, hat_y, "val", epoch)

                # Check for best
                if test_score > best_score:
                    best_score = test_score
                    best_ckpt = deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_ckpt)
        return self.model

    def predict(self, data):
        """
        Runs "data" through "self.model" to generate a prediction.

        Parameters
        ----------
        data : GraphDataset
            Graph dataset that corresponds to a single connected component.

        Returns
        -------
        torch.Tensor
            Ground truth.
        torch.Tensor
            Prediction.

        """
        # Run model
        x_dict, edge_index_dict, edge_attr_dict = gnn_util.get_inputs(
            data, "HeteroGNN"
        )
        hat_y = self.model(x_dict, edge_index_dict, edge_attr_dict)

        # Output
        y = data["proposal"]["y"]
        return truncate(hat_y, y), y

    def compute_metrics(self, y, hat_y, prefix, epoch):
        """
        Computes and logs evaluation metrics for binary classification.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.
        prefix : str
            Prefix to be added to the metric names when logging.
        epoch : int
            Current epoch.

        Returns
        -------
        float
            F1 score.

        """
        # Initializations
        y = np.array(y, dtype=int).tolist()
        hat_y = get_predictions(hat_y)

        # Compute
        accuracy = accuracy_score(y, hat_y)
        accuracy_dif = accuracy - np.sum(y) / len(y)
        precision = precision_score(y, hat_y)
        recall = recall_score(y, hat_y)
        f1 = f1_score(y, hat_y)

        # Log
        self.writer.add_scalar(prefix + "_accuracy:", accuracy, epoch)
        self.writer.add_scalar(prefix + "_accuracy_df:", accuracy_dif, epoch)
        self.writer.add_scalar(prefix + "_precision:", precision, epoch)
        self.writer.add_scalar(prefix + "_recall:", recall, epoch)
        self.writer.add_scalar(prefix + "_f1:", f1, epoch)
        return f1


# -- util --
def shuffler(my_list):
    """
    Shuffles a list of items.

    Parameters
    ----------
    my_list : list
        List to be shuffled.

    Returns
    -------
    list
        Shuffled list.

    """
    shuffle(my_list)
    return my_list


def truncate(hat_y, y):
    """
    Truncates "hat_y" so that this tensor has the same shape as "y". Note this
    operation removes the predictions corresponding to branches so that loss
    is computed over proposals.

    Parameters
    ----------
    hat_y : torch.Tensor
        Tensor to be truncated.
    y : torch.Tensor
        Tensor used as a reference.

    Returns
    -------
    torch.Tensor
        Truncated "hat_y".

    """
    return hat_y[: y.size(0), 0]


def get_predictions(hat_y, threshold=0.5):
    """
    Generate binary predictions based on the input probabilities.

    Parameters
    ----------
    hat_y : torch.Tensor
        Predicted probabilities generated by "self.model".
    threshold : float, optional
        The threshold value for binary classification. The default is 0.5.

    Returns
    -------
    list[int]
        Binary predictions based on the given threshold.

    """
    return (ml_util.sigmoid(np.array(hat_y)) > threshold).tolist()
