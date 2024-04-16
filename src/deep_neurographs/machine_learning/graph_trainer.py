"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training graph neural networks that classify edge proposals.

"""

from copy import deepcopy
from random import sample, shuffle

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

from deep_neurographs.machine_learning import ml_utils

LR = 1e-3
N_EPOCHS = 1000
TEST_PERCENT = 0.15
WEIGHT_DECAY = 5e-3


class GraphTrainer:
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
        self.model = model.to("cuda:0")
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.writer = SummaryWriter()

    def run_on_graphs(self, graph_datasets):
        """
        Trains a graph neural network in the case where "graph_datasets" is a
        dictionary of datasets such that each corresponds to a distinct graph.

        Parameters
        ----------
        graph_datasets : dict
            Dictionary where each key is a graph id and the value is the
            corresponding graph dataset.

        Returns
        -------
        model : torch.nn.Module
            Graph neural network that has been fit onto "graph_datasets".

        """
        # Initializations
        best_score = -np.inf
        best_ckpt = None
        scheduler = StepLR(self.optimizer, step_size=500, gamma=0.5)

        # Main
        train_ids, test_ids = train_test_split(list(graph_datasets.keys()))
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            self.model.train()
            for graph_id in train_ids:
                y_i, hat_y_i = self.train(graph_datasets[graph_id].data, epoch)
                y.extend(toCPU(y_i))
                hat_y.extend(toCPU(hat_y_i))
            self.compute_metrics(y, hat_y, "train", epoch)
            scheduler.step()

            # Test
            if epoch % 10 == 0:
                y, hat_y = [], []
                self.model.eval()
                for graph_id in test_ids:
                    y_i, hat_y_i = self.forward(graph_datasets[graph_id].data)
                    y.extend(toCPU(y_i))
                    hat_y.extend(toCPU(hat_y_i))
                test_score = self.compute_metrics(y, hat_y, "val", epoch)

                # Check for best
                if test_score > best_score:
                    best_score = test_score
                    best_ckpt = deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_ckpt)
        return self.model

    def run_on_graph(self):
        """
        Trains a graph neural network in the case where "graph_dataset" is a
        graph that may contain multiple connected components.

        Parameters
        ----------
        graph_dataset : dict
            Dictionary where each key is a graph id and the value is the
            corresponding graph dataset.

        Returns
        -------
        None

        """
        pass

    def train(self, graph_data, epoch):
        """
        Performs the forward pass and backpropagation to update the model's
        weights.

        Parameters
        ----------
        graph_data : GraphDataset
            Graph dataset that corresponds to a single connected component.
        epoch : int
            Current epoch.

        Returns
        -------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.

        """
        y, hat_y = self.forward(graph_data)
        self.backpropagate(y, hat_y, epoch)
        return y, hat_y

    def forward(self, graph_data):
        """
        Runs "graph_data" through "self.model" to generate a prediction.

        Parameters
        ----------
        graph_data : GraphDataset
            Graph dataset that corresponds to a single connected component.

        Returns
        -------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.

        """
        self.optimizer.zero_grad()
        x, y, edge_index = toGPU(graph_data)
        hat_y = self.model(x, edge_index)
        return y, truncate(hat_y, y)

    def backpropagate(self, y, hat_y, epoch):
        """
        Runs backpropagation to update the model's weights.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.
        epoch : int
            Current epoch.

        Returns
        -------
        None

        """
        loss = self.criterion(hat_y, y)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("loss", loss, epoch)

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
        f1 : float
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


# -- utils --
def shuffler(my_list):
    """
    Shuffles a list of items.

    Parameters
    ----------
    my_list : list
        List to be shuffled.

    Returns
    -------
    my_list : list
        Shuffled list.

    """
    shuffle(my_list)
    return my_list


def train_test_split(graph_ids):
    """
    Split a list of graph IDs into training and testing sets.

    Parameters
    ----------
    graph_ids : list[str]
        A list containing unique identifiers (IDs) for graphs.

    Returns
    -------
    train_ids : list
        A list containing IDs for the training set.
    test_ids : list
        A list containing IDs for the testing set.

    """
    n_test_examples = int(len(graph_ids) * TEST_PERCENT)
    test_ids = ["block_007", "block_010"]  # sample(graph_ids, n_test_examples)
    train_ids = list(set(graph_ids) - set(test_ids))
    return train_ids, test_ids


def toCPU(tensor):
    """
    Moves "tensor" from GPU to CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Dataset to be moved to GPU.

    Returns
    -------
    numpy.ndarray
        Array.

    """
    return np.array(tensor.detach().cpu()).tolist()


def toGPU(graph_data):
    """
    Moves "graph_data" from CPU to GPU.

    Parameters
    ----------
    graph_data : GraphDataset
        Dataset to be moved to GPU.

    Returns
    -------
    x : torch.Tensor
        Matrix of node feature vectors.
    y : torch.Tensor
        Ground truth.
    edge_idx : torch.Tensor
        Tensor containing edges in graph.

    """
    x = graph_data.x.to("cuda:0", dtype=torch.float32)
    y = graph_data.y.to("cuda:0", dtype=torch.float32)
    edge_index = graph_data.edge_index.to("cuda:0")
    return x, y, edge_index


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
    return (ml_utils.sigmoid(np.array(hat_y)) > threshold).tolist()
