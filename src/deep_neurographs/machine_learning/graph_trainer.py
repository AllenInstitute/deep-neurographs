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
from torch_geometric.utils import subgraph

from deep_neurographs.machine_learning import ml_utils
from deep_neurographs.machine_learning.gnn_utils import toCPU, toGPU

# Training
LR = 1e-3
N_EPOCHS = 200
SCHEDULER_GAMMA = 0.5
SCHEDULER_STEP_SIZE = 1000
TEST_PERCENT = 0.15
WEIGHT_DECAY = 1e-3

# Augmentation
MAX_PROPOSAL_DROPOUT = 0.1
SCALING_FACTOR = 0.05


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
        max_proposal_dropout=MAX_PROPOSAL_DROPOUT,
        scaling_factor=SCALING_FACTOR,
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
        self.model = model.to("cuda:0")
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.init_scheduler()
        self.writer = SummaryWriter()

        # Augmentation
        self.scaling_factor = scaling_factor
        self.max_proposal_dropout = max_proposal_dropout

    def init_scheduler(self):
        self.scheduler = StepLR(
            self.optimizer,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA,
        )

    def run_on_graphs(self, datasets, augment=False):
        """
        Trains a graph neural network in the case where "datasets" is a
        dictionary of datasets such that each corresponds to a distinct graph.

        Parameters
        ----------
        datasets : dict
            Dictionary where each key is a graph id and the value is the
            corresponding graph dataset.

        Returns
        -------
        model : torch.nn.Module
            Graph neural network that has been fit onto "datasets".

        """
        # Initializations
        best_score = -np.inf
        best_ckpt = None

        # Main
        train_ids, test_ids = train_test_split(list(datasets.keys()))
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            self.model.train()
            for graph_id in train_ids:
                y_i, hat_y_i = self.train(
                    datasets[graph_id], epoch, augment=augment
                )
                y.extend(toCPU(y_i))
                hat_y.extend(toCPU(hat_y_i))
            self.compute_metrics(y, hat_y, "train", epoch)
            self.scheduler.step()

            # Test
            if epoch % 10 == 0:
                y, hat_y = [], []
                self.model.eval()
                for graph_id in test_ids:
                    y_i, hat_y_i = self.forward(datasets[graph_id].data)
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
        Trains a graph neural network in the case where "dataset" is a
        graph that may contain multiple connected components.

        Parameters
        ----------
        dataset : dict
            Dictionary where each key is a graph id and the value is the
            corresponding graph dataset.

        Returns
        -------
        None

        """
        pass

    def train(self, dataset, epoch, augment=False):
        """
        Performs the forward pass and backpropagation to update the model's
        weights.

        Parameters
        ----------
        data : GraphDataset
            Graph dataset that corresponds to a single connected component.
        epoch : int
            Current epoch.
        augment : bool, optional
            Indication of whether to augment data. Default is False.

        Returns
        -------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.

        """
        # Data augmentation (if applicable)
        if self.augment:
            data = self.augment(dataset)
        else:
            data = deepcopy(dataset.data)

        # Forward
        y, hat_y = self.forward(data)
        self.backpropagate(y, hat_y, epoch)
        return y, hat_y

    def augment(self, dataset):
        augmented_data = rescale_data(dataset, self.scaling_factor)
        # augmented_data = proposal_dropout(data, self.max_proposal_dropout)
        return augmented_data

    def forward(self, data):
        """
        Runs "data" through "self.model" to generate a prediction.

        Parameters
        ----------
        data : GraphDataset
            Graph dataset that corresponds to a single connected component.

        Returns
        -------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.

        """
        self.optimizer.zero_grad()
        x, edge_index = toGPU(data)
        hat_y = self.model(x, edge_index)
        y = data.y.to("cuda:0", dtype=torch.float32)
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
    test_ids = ["block_000", "block_002"]  # sample(graph_ids, n_test_examples)
    train_ids = list(set(graph_ids) - set(test_ids))
    return train_ids, test_ids


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


def connected_components(data):
    cc_list = []
    cc_idxs = torch.unique(data.edge_index[0], return_inverse=True)[1]
    for i in range(cc_idxs.max().item() + 1):
        cc_list.append(torch.nonzero(cc_idxs == i, as_tuple=False).view(-1))
    return cc_list


def rescale_data(dataset, scaling_factor):
    # Get scaling factor
    low = 1.0 - scaling_factor
    high = 1.0 + scaling_factor
    scaling_factor = torch.tensor(np.random.uniform(low=low, high=high))

    # Rescale
    n = count_proposals(dataset)
    data = deepcopy(dataset.data)
    data.x[0:n, 1] = scaling_factor * data.x[0:n, 1]
    return data


def proposal_dropout(data, max_proposal_dropout):
    n_dropout_edges = len(data.dropout_edges) // 2
    dropout_prob = np.random.uniform(low=0, high=max_proposal_dropout)
    n_remove = int(dropout_prob * n_dropout_edges)
    remove_edges = sample(data.dropout_edges, n_remove)
    for edge in remove_edges:
        reversed_edge = [edge[1], edge[0]]
        edges_to_remove = torch.tensor([edge, reversed_edge], dtype=torch.long)
        edges_mask = torch.all(
            data.data.edge_index.T == edges_to_remove[:, None], dim=2
        ).any(dim=0)
        data.data.edge_index = data.data.edge_index[:, ~edges_mask]
    return data


def count_proposals(dataset):
    return dataset.data.y.size(0)
