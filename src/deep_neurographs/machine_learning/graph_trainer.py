"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training graph neural networks that classify edge proposals.

"""

from random import sample, shuffle
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.functional import sigmoid
from torch.utils.tensorboard import SummaryWriter


LR = 1e-3
N_EPOCHS = 1000
TEST_PERCENT = 0.15
WEIGHT_DECAY = 5e-4


class GraphTrainer:
    def __init__(
        self,
        model,
        criterion,
        lr=LR,
        n_epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
    ):
        self.model = model.to("cuda:0")
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.writer = SummaryWriter()

    def run_on_graphs(self, graph_datasets):
        # Initializations
        best_score = -np.inf
        best_ckpt = None

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
            train_score = self.compute_metrics(y, hat_y, "train", epoch)

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
        return self.model.load_state_dict(best_ckpt)

    def train(self, graph_data, epoch):
        y, hat_y = self.forward(graph_data)
        self.backpropagate(y, hat_y, epoch)
        return y, hat_y

    def forward(self, graph_data):
        self.optimizer.zero_grad()
        x, y, edge_index = toGPU(graph_data)
        hat_y = self.model(x, edge_index)
        return y, truncate(hat_y, y)

    def backpropagate(self, y, hat_y, epoch):
        loss = self.criterion(hat_y, y)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("loss", loss, epoch)

    def compute_metrics(self, y, hat_y, prefix, epoch):
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
        self.writer.add_scalar(prefix + '_accuracy:', accuracy, epoch)
        self.writer.add_scalar(prefix + '_accuracy_df:', accuracy_dif, epoch)
        self.writer.add_scalar(prefix + '_precision:', precision, epoch)
        self.writer.add_scalar(prefix + '_recall:', recall, epoch)
        self.writer.add_scalar(prefix + '_f1:', f1, epoch)
        return accuracy_dif


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
    n_test_examples = 1  # int(len(graph_ids) * TEST_PERCENT)
    test_ids = sample(graph_ids, n_test_examples)
    train_ids = list(set(graph_ids) - set(test_ids))
    return train_ids, test_ids


def toCPU(tensor):
    return np.array(tensor.detach().cpu()).tolist()


def toGPU(graph_data):
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
    return hat_y[0: y.size(0), 0]


def get_predictions(hat_y, threshold=0.5):
    return (sigmoid(np.array(hat_y)) > threshold).tolist()


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
